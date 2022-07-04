from PIL import ImageOps
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from jpegtran import JPEGImage
from jpegtran.lib import Transformation
from pathlib import Path
from typing import List

from iiif.exceptions import InvalidIIIFParameter
from iiif.profiles.base import ImageInfo
from iiif.utils import to_pillow, to_jpegtran, disable_bomb_errors

# this server currently supports IIIF level2
IIIF_LEVEL = 2


@dataclass
class Region:
    """
    This op is IIIF level 2 compliant: https://iiif.io/api/image/3.0/compliance/#31-region.
    """
    x: int
    y: int
    w: int
    h: int
    full: bool = False

    @property
    def ratio(self):
        return self.w / self.h

    def __iter__(self):
        yield from [self.x, self.y, self.w, self.h]

    def __str__(self):
        if self.full:
            return 'full'
        else:
            return f'{self.x}_{self.y}_{self.w}_{self.h}'

    @staticmethod
    def parse(region: str, info: ImageInfo) -> 'Region':
        """
        Given a region parameter, parse it into a Region object. If the region parameter is invalid,
        an exception is thrown.

        :param region: the region param string
        :param info: the ImageInfo object
        :return: a Region
        """
        if region == 'full':
            return Region(0, 0, info.width, info.height, full=True)

        if region == 'square':
            if info.width < info.height:
                # the image is portrait, we need to crop out a centre square the size of the width
                return Region(0, round((info.height - info.width) / 2), info.width, info.width)
            elif info.width > info.height:
                # the image is landscape, we need to crop out a centre square the size of the height
                return Region(round((info.width - info.height) / 2), 0, info.height, info.height)
            else:
                # the image is already square, return the whole thing
                return Region(0, 0, info.width, info.height, full=True)

        parts = region.split(',')
        if len(parts) == 4:
            with suppress(ValueError):
                if parts[0].startswith('pct:'):
                    # convert the percentages to actual float x, y, w, and h values
                    parts = [
                        (float(parts[0][4:]) / 100) * info.width,
                        (float(parts[1]) / 100) * info.height,
                        (float(parts[2]) / 100) * info.width,
                        (float(parts[3]) / 100) * info.height,
                    ]

                # use round rather than int as it's a bit more intuitive for users
                x, y, w, h = map(round, map(float, parts))

                # check the basics of the region are allowable
                if 0 <= x < info.width and 0 <= y < info.height and w > 0 and h > 0:
                    # now bring any out of bounds values back in bounds
                    if x + w > info.width:
                        w = info.width - x
                    if y + h > info.height:
                        h = info.height - y
                    return Region(x, y, w, h, full=(w == info.width and h == info.height))

        # if we get here, the region is no good :(
        raise InvalidIIIFParameter('Region', region)

    def process(self, image: JPEGImage) -> JPEGImage:
        """
        Processes a IIIF region parameter which essentially involves cropping the image.

        :param image: a jpegtran JPEGImage object
        :return: a jpegtran JPEGImage object
        """
        if self.full:
            # no crop required!
            return image

        # jpegtran can't handle crops that don't have an origin divisible by 16 therefore we're
        # going to do the crop in pillow, however, we're going to crop down to the next lowest
        # number below each of x and y that is divisible by 16 using jpegtran and then crop off the
        # remaining pixels in pillow to get to the desired size. This is all for performance,
        # jpegtran is so much quicker than pillow hence it's worth this hassle
        if self.x % 16 or self.y % 16:
            # work out how far we need to shift the x and y to get to the next lowest numbers
            # divisible by 16
            x_shift = self.x % 16
            y_shift = self.y % 16
            # crop the image using the shifted x and y and the shifted width and height
            image = image.crop(self.x - x_shift, self.y - y_shift,
                               self.w + x_shift, self.h + y_shift)
            # now shift over to pillow for the final crop
            pillow_image = to_pillow(image)
            # do the final crop to get us the desired size
            pillow_image = pillow_image.crop((x_shift, y_shift, x_shift + self.w, y_shift + self.h))
            return to_jpegtran(pillow_image)
        else:
            # if the crop has an origin divisible by 16 then we can just use jpegtran directly
            return image.crop(*self)


@dataclass
class Size:
    """
    This op is IIIF level 2 compliant: https://iiif.io/api/image/3.0/compliance/#32-size.
    """
    w: int
    h: int
    max: bool = False

    def __str__(self):
        if self.max:
            return 'max'
        else:
            return f'{self.w}_{self.h}'

    @staticmethod
    def parse(size: str, region: Region) -> 'Size':
        """
        Given a size parameter, parse it into a Size object. If the size parameter is invalid, an
        exception is thrown.

        :param size: the size param string
        :param region: the parsed Region object
        :return: a Size
        """
        if size == 'max':
            return Size(region.w, region.h, max=True)

        w = None
        h = None

        parts = size.split(',')
        if len(parts) == 1:
            if parts[0].startswith('pct:'):
                with suppress(ValueError):
                    percentage = float(parts[0][4:]) / 100
                    w = region.w * percentage
                    h = region.h * percentage
        elif len(parts) == 2:
            with suppress(ValueError):
                if size.startswith('!'):
                    confined_w = float(parts[0][1:])
                    confined_h = float(parts[1])
                    # try using the confined width first
                    w = confined_w
                    h = confined_w / region.ratio
                    if h > confined_h:
                        # if the result exceeds the confined height, confine by height instead
                        w = confined_h * region.ratio
                        h = confined_h
                else:
                    if any(parts):
                        w, h = (float(part) if part != '' else part for part in parts)
                        if h == '':
                            h = region.h * w / region.w
                        elif w == '':
                            w = region.w * h / region.h

        if w and h:
            w = round(w)
            h = round(h)
            if 0 < w <= region.w and 0 < h <= region.h:
                return Size(w, h, max=(w == region.w and h == region.h))

        raise InvalidIIIFParameter('Size', size)

    def process(self, image: JPEGImage) -> JPEGImage:
        """
        Processes a IIIF size parameter which essentially involves resizing the image.

        :param image: a jpegtran JPEGImage object
        :return: a jpegtran JPEGImage object
        """
        if self.max:
            return image
        return image.downscale(self.w, self.h)


@dataclass
class Rotation:
    """
    This op is IIIF level 2 compliant: https://iiif.io/api/image/3.0/compliance/#33-rotation.
    """
    # IIIF suggests only supporting rotating by 90 unless the png format is supported so that the
    # background can be made transparent
    angle: int
    mirror: bool = False

    def __str__(self):
        if self.mirror:
            return f'-{self.angle}'
        else:
            return str(self.angle)

    @staticmethod
    def parse(rotation: str) -> 'Rotation':
        """
        Given a rotation parameter, parse it into a Rotation object. If the rotation parameter is
        invalid, an exception is thrown.

        :param rotation: the rotation param string
        :return: a Rotation
        """
        if rotation.startswith('!'):
            mirror = True
            rotation = rotation[1:]
        else:
            mirror = False

        with suppress(ValueError):
            angle = int(rotation)
            if angle in {0, 90, 180, 270}:
                return Rotation(angle, mirror)

        raise InvalidIIIFParameter('Rotation', rotation)

    def process(self, image: JPEGImage) -> JPEGImage:
        """
        Processes a IIIF rotation parameter which can involve mirroring and/or rotating. We could do
        this in jpegtran but only if the width and height were divisible by 16 so we'll just do it
        in pillow for ease.

        :param image: a jpegtran JPEGImage object
        :return: a jpegtran JPEGImage object
        """
        pillow_image = to_pillow(image)
        if self.mirror:
            pillow_image = ImageOps.mirror(pillow_image)
        if self.angle > 0:
            pillow_image = pillow_image.rotate(-self.angle, expand=True)
        return to_jpegtran(pillow_image)


class Quality(Enum):
    """
    This op is IIIF level 2 compliant: https://iiif.io/api/image/3.0/compliance/#34-quality.
    """
    default = ('default',)
    color = ('color', 'colour')
    gray = ('gray', 'grey')
    bitonal = ('bitonal',)

    @staticmethod
    def extras() -> List[str]:
        """
        Returns the values that should be use in the info.json response. This should include
        eveything except the default value.

        :return: a list of extra qualities available on this IIIF server
        """
        return list(chain.from_iterable(
            quality.value for quality in Quality if quality != Quality.default)
        )

    def __str__(self) -> str:
        return self.value[0]

    @staticmethod
    def parse(quality: str) -> 'Quality':
        """
        Given a quality parameter, parse it into a Quality object. If the quality parameter is
        invalid, an exception is thrown.

        :param quality: the quality param string
        :return: a Quality
        """
        for option in Quality:
            if quality in option.value:
                return option

        raise InvalidIIIFParameter('Quality', quality)

    def process(self, image: JPEGImage) -> JPEGImage:
        """
        Processes a IIIF quality parameter. This usually results in the same image being returned as
        was passed in but can involve conversion to grayscale or pure black and white, single bit
        encoded images.

        :param image: a jpegtran JPEGImage object
        :return: a jpegtran JPEGImage object
        """
        if self == Quality.default or self == Quality.color:
            return image
        if self == Quality.gray:
            # not sure why the grayscale function isn't exposed on the JPEGImage class but heyho,
            # this does the same thing
            return JPEGImage(blob=Transformation(image.as_blob()).grayscale())
        if self == Quality.bitonal:
            # convert the image to just black and white using pillow
            return to_jpegtran(to_pillow(image).convert('1'))


class Format(Enum):
    """
    This op is IIIF level 2 compliant: https://iiif.io/api/image/3.0/compliance/#35-format.
    """
    jpg = 'jpg'
    png = 'png'

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def parse(fmt: str) -> 'Format':
        """
        Given a format parameter, parse it into a Format object. If the quality parameter is
        invalid, an exception is thrown.

        :param fmt: the format param string
        :return: a Format
        """
        for option in Format:
            if fmt == option.value:
                return option

        raise InvalidIIIFParameter('Format', fmt)

    def process(self, image: JPEGImage, output_path: Path):
        """
        Processes the IIIF format parameter by writing the image to the output path in the requested
        format.

        :param image: a jpegtran JPEGImage object
        :param output_path: the path to write the file to
        """
        with output_path.open('wb') as f:
            if self == Format.jpg:
                f.write(image.as_blob())
            elif self == Format.png:
                to_pillow(image).save(f, format='png')


@dataclass
class IIIFOps:
    region: Region
    size: Size
    rotation: Rotation
    quality: Quality
    format: Format

    @property
    def location(self) -> Path:
        """
        :return: the path where the image produced by this set of ops should be stored.
        """
        return Path(str(self.region), str(self.size), str(self.rotation),
                    f'{self.quality}.{self.format}')

    @staticmethod
    def parse(info: ImageInfo, region: str = 'full', size: str = 'max', rotation: str = '0',
              quality: str = 'default', fmt: str = 'jpg') -> 'IIIFOps':
        """
        Given a set of IIIF string parameters, parse each and return a IIIFOps object. If any of the
        parameters is invalid then an exception is raised.

        :param info: an ImageInfo object
        :param region: the region parameter
        :param size: the size parameter
        :param rotation: the rotation parameter
        :param quality: the quality parameter
        :param fmt: the format parameter
        :return: a IIIFOps object
        """
        parsed_region = Region.parse(region, info)
        return IIIFOps(
            parsed_region,
            Size.parse(size, parsed_region),
            Rotation.parse(rotation),
            Quality.parse(quality),
            Format.parse(fmt)
        )

    def process(self, source_path: Path, output_path: Path):
        """
        Process a single image at the source path and save it in the output path.

        :param source_path: the source image
        :param output_path: the target location
        """
        # given this is usually run in a separate process, make sure we have disabled bomb errors
        disable_bomb_errors()

        image = JPEGImage(str(source_path))

        # process each op in the right order
        image = self.region.process(image)
        image = self.size.process(image)
        image = self.rotation.process(image)
        image = self.quality.process(image)

        # ensure the full cache path exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # write the processed image to disk
        self.format.process(image, output_path)
