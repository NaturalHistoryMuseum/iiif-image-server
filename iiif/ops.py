from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from contextlib import suppress

from iiif.exceptions import invalid_iiif_parameter
from iiif.profiles.base import ImageInfo

# this server currently supports IIIF level1
IIIF_LEVEL = 'level1'


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    full: bool = False

    def __iter__(self):
        yield from [self.x, self.y, self.w, self.h]

    def __str__(self):
        if self.full:
            return 'full'
        else:
            return f'{self.x}_{self.y}_{self.w}_{self.h}'


def parse_region(region: str, info: ImageInfo) -> Region:
    """
    Given a region parameter, parse it into a Region object. If the region parameter is invalid, an
    exception is thrown.

    This function is IIIF level 2 compliant: https://iiif.io/api/image/3.0/compliance/#31-region.

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
                parts[0] = (float(parts[0][4:]) / 100) * info.width
                parts[1] = (float(parts[1]) / 100) * info.height
                parts[2] = (float(parts[2]) / 100) * info.width
                parts[3] = (float(parts[3]) / 100) * info.height

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
    raise invalid_iiif_parameter('Region', region)


@dataclass
class Size:
    w: int
    h: int
    max: bool = False

    def __str__(self):
        if self.max:
            return 'max'
        else:
            return f'{self.w}_{self.h}'


# TODO: make this level 2 compliant
def parse_size(size: str, region: Region) -> Size:
    """
    Given a size parameter, parse it into a Size object. If the size parameter is invalid, an
    exception is thrown.

    This function is IIIF level 1 compliant: https://iiif.io/api/image/3.0/compliance/#32-size.

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

    raise invalid_iiif_parameter('Size', size)


@dataclass
class Rotation:
    # jpegtran only supports rotating in 90 degree increments and IIIF suggests only supporting
    # rotating by 90 unless the png format is supported so that the background can be made
    # transparent
    angle: int
    mirror: bool = False

    def __str__(self):
        if self.mirror:
            return f'-{self.angle}'
        else:
            return str(self.angle)


allowed_angles = {0, 90, 180, 270}


def parse_rotation(rotation: str) -> Rotation:
    """
    Given a rotation parameter, parse it into a Rotation object. If the rotation parameter is
    invalid, an exception is thrown.

    This function is IIIF level 2 compliant: https://iiif.io/api/image/3.0/compliance/#33-rotation.

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
        if angle in allowed_angles:
            return Rotation(angle, mirror)

    raise invalid_iiif_parameter('Rotation', rotation)


class Quality(Enum):
    default = 'default'
    color = 'color'
    gray = 'gray'
    bitonal = 'bitonal'


def parse_quality(quality: str) -> Quality:
    """
    Given a quality parameter, parse it into a Quality object. If the quality parameter is invalid,
    an exception is thrown.

    This function is IIIF level 2 compliant: https://iiif.io/api/image/3.0/compliance/#34-quality.

    :param quality: the quality param string
    :return: a Quality
    """
    for option in Quality:
        if option.value == quality:
            return option

    raise invalid_iiif_parameter('Quality', quality)


class Format(Enum):
    jpg = 'jpg'
    png = 'png'


def parse_format(fmt: str) -> Format:
    """
    Given a format parameter, parse it into a Format object. If the quality parameter is invalid,
    an exception is thrown.

    This function is IIIF level 2 compliant: https://iiif.io/api/image/3.0/compliance/#35-format.

    :param fmt: the format param string
    :return: a Format
    """
    for option in Format:
        if option.value == fmt:
            return option

    raise invalid_iiif_parameter('Format', fmt)


def parse_params(info: ImageInfo, region: str = 'full', size: str = 'max', rotation: str = '0',
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
    parsed_region = parse_region(region, info)
    return IIIFOps(
        parsed_region,
        parse_size(size, parsed_region),
        parse_rotation(rotation),
        parse_quality(quality),
        parse_format(fmt)
    )


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
                    f'{self.quality.value}.{self.format.value}')
