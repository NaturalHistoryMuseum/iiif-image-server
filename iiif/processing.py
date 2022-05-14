#!/usr/bin/env python3
# encoding: utf-8
from concurrent.futures import ProcessPoolExecutor

import asyncio
import os
from PIL import ImageOps
from dataclasses import dataclass
from jpegtran import JPEGImage
from jpegtran.lib import Transformation
from pathlib import Path
from typing import Optional, Tuple

from iiif.ops import IIIFOps, Region, Size, Rotation, Quality, Format
from iiif.profiles.base import ImageInfo, AbstractProfile
from iiif.utils import to_pillow, to_jpegtran, FetchCache, Fetchable


def process_region(image: JPEGImage, region: Region) -> JPEGImage:
    """
    Processes a IIIF region parameter which essentially involves cropping the image.

    :param image: a jpegtran JPEGImage object
    :param region: the IIIF region parameter
    :return: a jpegtran JPEGImage object
    """
    if region.full:
        # no crop required!
        return image

    # jpegtran can't handle crops that don't have an origin divisible by 16 therefore we're going to
    # do the crop in pillow, however, we're going to crop down to the next lowest number below each
    # of x and y that is divisible by 16 using jpegtran and then crop off the remaining pixels in
    # pillow to get to the desired size. This is all for performance, jpegtran is so much quicker
    # than pillow hence it's worth this hassle
    if region.x % 16 or region.y % 16:
        # work out how far we need to shift the x and y to get to the next lowest numbers divisible
        # by 16
        x_shift = region.x % 16
        y_shift = region.y % 16
        # crop the image using the shifted x and y and the shifted width and height
        image = image.crop(region.x - x_shift, region.y - y_shift,
                           region.w + x_shift, region.h + y_shift)
        # now shift over to pillow for the final crop
        pillow_image = to_pillow(image)
        # do the final crop to get us the desired size
        pillow_image = pillow_image.crop((x_shift, y_shift, x_shift + region.w, y_shift + region.h))
        return to_jpegtran(pillow_image)
    else:
        # if the crop has an origin divisible by 16 then we can just use jpegtran directly
        return image.crop(*region)


def process_size(image: JPEGImage, size: Size) -> JPEGImage:
    """
    Processes a IIIF size parameter which essentially involves resizing the image.

    :param image: a jpegtran JPEGImage object
    :param size: the IIIF size parameter
    :return: a jpegtran JPEGImage object
    """
    if size.max:
        return image
    return image.downscale(size.w, size.h)


def process_rotation(image: JPEGImage, rotation: Rotation) -> JPEGImage:
    """
    Processes a IIIF rotation parameter which can involve mirroring and/or rotating. We could do
    this in jpegtran but only if the width and height were divisible by 16 so we'll just do it in
    pillow for ease.

    :param image: a jpegtran JPEGImage object
    :param rotation: the IIIF rotate parameter
    :return: a jpegtran JPEGImage object
    """
    pillow_image = to_pillow(image)
    if rotation.mirror:
        pillow_image = ImageOps.mirror(pillow_image)
    if rotation.angle > 0:
        pillow_image = pillow_image.rotate(-rotation.angle, expand=True)
    return to_jpegtran(pillow_image)


def process_quality(image: JPEGImage, quality: Quality) -> JPEGImage:
    """
    Processes a IIIF quality parameter. This usually results in the same image being returned as was
    passed in but can involve conversion to grayscale or pure black and white, single bit encoded
    images.

    :param image: a jpegtran JPEGImage object
    :param quality: the IIIF quality parameter
    :return: a jpegtran JPEGImage object
    """
    if quality == Quality.default or quality == Quality.color:
        return image
    if quality == Quality.gray:
        # not sure why the grayscale function isn't exposed on the JPEGImage class but heyho, this
        # does the same thing
        return JPEGImage(blob=Transformation(image.as_blob()).grayscale())
    if quality == Quality.bitonal:
        # convert the image to just black and white using pillow
        return to_jpegtran(to_pillow(image).convert('1'))


def process_format(image: JPEGImage, fmt: Format, output_path: Path):
    """
    Processes the IIIF format parameter by writing the image to the output path in the requested
    format.

    :param image: a jpegtran JPEGImage object
    :param fmt: the format to write the file in
    :param output_path: the path to write the file to
    """
    with output_path.open('wb') as f:
        if fmt == Format.jpg:
            f.write(image.as_blob())
        elif fmt == Format.png:
            to_pillow(image).save(f, format='png')


def process_image_request(source_path: Path, output_path: Path, ops: IIIFOps):
    """
    Process a single image at the source path using the given ops and save it in the output path.

    :param source_path: the source image
    :param output_path: the target location
    :param ops: the IIIF operations to perform
    """
    image = JPEGImage(str(source_path))

    image = process_region(image, ops.region)
    image = process_size(image, ops.size)
    image = process_rotation(image, ops.rotation)
    image = process_quality(image, ops.quality)

    # ensure the full cache path exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # write the processed image to disk
    process_format(image, ops.format, output_path)


@dataclass
class ImageProcessorTask(Fetchable):
    """
    Class used to manage the ImageProcessor tasks.
    """
    profile: AbstractProfile
    info: ImageInfo
    ops: IIIFOps

    @property
    def public_name(self) -> str:
        return self.info.identifier

    @property
    def store_path(self) -> Path:
        return self.profile.cache_path / self.info.name / self.ops.location

    @property
    def size_hint(self) -> Optional[Tuple[int, int]]:
        """
        This is used as a performance enhancement. By providing a hint at the size of the image we
        need to serve up, we can (sometimes!) use a smaller source image thus reducing downloading,
        storage, and processing time.

        :return: the hint size as a tuple or None if we can't hint at anything
        """
        # this is a performance enhancement.
        if self.ops.region.full:
            # the full image region is selected so we can take the hint from the size parameter
            return self.ops.size.w, self.ops.size.h
        else:
            # a region has been specified, we'll have to use the whole thing
            # TODO: can we do better than this?
            return None


class ImageProcessor(FetchCache):
    """
    Class controlling IIIF image processing.
    """

    def __init__(self, root: Path, ttl: float, max_size: float, max_workers: int = os.cpu_count()):
        """
        :param root: the root path to cache the processed images in
        :param ttl: how long processed images should exist on disk after they've been last used
        :param max_size: maximum bytes to store in this cache
        :param max_workers: maximum number of worker processes to use to work on processing images
        """
        super().__init__(root, ttl, max_size)
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers=max_workers)

    async def process(self, profile: AbstractProfile, info: ImageInfo, ops: IIIFOps) -> Path:
        """
        Process an image according to the IIIF ops.

        Technically, the path returned could become invalid immediately as this is not a context
        manager and therefore doesn't guarantee that another coroutine wouldn't delete it, but
        realistically this would only happen if the amount of time it takes you to do whatever you
        need to do with the file exceeds the ttl of the file.

        If you need to ensure the file exists, use the `use` function as a context manager instead.

        :param profile: the profile
        :param info: the image info
        :param ops: IIIF ops to perform
        :return: the path of the processed image
        """
        async with self.use(ImageProcessorTask(profile, info, ops)) as path:
            return path

    async def _fetch(self, task: ImageProcessorTask):
        """
        Perform the actual processing to produce the derived image.

        :param task: the task information
        """
        loop = asyncio.get_event_loop()
        async with task.profile.use_source(task.info, task.size_hint) as source_path:
            await loop.run_in_executor(self.executor, process_image_request, source_path,
                                       task.store_path, task.ops)

    def stop(self):
        """
        Shuts down the processing pool. This will block until they're all done.
        """
        self.executor.shutdown()

    async def get_status(self) -> dict:
        """
        Returns some basic stats info as a dict.

        :return: a dict of stats
        """
        status = await super().get_status()
        status['pool_size'] = self.max_workers
        return status
