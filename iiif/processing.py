#!/usr/bin/env python3
# encoding: utf-8
import asyncio
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from iiif.ops import IIIFOps
from iiif.profiles.base import AbstractProfile, ImageInfo
from iiif.utils import Fetchable, FetchCache


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
        This is used as a performance enhancement. By providing a hint at the size of
        the image we need to serve up, we can (sometimes!) use a smaller source image
        thus reducing downloading, storage, and processing time.

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

    def __init__(self, root: Path, pool: Executor, ttl: float, max_size: float):
        """
        :param root: the root path to cache the processed images in
        :param pool: the general purpose pool for offloading processing if necessary
        :param ttl: how long processed images should exist on disk after they've been last used
        :param max_size: maximum bytes to store in this cache
        """
        super().__init__(root, ttl, max_size)
        self.pool = pool

    async def process(
        self, profile: AbstractProfile, info: ImageInfo, ops: IIIFOps
    ) -> Path:
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
            await loop.run_in_executor(
                self.pool, task.ops.process, source_path, task.store_path
            )
