from concurrent.futures import Executor

import aiofiles
import asyncio
import imghdr
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from iiif.config import Config
from iiif.exceptions import ImageNotFound, IIIFServerException
from iiif.profiles.base import AbstractProfile, ImageInfo
from iiif.utils import get_size, FetchCache, Fetchable, convert_image


class MissingFile(ImageNotFound):

    def __init__(self, profile: str, name: str, source: Path):
        super().__init__(profile, name,
                         log=f"Couldn't find the image file for {name} on disk at {source}")
        self.source = source


class OnDiskConversionFailure(IIIFServerException):
    def __init__(self, fetchable: 'Fetchable', cause: Exception):
        super().__init__(f'Failed to convert source image',
                         log=f'Failed to convert {fetchable.public_name} due to {cause}')


class OnDiskProfile(AbstractProfile):
    """
    A profile representing source files that are already on disk and don't need to be fetched from
    an external source.
    """

    def __init__(self,
                 name: str,
                 config: Config,
                 pool: Executor,
                 rights: str,
                 cache_for: float = 60,
                 cache_size: int = 1024 * 1024 * 256,
                 convert_quality: int = 85,
                 convert_subsampling: str = '4:2:0',
                 **kwargs
                 ):
        """
        :param name: the name of the profile, should be unique across profiles
        :param config: the config object
        :param pool: the general purpose pool for offloading processing if necessary
        :param rights: the rights definition for all images handled by this profile
        :param cache_for: how long in seconds a client should cache the results from this profile
                          (both info.json and image data)
        :param cache_size: max size in bytes of the source cache on disk
        :param convert_quality: quality to use when converting a source to a jpeg
        :param convert_subsampling: subsampling value to use when converting a source to a jpeg
        :param kwargs: extra kwargs for the AbstractProfile base class __init__
        """
        super().__init__(name, config, pool, rights, cache_for, **kwargs)
        self.store = OnDiskStore(self.cache_path / 'jpeg', pool, cache_for, cache_size,
                                 convert_quality, convert_subsampling)

    async def get_info(self, name: str) -> ImageInfo:
        """
        Given an image name, returns an info object for it. If the image doesn't exist on disk then
        an error is raised.

        :param name: the image name
        :return: an ImageInfo instance
        :raises: MissingFile if the file is missing
        """
        source = self._get_source(name)
        return ImageInfo(self.name, name, *get_size(source))

    @asynccontextmanager
    async def use_source(self, info: ImageInfo, *args, **kwargs) -> Path:
        """
        Given an info object, yields the path to the on disk image source. The target size is
        ignored by this function because we only have the full size originals and nothing else.

        :param info: the image info object
        :return: the path to the source image on disk
        :raises: HTTPException if the file is missing
        """
        source_file_path = self._get_source(info.name)
        source_file_type = imghdr.what(source_file_path)
        if source_file_type == 'jpeg':
            yield source_file_path
        else:
            source = OnDiskSourceFile(info.name, source_file_path)
            async with self.store.use(source) as path:
                yield path

    def _get_source(self, name: str) -> Path:
        """
        Returns the path to the given name in this profile. If the file doesn't exist, raises a
        MissingFile error.

        :param name: the name of the image
        :return: the path to the image
        :raises: MissingFile exception if the file doesn't exist
        """
        source = self.source_path / name
        if not source.exists():
            raise MissingFile(self.name, name, source)
        return source

    async def resolve_filename(self, name: str) -> str:
        """
        Given an image name, resolves the filename..

        :param name: the image name
        :return: the source filename
        :raises: MissingFile exception if the file doesn't exist
        """
        return self._get_source(name).name

    async def resolve_original_size(self, name: str) -> int:
        """
        Given an image, returns the size of the source image.

        :param name: the image name
        :return: the size of the source file in bytes
        :raises: MissingFile exception if the file doesn't exist
        """
        return self._get_source(name).stat().st_size

    async def stream_original(self, name: str, chunk_size: int = 4096):
        """
        Streams the source file for the given image name from disk to the requester. This function
        uses aiofiles to avoid locking up the server.

        :param name: the image name
        :param chunk_size: the size in bytes of each chunk
        :return: yields chunks of bytes
        :raises: MissingFile exception if the file doesn't exist
        """
        source = self._get_source(name)
        async with aiofiles.open(file=str(source), mode='rb') as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk


@dataclass
class OnDiskSourceFile(Fetchable):
    """
    Fetchable subclass representing an image on disk.
    """
    name: str
    original_file: Path

    @property
    def public_name(self) -> str:
        return str(self.name)

    @property
    def store_path(self) -> Path:
        suffixes = self.original_file.suffixes + ['.jpg']
        return self.original_file.with_suffix(''.join(suffixes))


class OnDiskStore(FetchCache):
    def __init__(self, root: Path, pool: Executor, ttl: float, max_size: float,
                 quality: int = 85, subsampling: str = '4:2:0'):
        """
        Note that this init will automatically call self.load() and therefore populate the cache.
        This could take time if the cache is enormous.

        :param root: the root under which all data will be stored
        :param pool: the general purpose pool for offloading processing if necessary
        :param ttl: how long untouched files can stay in the cache before being removed
        :param max_size: the maximum number of bytes that can be stored in the cache
        :param quality: jpeg quality of converted source files
        :param subsampling: jpeg subsampling value of converted source files
        """
        super().__init__(root, ttl, max_size)
        self.pool = pool
        self._convert = partial(convert_image, quality=quality, subsampling=subsampling)

    async def _fetch(self, disk_source: OnDiskSourceFile):
        # convert the image file, saving the data in a temp file but then moving it
        # to the source_path after the conversion is complete
        with tempfile.NamedTemporaryFile(delete=False) as g:
            target_path = Path(g.name)

            convert = partial(self._convert, disk_source.original_file, target_path)
            try:
                await asyncio.get_running_loop().run_in_executor(self.pool, convert)
            except Exception as cause:
                raise OnDiskConversionFailure(disk_source, cause)

            cache_path = self.root / disk_source.store_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(target_path, cache_path)
