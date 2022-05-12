import aiofiles
from contextlib import asynccontextmanager
from pathlib import Path

from iiif.exceptions import ImageNotFound
from iiif.profiles.base import AbstractProfile, ImageInfo
from iiif.utils import get_size


class MissingFile(ImageNotFound):

    def __init__(self, profile: str, name: str, source: Path):
        super().__init__(profile, name, log=f"Couldn't find the image file for {name} on disk at"
                                            f" {source}")
        self.source = source


class OnDiskProfile(AbstractProfile):
    """
    A profile representing source files that are already on disk and don't need to be fetched from
    an external source.
    """

    async def get_info(self, name: str) -> ImageInfo:
        """
        Given an image name, returns an info object for it. If the image doesn't exist on disk then
        an error is raised.

        :param name: the image name
        :return: an ImageInfo instance
        :raises: HTTPException if the file is missing
        """
        source = self._get_source(name)
        if not source.exists():
            raise MissingFile(self.name, name, source)
        else:
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
        source = self._get_source(info.name)
        if not source.exists():
            raise MissingFile(self.name, info.name, source)
        yield source

    def _get_source(self, name: str) -> Path:
        """
        Returns the path to the given name in this profile.

        :param name: the name of the image
        :return: the path to the image
        """
        return self.source_path / name

    async def resolve_filename(self, name: str) -> str:
        """
        Given an image name, resolves the filename..

        :param name: the image name
        :return: the source filename
        """
        return self._get_source(name).name

    async def resolve_original_size(self, name: str) -> int:
        """
        Given an image, returns the size of the source image.

        :param name: the image name
        :return: the size of the source file in bytes
        """
        source = self._get_source(name)
        if not source.exists():
            raise MissingFile(self.name, name, source)
        return source.stat().st_size

    async def stream_original(self, name: str, chunk_size: int = 4096, raise_errors=True):
        """
        Streams the source file for the given image name from disk to the requester. This function
        uses aiofiles to avoid locking up the server.

        :param name: the image name
        :param chunk_size: the size in bytes of each chunk
        :param raise_errors: whether to raise errors if they occur, or just stop (default: True)
        :return: yields chunks of bytes
        """
        source = self._get_source(name)
        if source.exists():
            try:
                async with aiofiles.open(file=str(source), mode='rb') as f:
                    while True:
                        chunk = await f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                if raise_errors:
                    raise e
        else:
            if raise_errors:
                raise MissingFile(self.name, name, source)
