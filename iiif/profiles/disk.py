from pathlib import Path
from typing import Tuple, Optional

import aiofiles

from iiif.profiles.base import AbstractProfile, ImageInfo
from iiif.utils import get_size


class OnDiskProfile(AbstractProfile):
    """
    A profile representing source files that are already on disk and don't need to be fetched from
    an external source.
    """

    async def get_info(self, name: str) -> Optional[ImageInfo]:
        """
        Given an image name, returns an info object for it. If the image doesn't exist on disk then
        None is returned.

        :param name: the image name
        :return: None if the image doesn't exist on disk, or an ImageInfo instance
        """
        source = self._get_source(name)
        if not source.exists():
            return None
        else:
            return ImageInfo(self.name, name, *get_size(source))

    async def fetch_source(self, info: ImageInfo,
                           target_size: Optional[Tuple[int, int]] = None) -> Optional[Path]:
        """
        Given an info object returns the path to the on disk image source. The target size is
        ignored by this function because we only have the full size original and nothing else.

        :param info: the image info object
        :param target_size: a target size - this is ignored by this function
        :return: the path to the source image on disk
        """
        source_path = self._get_source(info.name)
        return source_path if source_path.exists() else None

    def _get_source(self, name: str) -> Path:
        """
        Returns the path to the given name in this profile.

        :param name: the name of the image
        :return: the path to the image
        """
        return self.source_path / name

    async def resolve_filename(self, name: str) -> Optional[str]:
        """
        Given an image name, resolves the filename. Given that the name == the filename, this just
        checks that the name source exists on disk and returns the name if it does.

        :param name: the image name
        :return: the source filename
        """
        source_path = self._get_source(name)
        return source_path.name if source_path.exists() else None

    async def stream_original(self, name: str, chunk_size: int = 4096, raise_errors=True):
        """
        Streams the source file for the given image name from disk to the requester. This function
        uses aiofiles to avoid locking up the server.

        :param name: the image name
        :param chunk_size: the size in bytes of each chunk
        :param raise_errors: whether to raise errors if they occur, or just stop (default: True)
        :return: yields chunks of bytes
        """
        source_path = self._get_source(name)
        if source_path.exists():
            try:
                async with aiofiles.open(file=str(source_path), mode='rb') as f:
                    while True:
                        chunk = await f.read(chunk_size)
                        if chunk:
                            yield chunk
                        else:
                            break
            except Exception as e:
                if raise_errors:
                    raise e
