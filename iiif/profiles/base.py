import abc
from concurrent.futures import Executor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterable, Optional, Tuple

from iiif.config import Config


class ImageInfo:
    """
    Base info class which holds the basic details of an image.
    """

    def __init__(self, profile_name: str, name: str, width: int, height: int):
        """
        :param profile_name: the name of the profile this image belongs to
        :param name: the name of image (this should be unique within the profile's jurisdiction)
        :param width: the width of the image
        :param height: the height of the image
        """
        self.profile_name = profile_name
        self.name = name
        self.width = width
        self.height = height
        self.identifier = f'{self.profile_name}:{self.name}'

    @property
    def size(self) -> Tuple[int, int]:
        """
        Returns width and height as a 2-tuple.

        :return: a 2-tuple containing the width and height
        """
        return self.width, self.height

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ImageInfo):
            return self.identifier == other.identifier
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.identifier)


class AbstractProfile(abc.ABC):
    """
    Abstract base class for all profiles.

    Each subclass defines a profile which can be used to retrieve the source image data
    required to complete a IIIF request (either info.json or actual data). Once a
    profile has been used to retrieve a source image and make it real on disk as a jpeg
    file, it can be processed in a common way (see the processing module).
    """

    def __init__(
        self,
        name: str,
        config: Config,
        pool: Executor,
        rights: str,
        cache_for: float = 60,
        **kwargs,
    ):
        """
        :param name: the name of the profile, should be unique across profiles
        :param config: the config object
        :param pool: the general purpose pool for offloading processing if necessary
        :param rights: the rights definition for all images handled by this profile
        :param cache_for: how long in seconds a client should cache the results from this profile
                          (both info.json and image data)
        """
        self.name = name
        self.config = config
        self.pool = pool
        # this is where all of our source images will be stored
        self.source_path = config.source_path / name
        # this is where all of our processed images will be stored
        self.cache_path = config.cache_path / name
        self.rights = rights
        self.source_path.mkdir(exist_ok=True)
        self.cache_path.mkdir(exist_ok=True)
        self.cache_for = cache_for

    @abc.abstractmethod
    async def get_info(self, name: str) -> ImageInfo:
        """
        Returns an instance of ImageInfo or one of it's subclasses for the given name.
        Note that this function does not generate the info.json, see generate_info_json
        below for that!

        :param name: the name of the image
        :return: an info object
        """
        ...

    @abc.abstractmethod
    @asynccontextmanager
    async def use_source(
        self, info: ImageInfo, size: Optional[Tuple[int, int]] = None
    ) -> Path:
        """
        Ensures the source file required for the image is available, using the optional
        size hint to choose the best file if multiple sizes are available, and then
        yields the path to it. This function should ensure that while the context is up
        the source is available at the path.

        :param info: the ImageInfo
        :param size: an 2-tuple of ints to hint at the target size required for
            subsequent ops
        :return: the Path to the source file
        """
        ...

    @abc.abstractmethod
    async def resolve_filename(self, name: str) -> Optional[str]:
        """
        Given the name of an image produced by this profile, returns the filename that
        should be used for it. This is used when original images are downloaded to give
        them the right name.

        :param name: the image name
        :return: the filename
        """
        ...

    @abc.abstractmethod
    async def resolve_original_size(self, name: str) -> int:
        """
        Given the name of an image managed by this profile, returns the size of the
        original source image.

        :param name: the name of the image
        :return: the size of the original image in bytes
        """
        ...

    @abc.abstractmethod
    async def stream_original(
        self, name: str, chunk_size: int = 4096
    ) -> AsyncIterable[bytes]:
        """
        Streams the original file associated with the given name, if valid. The chunk
        size can be configured to define how much data is yielded each time this
        function is scheduled to run and is defaulted to a relatively low 4096 to ensure
        the server doesn't lock up serving large originals to users.

        :param name: the name of the image
        :param chunk_size: the number of bytes to yield at a time
        :return: an asynchronous generator of bytes
        """
        ...

    async def close(self):
        """
        Close down the profile ensuring any resources are released.

        This will be called before server shutdown.
        """
        pass

    async def get_status(self) -> dict:
        """
        Returns some stats about the profile.

        :return: a dict of stats
        """
        status = {
            'name': self.name,
        }
        return status
