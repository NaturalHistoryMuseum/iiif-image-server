import abc
from cachetools import TTLCache
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Tuple, Optional, Any, AsyncIterable

from iiif.config import Config
from iiif.utils import generate_sizes


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
    Abstract base class for all profiles. Each subclass defines a profile which can be used to
    retrieve the source image data required to complete a IIIF request (either info.json or actual
    data). Once a profile has been used to retrieve a source image and make it real on disk as a
    jpeg file, it can be processed in a common way (see the processing module).
    """

    def __init__(self, name: str, config: Config, rights: str, info_json_cache_size: int = 1000,
                 cache_for: float = 60):
        """
        :param name: the name of the profile, should be unique across profiles
        :param config: the config object
        :param rights: the rights definition for all images handled by this profile
        :param info_json_cache_size: the size of the info.json cache
        :param cache_for: how long in seconds a client should cache the results from this profile
                          (both info.json and image data)
        """
        self.name = name
        self.config = config
        # this is where all of our source images will be stored
        self.source_path = config.source_path / name
        # this is where all of our processed images will be stored
        self.cache_path = config.cache_path / name
        self.rights = rights
        self.source_path.mkdir(exist_ok=True)
        self.cache_path.mkdir(exist_ok=True)
        self.info_json_cache = TTLCache(maxsize=info_json_cache_size, ttl=cache_for)
        self.cache_for = cache_for

    @abc.abstractmethod
    async def get_info(self, name: str) -> ImageInfo:
        """
        Returns an instance of ImageInfo or one of it's subclasses for the given name. Note that
        this function does not generate the info.json, see generate_info_json below for that!

        :param name: the name of the image
        :return: an info object or None if the image wasn't available (for any reason)
        """
        pass

    @abc.abstractmethod
    @asynccontextmanager
    async def use_source(self, info: ImageInfo, size: Optional[Tuple[int, int]] = None) -> Path:
        """
        Ensures the source file required for the image is available, using the optional size hint to
        choose the best file if multiple sizes are available.

        :param info: the ImageInfo
        :param size: an 2-tuple of ints to hint at the target size required for subsequent ops
        :return: the Path to the source file or None if one could not be found
        """
        pass

    @abc.abstractmethod
    async def resolve_filename(self, name: str) -> Optional[str]:
        """
        Given the name of an image produced by this profile, returns the filename that should be
        used for it. This is used when original images are downloaded to give them the right name.

        :param name: the image name
        :return: the filename or None if the name is invalid
        """
        pass

    @abc.abstractmethod
    async def stream_original(self, name: str, chunk_size: int = 4096,
                              raise_errors=True) -> AsyncIterable[bytes]:
        """
        Streams the original file associated with the given name, if valid. The chunk size can be
        configured to define how much data is yielded each time this function is scheduled to run
        and is defaulted to a relatively low 4096 to ensure the server doesn't lock up serving large
        originals to users.

        :param name: the name of the image
        :param chunk_size: the number of bytes to yield at a time
        :param raise_errors: whether to raise errors as they occur or just stop (default: True)
        :return: an asynchronous generator of bytes
        """
        pass

    async def generate_info_json(self, info: ImageInfo, iiif_level: int) -> dict:
        """
        Generates an info.json dict for the given image. The info.json is cached locally in this
        profile's attributes.

        :param info: the ImageInfo object to create the info.json dict for
        :param iiif_level: the IIIF image server compliance level to include in the info.json
        :return: the generated or cached info.json dict for the image
        """
        # if the image's info.json isn't cached, create and add the complete info.json to the cache
        if info not in self.info_json_cache:
            id_url = f'{self.config.base_url}/{info.identifier}'
            self.info_json_cache[info] = {
                '@context': 'http://iiif.io/api/image/3/context.json',
                'id': id_url,
                # mirador/openseadragon seems to need this to work even though I don't think it's
                # correct under the IIIF image API v3
                '@id': id_url,
                'type': 'ImageService3',
                'protocol': 'http://iiif.io/api/image',
                'width': info.width,
                'height': info.height,
                'rights': self.rights,
                'profile': f'level{iiif_level}',
                'tiles': [
                    {'width': 512, 'scaleFactors': [1, 2, 4, 8, 16]},
                    {'width': 256, 'scaleFactors': [1, 2, 4, 8, 16]},
                    {'width': 1024, 'scaleFactors': [1, 2, 4, 8, 16]},
                ],
                'sizes': generate_sizes(info.width, info.height, self.config.min_sizes_size),
                # suggest to clients that upscaling isn't supported
                'maxWidth': info.width,
                'maxHeight': info.height,
            }

        return self.info_json_cache[info]

    async def close(self):
        """
        Close down the profile ensuring any resources are released. This will be called before
        server shutdown.
        """
        pass

    async def get_status(self) -> dict:
        """
        Returns some stats about the profile.

        :return: a dict of stats
        """
        status = {
            'name': self.name,
            'info_json_cache_size': len(self.info_json_cache),
        }
        return status
