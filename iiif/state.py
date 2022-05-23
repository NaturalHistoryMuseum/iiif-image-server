from typing import Tuple, Optional

from iiif.config import load_config, Config
from iiif.exceptions import ProfileNotFound, ImageNotFound
from iiif.processing import ImageProcessor
from iiif.profiles import load_profiles
from iiif.profiles.base import AbstractProfile, ImageInfo
from iiif.utils import parse_identifier


class State:
    """
    Class that will be created as a singleton object to hold the state of the current app.
    """

    def __init__(self):
        self.config: Config = load_config()
        self.profiles = load_profiles(self.config)
        # create the processor which actually does the IIIF image processing
        self.processor = ImageProcessor(self.config.cache_path, self.config.processed_cache_ttl,
                                        self.config.processed_cache_size,
                                        self.config.processing_pool_size)

    def get_profile(self, profile_name: Optional[str] = None) -> AbstractProfile:
        """
        Helper function that gets the AbstractProfile object associated with the given name and
        returns it. If one cannot be found then an error is raised.

        :param profile_name: the profile name
        :return: the profile object (this will be a subclass of the AbstractProfile abstract class)
        """
        if profile_name is None:
            profile_name = self.config.default_profile_name

        profile = self.profiles.get(profile_name, None)
        if profile is None:
            raise ProfileNotFound(profile_name)
        return profile

    @staticmethod
    async def get_info(profile: AbstractProfile, name: str) -> ImageInfo:
        """
        Helper function that gets the ImageInfo from the profile and returns it. If the info isn't
        available then an error is raised.

        :param profile: the profile object (this must be a subclass of the AbstractProfile abstract
                        class)
        :param name: the image name
        :return: the info object (this will be a subclass of the ImageInfo abstract class)
        """
        info = await profile.get_info(name)
        if info is None:
            raise ImageNotFound(profile.name, name)
        return info

    async def get_profile_and_info(self, identifier: str) -> Tuple[AbstractProfile, ImageInfo]:
        """
        Helper function that gets the profile and the info at the same time.

        :param identifier: the image identifier
        :return: a 2-tuple containing the profile object and the info object
        """
        profile_name, name = parse_identifier(identifier)
        profile = self.get_profile(profile_name)
        info = await self.get_info(profile, name)
        return profile, info


state = State()
