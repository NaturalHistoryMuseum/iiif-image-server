from copy import deepcopy
from typing import Callable

import os
import yaml
from pathlib import Path


class ProfileRegistry:
    """
    This class is a registry for profiles! Funny that.
    """

    def __init__(self):
        self.profile_types = {}

    def register(self, profile_type: str):
        """
        Registers the decorated class/function with the registry under the given type.

        :param profile_type: the type name
        :return: a decorator function
        """

        def decorator(to_register: Callable):
            self.profile_types[profile_type] = to_register
            return to_register

        return decorator

    def load_profiles(self, config: 'Config', profile_options: dict) -> dict:
        """
        Given the config object, create all the profiles that are defined within it.

        :param config: the config object
        :param profile_options: the profile options dict from the config source
        :return: a dict of profiles keyed by name
        """
        profiles = {}
        # use a deep copy of the profiles config so that we don't mess with the one stored on the
        # config object (we do a pop in the loop!)
        profile_options = deepcopy(profile_options)
        for name, options in profile_options.items():
            # here's that pop, wow!
            profile_creator = self.profile_types[options.pop('type')]
            rights = options.pop('rights')
            profiles[name] = profile_creator(name, config, rights, **options)
        return profiles


registry = ProfileRegistry()


class Config:

    def __init__(self, **options):
        self.raw = options
        # basic network configuration
        self.base_url = options.get('base_url', 'http://10.0.11.20/iiif')

        # paths
        self.source_path = Path(options.get('source_path', '/base/data/iiif/source'))
        self.source_path.mkdir(exist_ok=True)
        self.cache_path = Path(options.get('cache_path', '/base/data/iiif/cache'))
        self.cache_path.mkdir(exist_ok=True)

        # info.json settings
        self.min_sizes_size = options.get('min_sizes_size', 200)
        self.info_cache_size = options.get('info_cache_size', 1000)

        # image processing settings
        self.image_pool_size = options.get('image_pool_size', 2)
        self.image_cache_size_per_process = options.get('image_cache_size_per_process', 5)

        # size definitions for the quick access endpoints
        self.thumbnail_width = options.get('thumbnail_width', 500)
        self.preview_width = options.get('preview_width', 1500)

        # original and batch download options
        self.download_chunk_size = options.get('download_chunk_size', 4096)
        self.download_max_files = options.get('download_max_files', 20)

        # load the profile configurations
        self.profiles = registry.load_profiles(self, options.get('profiles', {}))


def load_config() -> Config:
    """
    Load the configuration and return it. The configuration must be a yaml file and will be loaded
    from either the path specified by the IIIF_CONFIG env var or by looking for the file in the
    folder above the location of this script. The env var takes priority.

    :return: a new Config object
    """
    env_path = os.environ.get('IIIF_CONFIG')
    if env_path is not None:
        config_path = Path(env_path)
    else:
        # no env var, just load the config file from the folder above this script's location
        config_path = Path(__file__).parent.parent / 'config.yml'

    if not config_path.exists():
        raise Exception(f'The config path "{config_path}" does not exist :(')

    with config_path.open('rb') as cf:
        return Config(**yaml.safe_load(cf))
