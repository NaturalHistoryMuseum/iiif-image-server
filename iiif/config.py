#!/usr/bin/env python3
# encoding: utf-8

from copy import deepcopy
from typing import Callable

import os
import yaml
from pathlib import Path


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

        # image processing settings
        self.image_pool_size = options.get('image_pool_size', 2)
        self.image_cache_size_per_process = options.get('image_cache_size_per_process', 5)

        # size definitions for the quick access endpoints
        self.thumbnail_width = options.get('thumbnail_width', 500)
        self.preview_width = options.get('preview_width', 1500)

        # original and batch download options
        self.download_chunk_size = options.get('download_chunk_size', 4096)
        self.download_max_files = options.get('download_max_files', 20)

        self.default_profile_name = options.get('default_profile', None)
        self.profile_options = options.get('profiles', {})

    def has_default_profile(self) -> bool:
        return self.default_profile_name is not None


def load_config() -> Config:
    """
    Load the configuration and return it. The configuration must be a yaml file and will be loaded
    from the path specified by the IIIF_CONFIG env var.

    :return: a new Config object
    """
    env_path = os.environ.get('IIIF_CONFIG')
    if env_path is None:
        raise Exception('The config path was not set using env var IIIF_CONFIG')

    config_path = Path(env_path)
    if not config_path.exists():
        raise Exception(f'The config path "{config_path}" does not exist :(')

    with config_path.open('rb') as cf:
        return Config(**yaml.safe_load(cf))
