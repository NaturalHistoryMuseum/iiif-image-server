#!/usr/bin/env python3
# encoding: utf-8

from PIL import Image
from pathlib import Path

from iiif.config import Config


def create_image(config: Config, width: int, height: int, profile: str = 'test',
                 name: str = 'image') -> Path:
    """
    Create a real image file for testing and returns the path to it.

    :param config: the config object
    :param width: the width of the image to create
    :param height: the height of the image to create
    :param profile: the profile name
    :param name: the image name
    :return: the path to the image
    """
    path = config.source_path / profile / name
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (width, height), color='red')
    img.save(path, format='jpeg')
    return path


def create_file(path: Path, size: int):
    with path.open('wb') as f:
        f.write(bytes(size))
    return path
