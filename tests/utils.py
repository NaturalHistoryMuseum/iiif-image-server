#!/usr/bin/env python3
# encoding: utf-8

from PIL import Image
from pathlib import Path

from iiif.config import Config


def create_image(config: Config, width: int, height: int, profile: str = 'test',
                 name: str = 'image', img_format='jpeg', mode='RGB') -> Path:
    """
    Create a real image file for testing and returns the path to it.

    :param config: the config object
    :param width: the width of the image to create
    :param height: the height of the image to create
    :param profile: the profile name
    :param name: the image name
    :param img_format: the image format
    :param mode: the image mode (e.g. RGB or RGBA); will be set to RGB if format is jpeg
    :return: the path to the image
    """
    path = config.source_path / profile / name
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'RGB' if img_format == 'jpeg' else mode
    img = Image.new(mode, (width, height), color='red')
    img.save(path, format=img_format)
    return path


def create_file(path: Path, size: int):
    with path.open('wb') as f:
        f.write(bytes(size))
    return path
