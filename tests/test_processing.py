#!/usr/bin/env python3
# encoding: utf-8

import hashlib
from pathlib import Path
from queue import Queue
from typing import Union

import pytest
from jpegtran import JPEGImage
from PIL import Image, ImageOps

from iiif.ops import Format, Quality, Region, Rotation, Size
from iiif.profiles.base import ImageInfo
from iiif.utils import to_jpegtran, to_pillow
from tests.helpers.utils import create_image

DEFAULT_IMAGE_WIDTH = 4000
DEFAULT_IMAGE_HEIGHT = 5000


@pytest.fixture
def source_path(config):
    return create_image(config, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)


@pytest.fixture
def cache_path(config):
    return config.cache_path / 'test' / 'image'


@pytest.fixture
def info():
    return ImageInfo(
        'test_profile', 'test_image', DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT
    )


@pytest.fixture
def task_queue():
    # a real queue but not a multiprocessing one
    return Queue()


@pytest.fixture
def result_queue():
    # a real queue but not a multiprocessing one
    return Queue()


def assert_same(
    image1: Union[JPEGImage, Image.Image], image2: Union[JPEGImage, Image.Image]
):
    if not isinstance(image1, JPEGImage):
        image1 = to_jpegtran(image1)
    image1_hash = hashlib.sha256(image1.as_blob()).hexdigest()
    if not isinstance(image2, JPEGImage):
        image2 = to_jpegtran(image2)
    image2_hash = hashlib.sha256(image2.as_blob()).hexdigest()
    assert image1_hash == image2_hash


@pytest.fixture
def image() -> JPEGImage:
    return to_jpegtran(
        Image.new('RGB', (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT), color='red')
    )


class TestProcessRegion:
    def test_full(self, image: JPEGImage):
        region = Region(0, 0, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, full=True)
        assert region.process(image) is image

    def test_fast(self, image: JPEGImage):
        region = Region(16, 32, 400, 300)
        result = region.process(image)
        assert result.width == 400
        assert result.height == 300
        assert_same(result, image.crop(16, 32, 400, 300))

    def test_slow(self, image: JPEGImage):
        region = Region(14, 37, 400, 300)
        result = region.process(image)
        assert result.width == 400
        assert result.height == 300
        assert_same(result, to_pillow(image).crop((14, 37, 414, 337)))


class TestProcessSize:
    def test_max(self, image: JPEGImage):
        size = Size(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, max=True)
        assert size.process(image) is image

    def test_down(self, image: JPEGImage):
        size = Size(500, 600)
        result = size.process(image)
        assert_same(result, image.downscale(500, 600))


class TestProcessRotation:
    def test_rotate(self, image: JPEGImage):
        rotation = Rotation(90)
        result = rotation.process(image)
        assert_same(result, to_pillow(image).rotate(-90, expand=True))

    def test_mirror(self, image: JPEGImage):
        rotation = Rotation(0, mirror=True)
        result = rotation.process(image)
        assert_same(result, image.flip('horizontal'))

    def test_rotate_and_mirror(self, image: JPEGImage):
        rotation = Rotation(90, mirror=True)
        result = rotation.process(image)
        assert_same(result, ImageOps.mirror(to_pillow(image)).rotate(-90, expand=True))


class TestQuality:
    def test_default(self, image: JPEGImage):
        assert Quality.default.process(image) is image

    def test_color(self, image: JPEGImage):
        assert Quality.color.process(image) is image

    def test_gray(self, image: JPEGImage):
        result = Quality.gray.process(image)
        assert_same(result, to_pillow(image).convert('L'))

    def test_bitonal(self, image: JPEGImage):
        result = Quality.bitonal.process(image)
        assert_same(result, to_pillow(image).convert('1'))


@pytest.mark.parametrize('fmt,expected_format', zip(Format, ['JPEG', 'PNG']))
def test_format(fmt: Format, expected_format: str, tmp_path: Path, image: JPEGImage):
    output_path = tmp_path / 'image'
    fmt.process(image, output_path)
    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.format == expected_format


# TODO: write processor tests
