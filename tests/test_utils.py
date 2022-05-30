#!/usr/bin/env python3
# encoding: utf-8
import asyncio
import pytest
from PIL import Image
from dataclasses import dataclass
from jpegtran import JPEGImage
from pathlib import Path
from unittest.mock import patch, MagicMock
from wand.exceptions import MissingDelegateError

from iiif.utils import convert_image, generate_sizes, get_size, get_mimetype, parse_identifier, \
    to_pillow, to_jpegtran, Locker, FetchCache, Fetchable, generate_tiles
from tests.utils import create_image, create_file


class TestConvertImage:

    def test_jpeg(self, tmp_path):
        image_path = tmp_path / 'image'
        img = Image.new('RGB', (400, 400), color='red')
        img.save(image_path, format='jpeg')

        target = tmp_path / 'converted'
        convert_image(image_path, target)

        target_fallback = tmp_path / 'converted_fallback'
        wand_mock = MagicMock(side_effect=MissingDelegateError())
        with patch('iiif.utils.WandImage', wand_mock):
            convert_image(image_path, target_fallback)

        assert target.exists()
        assert target_fallback.exists()
        with Image.open(target) as converted_image:
            assert converted_image.format.lower() == 'jpeg'
            with Image.open(target) as converted_fallback_image:
                assert converted_fallback_image.format.lower() == 'jpeg'
                assert converted_fallback_image == converted_image

    def test_jpeg_with_exif_orientation(self, tmp_path):
        image_path = tmp_path / 'image'
        img = Image.new('RGB', (700, 400), color='red')
        exif = img.getexif()
        exif[0x0112] = 6
        img.info['exif'] = exif.tobytes()
        img.save(image_path, format='jpeg')

        target = tmp_path / 'converted'
        convert_image(image_path, target)

        target_fallback = tmp_path / 'converted_fallback'
        wand_mock = MagicMock(side_effect=MissingDelegateError())
        with patch('iiif.utils.WandImage', wand_mock):
            convert_image(image_path, target_fallback)

        assert target.exists()
        assert target_fallback.exists()
        with Image.open(target) as converted_image:
            assert converted_image.format.lower() == 'jpeg'
            assert 0x0112 not in converted_image.getexif()
            with Image.open(target) as converted_fallback_image:
                assert converted_fallback_image.format.lower() == 'jpeg'
                assert 0x0112 not in converted_fallback_image.getexif()
                assert converted_fallback_image == converted_image

    def test_tiff(self, tmp_path):
        image_path = tmp_path / 'image'
        img = Image.new('RGB', (400, 400), color='red')
        img.save(image_path, format='tiff')

        target = tmp_path / 'converted'
        convert_image(image_path, target)

        target_fallback = tmp_path / 'converted_fallback'
        wand_mock = MagicMock(side_effect=MissingDelegateError())
        with patch('iiif.utils.WandImage', wand_mock):
            convert_image(image_path, target_fallback)

        assert target.exists()
        assert target_fallback.exists()
        with Image.open(target) as converted_image:
            assert converted_image.format.lower() == 'jpeg'
            with Image.open(target) as converted_fallback_image:
                assert converted_fallback_image.format.lower() == 'jpeg'
                assert converted_fallback_image == converted_image

    def test_jpeg_options(self, tmp_path):
        image_path = tmp_path / 'image'
        img = Image.new('RGB', (400, 400), color='red')
        img.save(image_path, format='jpeg')

        target = tmp_path / 'converted'
        convert_image(image_path, target, quality=40, subsampling='4:2:2')

        target_fallback = tmp_path / 'converted_fallback'
        wand_mock = MagicMock(side_effect=MissingDelegateError())
        with patch('iiif.utils.WandImage', wand_mock):
            convert_image(image_path, target_fallback, quality=40, subsampling='4:2:2')

        assert target.exists()
        assert target_fallback.exists()
        with Image.open(target) as converted_image:
            assert converted_image.format.lower() == 'jpeg'
            with Image.open(target) as converted_fallback_image:
                assert converted_fallback_image.format.lower() == 'jpeg'
                assert converted_fallback_image == converted_image


def test_generate_sizes():
    sizes = generate_sizes(1000, 1001, 200)

    # test that the call result is cached
    assert generate_sizes(1000, 1001, 200) is sizes

    assert len(sizes) == 3
    assert sizes[0] == {'width': 1000, 'height': 1001}
    assert sizes[1] == {'width': 500, 'height': 500}
    assert sizes[2] == {'width': 250, 'height': 250}


def test_generate_tiles_exact_match():
    tiles = generate_tiles(256, 4096, 8192)

    # test that the call result is cached
    assert generate_tiles(256, 4096, 8192) is tiles

    assert tiles['width'] == 256
    assert tiles['scaleFactors'] == [1, 2, 4, 8, 16, 32]


def test_generate_tiles_not_exact():
    tiles = generate_tiles(256, 8191, 8193)

    # test that the call result is cached
    assert generate_tiles(256, 8191, 8193) is tiles

    assert tiles['width'] == 256
    assert tiles['scaleFactors'] == [1, 2, 4, 8, 16, 32, 64]


def test_get_size(config):
    image_path = create_image(config, 289, 4390)
    assert get_size(image_path) == (289, 4390)


mss_base_url_scenarios = [
    ('0', '0/000'),
    ('1', '0/001'),
    ('14', '0/014'),
    ('305', '0/305'),
    ('9217', '9/217'),
    ('2389749823', '2389749/823'),
]


class TestGetMimetype:

    def test_normal(self):
        assert get_mimetype('something.jpg') == 'image/jpeg'

    def test_default(self):
        assert get_mimetype('unknown') == 'application/octet-stream'


def test_parse_identifier():
    assert parse_identifier('beans:goats') == ('beans', 'goats')
    assert parse_identifier('goats') == (None, 'goats')


def test_to_pillow_and_to_jpegtran():
    pillow_image = Image.new('RGB', (100, 400), color='red')
    jpegtran_image = to_jpegtran(pillow_image)
    assert isinstance(jpegtran_image, JPEGImage)
    pillow_image_again = to_pillow(jpegtran_image)
    assert isinstance(pillow_image_again, Image.Image)


@dataclass
class FetchableForTesting(Fetchable):

    def __init__(self, size: int):
        self.size = size
        self.name = f'{self.size}bytes.bin'

    @property
    def public_name(self) -> str:
        return self.name

    @property
    def store_path(self) -> Path:
        return Path('test', self.name)


class FetchCacheForTesting(FetchCache):

    def __init__(self, root: Path, ttl: float = 1, max_size: float = 20):
        super().__init__(Path(root), ttl, max_size)

    async def _fetch(self, fetchable: FetchableForTesting):
        path = self.root / fetchable.store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        create_file(path, fetchable.size)


@pytest.fixture
def cache(tmpdir: Path) -> FetchCacheForTesting:
    return FetchCacheForTesting(tmpdir)


class TestFetchCache:

    async def test_simple_usage(self, cache: FetchCacheForTesting):
        fetchable = FetchableForTesting(4)
        async with cache.use(fetchable) as path:
            assert cache.total_size == 4
            assert cache.errors == 0
            assert cache.requests == 1
            assert cache.completed == 0
            assert path.exists()
            assert fetchable.store_path in cache
            await asyncio.sleep(cache.ttl + 0.5)
            assert path.exists()
            assert fetchable.store_path in cache
        assert cache.completed == 1
        await asyncio.sleep(cache.ttl + 0.5)
        assert not path.exists()
        assert fetchable.store_path not in cache
        assert not path.parent.exists()


class TestLocker:

    async def test_is_locked(self):
        locker = Locker()
        async with locker.acquire('test'):
            assert locker.is_locked('test')

    async def test_acquire_no_timeout(self):
        locker = Locker()

        async def first():
            async with locker.acquire('test'):
                await asyncio.sleep(3)

        async def second():
            try:
                async with locker.acquire('test', timeout=0):
                    assert True
            except asyncio.TimeoutError:
                assert False

        task = asyncio.ensure_future(first())
        await asyncio.sleep(1)
        await second()
        await task
        assert not locker.is_locked('test')

    async def test_acquire_with_shorter_timeout(self):
        locker = Locker()

        async def first():
            async with locker.acquire('test'):
                await asyncio.sleep(5)

        async def second():
            try:
                async with locker.acquire('test', timeout=1):
                    # should never get here because we wait for the lock and then get cancelled
                    # before it's available
                    assert False
            except asyncio.TimeoutError:
                assert locker.is_locked('test')

        task = asyncio.ensure_future(first())
        await asyncio.sleep(1)
        await second()
        await task
        assert not locker.is_locked('test')

    async def test_acquire_with_longer_timeout(self):
        locker = Locker()

        async def first():
            async with locker.acquire('test'):
                await asyncio.sleep(3)

        async def second():
            try:
                async with locker.acquire('test', timeout=5):
                    assert True
            except asyncio.TimeoutError:
                assert False

        task = asyncio.ensure_future(first())
        await asyncio.sleep(1)
        await second()
        await task
        assert not locker.is_locked('test')
