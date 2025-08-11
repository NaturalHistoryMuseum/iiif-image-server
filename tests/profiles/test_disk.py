import math
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import MagicMock

import pytest

from iiif.profiles.base import ImageInfo
from iiif.profiles.disk import MissingFile, OnDiskProfile
from tests.helpers.utils import create_image


@pytest.fixture(scope='function')
def disk_profile(config):
    return OnDiskProfile(
        'test', config, MagicMock(), 'http://creativecommons.org/licenses/by/4.0/'
    )


@pytest.fixture(scope='function')
def disk_profile_with_pool(config):
    return OnDiskProfile(
        'test',
        config,
        ProcessPoolExecutor(max_workers=1),
        'http://creativecommons.org/licenses/by/4.0/',
    )


class TestOnDiskProfile:
    async def test_get_info_no_file(self, disk_profile):
        with pytest.raises(MissingFile) as exc_info:
            await disk_profile.get_info('image')
        assert exc_info.value.status_code == 404

    async def test_get_info_with_file(self, disk_profile, config):
        create_image(config, 100, 100, 'test', 'image')
        info = await disk_profile.get_info('image')
        assert info.size == (100, 100)

    async def test_use_source_no_file(self, config, disk_profile):
        info = ImageInfo('test', 'image', 100, 100)
        with pytest.raises(MissingFile) as exc_info:
            async with disk_profile.use_source(info) as _:
                pass
        assert exc_info.value.status_code == 404

    async def test_use_source_with_file(self, config, disk_profile):
        info = ImageInfo('test', 'image', 100, 100)
        create_image(config, 100, 100, 'test', 'image')
        async with disk_profile.use_source(info) as source_path:
            assert source_path == disk_profile.source_path / 'image'

        info = ImageInfo('test', 'image', 100, 100)
        create_image(config, 100, 100, 'test', 'image')
        # target size doesn't matter for on disk images
        async with disk_profile.use_source(info, (50, 50)) as source_path:
            assert source_path == disk_profile.source_path / 'image'

    @pytest.mark.parametrize(
        'img_format,img_mode', [('tiff', 'RGB'), ('tiff', 'RGBA'), ('png', 'RGB')]
    )
    async def test_use_source_converts_non_jpeg_file(
        self, config, disk_profile_with_pool, img_format, img_mode
    ):
        img_name = f'image_{img_mode}.{img_format}'
        info = ImageInfo('test', img_name, 100, 100)
        create_image(config, 100, 100, 'test', img_name, img_format, img_mode)
        async with disk_profile_with_pool.use_source(info) as converted_path:
            assert converted_path == disk_profile_with_pool.cache_path / 'jpeg' / (
                img_name + '.jpg'
            )

    async def test_resolve_filename_no_file(self, config, disk_profile):
        create_image(config, 100, 100, 'test', 'image')
        filename = await disk_profile.resolve_filename('image')
        assert filename == 'image'

    async def test_stream_original_no_file(self, disk_profile):
        with pytest.raises(MissingFile) as exc_info:
            async for _ in disk_profile.stream_original('image'):
                pass
        assert exc_info.value.status_code == 404

    async def test_stream_original(self, config, disk_profile):
        path = create_image(config, 10000, 10000, 'test', 'image')
        size = path.stat().st_size
        chunk_size = 1024
        expected_count = int(math.ceil(size / chunk_size))

        count = 0
        data = b''
        async for chunk in disk_profile.stream_original('image', chunk_size=chunk_size):
            count += 1
            data += chunk

        assert count == expected_count
        with path.open('rb') as f:
            assert f.read() == data

    async def test_get_original_size(self, config, disk_profile):
        path = create_image(config, 10000, 10000, 'test', 'image')
        size = path.stat().st_size
        profile_size = await disk_profile.resolve_original_size('image')
        assert size == profile_size

    async def test_get_status(self, config, disk_profile_with_pool):
        img_name = 'image.tiff'
        info = ImageInfo('test', img_name, 100, 100)
        create_image(config, 100, 100, 'test', img_name, 'tiff')
        async with disk_profile_with_pool.use_source(info) as converted_path:
            size = converted_path.stat().st_size
        status = await disk_profile_with_pool.get_status()
        assert 'converted_cache' in status
        assert status['converted_cache']['cache_size'] == f'{size} Bytes'
