from unittest.mock import MagicMock

import math
import pytest

from iiif.profiles.base import ImageInfo
from iiif.profiles.disk import OnDiskProfile, MissingFile
from tests.utils import create_image


@pytest.fixture(scope='function')
def disk_profile(config):
    return OnDiskProfile('test', config, MagicMock(), 'http://creativecommons.org/licenses/by/4.0/')


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
