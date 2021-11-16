from unittest.mock import patch, MagicMock

import math
import pytest

from iiif.profiles.base import ImageInfo
from iiif.profiles.disk import OnDiskProfile
from tests.utils import create_image


@pytest.fixture(scope='function')
def disk_profile(config):
    return OnDiskProfile('test', config, 'http://creativecommons.org/licenses/by/4.0/')


class TestOnDiskProfile:

    @pytest.mark.asyncio
    async def test_get_info_no_file(self, disk_profile):
        info = await disk_profile.get_info('image')
        assert info is None

    @pytest.mark.asyncio
    async def test_get_info_with_file(self, disk_profile, config):
        create_image(config, 100, 100, 'test', 'image')
        info = await disk_profile.get_info('image')
        assert info.size == (100, 100)

    @pytest.mark.asyncio
    async def test_fetch_source_no_file(self, config, disk_profile):
        info = ImageInfo('test', 'image', 100, 100)
        source_path = await disk_profile.fetch_source(info)
        assert source_path is None

    @pytest.mark.asyncio
    async def test_fetch_source_with_file(self, config, disk_profile):
        info = ImageInfo('test', 'image', 100, 100)
        create_image(config, 100, 100, 'test', 'image')
        source_path = await disk_profile.fetch_source(info)
        assert source_path == disk_profile.source_path / 'image'

        info = ImageInfo('test', 'image', 100, 100)
        create_image(config, 100, 100, 'test', 'image')
        # target size doesn't matter for on disk images
        source_path = await disk_profile.fetch_source(info, (50, 50))
        assert source_path == disk_profile.source_path / 'image'

    @pytest.mark.asyncio
    async def test_resolve_filename_no_file(self, disk_profile):
        filename = await disk_profile.resolve_filename('image')
        assert filename is None

    @pytest.mark.asyncio
    async def test_resolve_filename_with_file(self, config, disk_profile):
        create_image(config, 100, 100, 'test', 'image')
        filename = await disk_profile.resolve_filename('image')
        assert filename == 'image'

    @pytest.mark.asyncio
    async def test_stream_original_no_file(self, disk_profile):
        count = 0
        async for _ in disk_profile.stream_original('image'):
            count += 1
        assert count == 0

    @pytest.mark.asyncio
    async def test_stream_original_with_file(self, config, disk_profile):
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

    @pytest.mark.asyncio
    async def test_stream_original_with_file_errors_on(self, config, disk_profile):
        disk_profile._get_source = MagicMock(return_value=MagicMock(
            exists=MagicMock(return_value=True), __str__=MagicMock(return_value='/dev/null/nope')))
        with pytest.raises(Exception):
            async for _ in disk_profile.stream_original('image', raise_errors=True):
                pass

    @pytest.mark.asyncio
    async def test_stream_original_with_file_errors_off(self, config, disk_profile):
        disk_profile._get_source = MagicMock(return_value=MagicMock(
            exists=MagicMock(return_value=True), __str__=MagicMock(return_value='/dev/null/nope')))
        data = b''
        async for chunk in disk_profile.stream_original('image', raise_errors=False):
            data += chunk
        assert not data
