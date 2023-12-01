from concurrent.futures import Executor, ProcessPoolExecutor

import pytest
from PIL import Image
from pathlib import Path
import os
from unittest.mock import patch, Mock

from iiif.profiles.disk import OnDiskStore, OnDiskSourceFile, OnDiskConversionFailure


@pytest.fixture
def source_root(tmpdir) -> Path:
    return Path(tmpdir, 'test_src')


@pytest.fixture
def cache_root(tmpdir) -> Path:
    return Path(tmpdir, 'test_cache')


@pytest.fixture
def pool() -> Executor:
    return ProcessPoolExecutor(max_workers=1)


async def test_use_converts_image(source_root: Path, cache_root: Path, pool: Executor):
    os.mkdir(source_root)
    store = OnDiskStore(cache_root, pool, 10, 10)

    img_name = 'image.tif'
    img_path = source_root / img_name
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path, format='tiff')

    source = OnDiskSourceFile(img_name, img_path)
    async with store.use(source) as converted_path:
        assert converted_path.exists()
        with Image.open(converted_path) as image:
            assert image.format.lower() == 'jpeg'


@patch('iiif.utils.convert_image', return_value=Mock(side_effect=Exception('Oh no')))
async def test_use_convert_error_raises_conversion_error(mock_convert, source_root: Path, cache_root: Path, pool: Executor):
    os.mkdir(source_root)
    store = OnDiskStore(cache_root, pool, 10, 10)

    img_name = 'image.tif'
    img_path = source_root / img_name
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path, format='tiff')

    source = OnDiskSourceFile(img_name, img_path)
    with pytest.raises(OnDiskConversionFailure):
        async with store.use(source):
            pass