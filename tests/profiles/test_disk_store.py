import os
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from iiif.profiles.disk import OnDiskConversionFailure, OnDiskConvertedFile, OnDiskStore


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

    source = OnDiskConvertedFile(img_name, img_path)
    async with store.use(source) as converted_path:
        assert converted_path.exists()
        with Image.open(converted_path) as image:
            assert image.format.lower() == 'jpeg'


def convert_image(*args, **kwargs):
    # we have to patch with this instead of a Mock because Mocks aren't pickleable
    raise Exception('Oh no')


@patch('iiif.profiles.disk.convert_image', new=convert_image)
async def test_use_convert_error_raises_conversion_error(
    source_root: Path, cache_root: Path, pool: Executor
):
    os.mkdir(source_root)
    store = OnDiskStore(cache_root, pool, 10, 10)

    img_name = 'image.tif'
    img_path = source_root / img_name
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path, format='tiff')

    source = OnDiskConvertedFile(img_name, img_path)
    with pytest.raises(OnDiskConversionFailure) as exc_info:
        async with store.use(source):
            pass
    assert exc_info.value.cause.args[0] == 'Oh no'
