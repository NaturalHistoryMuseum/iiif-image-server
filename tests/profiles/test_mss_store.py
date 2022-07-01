from concurrent.futures import Executor, ProcessPoolExecutor

import pytest
from PIL import Image
from aiohttp import ClientResponseError
from aioresponses import aioresponses
from io import BytesIO
from pathlib import Path

from iiif.profiles.mss import MSSSourceStore, MSSSourceFile, StoreStreamError, StoreStreamNoLength

MOCK_HOST = 'http://not.the.real.mss.host'


@pytest.fixture
def source_root(tmpdir) -> Path:
    return Path(tmpdir, 'test')


@pytest.fixture
def pool() -> Executor:
    return ProcessPoolExecutor(max_workers=1)


async def test_check_access_ok(source_root: Path, pool: Executor):
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    try:
        with aioresponses() as m:
            m.get(MSSSourceFile.check_url(emu_irn), status=204)
            assert await store.check_access(emu_irn)
    finally:
        await store.close()


async def test_check_access_failed(source_root: Path, pool: Executor):
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    try:
        with aioresponses() as m:
            m.get(MSSSourceFile.check_url(emu_irn), status=401)
            assert not await store.check_access(emu_irn)
    finally:
        await store.close()


async def test_stream(source_root: Path, pool: Executor):
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'
    content = b'1234 look at me i am a jpg image'
    chunk_size = 2
    try:
        with aioresponses() as m:
            source = MSSSourceFile(emu_irn, file, False, chunk_size=chunk_size)
            m.get(source.url, body=content)
            buffer = BytesIO()
            async for chunk in store.stream(source):
                buffer.write(chunk)
                assert len(chunk) <= chunk_size
            assert buffer.getvalue() == content
    finally:
        await store.close()


async def test_stream_access_denied(source_root, pool: Executor):
    source_root: Path
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'
    try:
        with aioresponses() as m:
            source = MSSSourceFile(emu_irn, file, False)
            m.get(source.url, status=401)
            with pytest.raises(StoreStreamError) as exc_info:
                async for chunk in store.stream(source):
                    pass
            assert exc_info.value.url.endswith(source.url)
            assert exc_info.value.source is source
            assert isinstance(exc_info.value.cause, ClientResponseError)
            assert exc_info.value.cause.status == 401
            assert store.stream_errors['mss_direct'] == 1
    finally:
        await store.close()


async def test_stream_missing(source_root, pool: Executor):
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'
    try:
        with aioresponses() as m:
            source = MSSSourceFile(emu_irn, file, False)
            m.get(source.url, status=404)
            m.get(source.dams_url, status=404)
            with pytest.raises(StoreStreamError) as exc_info:
                async for chunk in store.stream(source):
                    pass
            assert exc_info.value.url.endswith(source.dams_url)
            assert exc_info.value.source is source
            assert isinstance(exc_info.value.cause, ClientResponseError)
            assert exc_info.value.cause.status == 404
            assert store.stream_errors['mss_indirect'] == 1
    finally:
        await store.close()


async def test_stream_use_dams(source_root: Path, pool: Executor):
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'
    dams_host = 'http://not.the.real.dams'
    dams_url = f'{dams_host}/some_original.tiff'
    content = b'1234 look at me i am a tiff image'
    chunk_size = 2
    try:
        with aioresponses() as m:
            source = MSSSourceFile(emu_irn, file, True, chunk_size=chunk_size)
            m.get(source.url, status=404)
            m.get(source.dams_url, body=dams_url)
            m.get(dams_url, body=content)

            buffer = BytesIO()
            async for chunk in store.stream(source):
                buffer.write(chunk)
                assert len(chunk) <= chunk_size
            assert buffer.getvalue() == content
    finally:
        await store.close()


async def test_stream_dams_fails(source_root: Path, pool: Executor):
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'
    dams_host = 'http://not.the.real.dams'
    dams_url = f'{dams_host}/some_original.tiff'
    try:
        with aioresponses() as m:
            source = MSSSourceFile(emu_irn, file, True)
            m.get(source.url, status=404)
            m.get(source.dams_url, body=dams_url)
            m.get(dams_url, status=404)

            with pytest.raises(StoreStreamError) as exc_info:
                async for chunk in store.stream(source):
                    pass
            assert exc_info.value.url == dams_url
            assert exc_info.value.source is source
            assert isinstance(exc_info.value.cause, ClientResponseError)
            assert exc_info.value.cause.status == 404
            assert store.stream_errors['dams'] == 1
    finally:
        await store.close()


async def test_use(source_root: Path, pool: Executor):
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'

    buffer = BytesIO()
    image = Image.new('RGB', size=(100, 100), color=(255, 0, 0))
    image.save(buffer, 'png')

    try:
        with aioresponses() as m:
            source = MSSSourceFile(emu_irn, file, False)
            m.get(source.url, body=buffer.getvalue())
            async with store.use(source) as path:
                assert path.exists()
                with Image.open(path) as image:
                    assert image.format.lower() == 'jpeg'
    finally:
        await store.close()


async def test_get_file_size(source_root: Path, pool: Executor):
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'
    content_length = 489345

    with aioresponses() as m:
        source = MSSSourceFile(emu_irn, file, False)
        m.get(source.url, headers={'content-length': str(content_length)})
        assert await store.get_file_size(source) == content_length


async def test_get_file_size_fail(source_root: Path, pool: Executor):
    store = MSSSourceStore(source_root, pool, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'

    with aioresponses() as m:
        source = MSSSourceFile(emu_irn, file, False)
        m.get(source.url, headers={})
        with pytest.raises(StoreStreamNoLength):
            assert await store.get_file_size(source)
