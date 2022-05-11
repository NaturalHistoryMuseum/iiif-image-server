import pytest
from PIL import Image
from aiohttp import ClientResponseError
from aioresponses import aioresponses
from fastapi import HTTPException
from io import BytesIO
from pathlib import Path

from iiif.profiles.mss import MSSSourceStore, MSSSourceFile

MOCK_HOST = 'http://not.the.real.mss.host'


@pytest.fixture
def source_root(tmpdir) -> Path:
    return Path(tmpdir, 'test')


async def test_choose_convert_pool(source_root: Path):
    store = MSSSourceStore(source_root, MOCK_HOST, 10, 10, 10)
    try:
        assert store._choose_convert_pool('beans.jpg') == store._fast_pool
        assert store._choose_convert_pool('beans.jpeg') == store._fast_pool
        assert store._choose_convert_pool('beans.tiff') == store._slow_pool
        assert store._choose_convert_pool('beans.any') == store._slow_pool
    finally:
        await store.close()


async def test_check_access_ok(source_root: Path):
    store = MSSSourceStore(source_root, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    try:
        with aioresponses() as m:
            m.get(MSSSourceFile.check_url(emu_irn), status=204)
            assert await store.check_access(emu_irn)
    finally:
        await store.close()


async def test_check_access_failed(source_root: Path):
    store = MSSSourceStore(source_root, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    try:
        with aioresponses() as m:
            m.get(MSSSourceFile.check_url(emu_irn), status=401)
            assert not await store.check_access(emu_irn)
    finally:
        await store.close()


async def test_stream(source_root: Path):
    store = MSSSourceStore(source_root, MOCK_HOST, 10, 10, 10)
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


async def test_stream_access_denied(source_root):
    source_root: Path
    store = MSSSourceStore(source_root, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'
    try:
        with aioresponses() as m:
            source = MSSSourceFile(emu_irn, file, False)
            m.get(source.url, status=401)
            with pytest.raises(HTTPException):
                async for chunk in store.stream(source):
                    pass
    finally:
        await store.close()


async def test_stream_missing(source_root):
    store = MSSSourceStore(source_root, MOCK_HOST, 10, 10, 10)
    emu_irn = 12345
    file = 'some_file.jpg'
    try:
        with aioresponses() as m:
            source = MSSSourceFile(emu_irn, file, False)
            m.get(source.url, status=404)
            with pytest.raises(ClientResponseError):
                async for chunk in store.stream(source):
                    pass
    finally:
        await store.close()


async def test_stream_use_dams(source_root: Path):
    store = MSSSourceStore(source_root, MOCK_HOST, 10, 10, 10)
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


async def test_stream_dams_fails(source_root: Path):
    store = MSSSourceStore(source_root, MOCK_HOST, 10, 10, 10)
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

            with pytest.raises(ClientResponseError):
                async for chunk in store.stream(source):
                    pass
    finally:
        await store.close()


async def test_use(source_root: Path):
    store = MSSSourceStore(source_root, MOCK_HOST, 10, 10, 10)
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