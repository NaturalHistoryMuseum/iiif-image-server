#!/usr/bin/env python3
# encoding: utf-8
import asyncio
import pytest
from contextlib import asynccontextmanager
from fastapi import HTTPException
from pathlib import Path
from typing import Optional, Union
from unittest.mock import AsyncMock, patch, Mock, MagicMock

from iiif.config import Config
from iiif.exceptions import ImageNotFound
from iiif.profiles.mss import MSSImageInfo, MSSProfile
from tests.utils import create_image


def test_mss_choose_file_no_derivatives():
    doc = {
        'id': 23,
        'file': 'original.tif',
        'width': 1000,
        'height': 2000,
    }
    info = MSSImageInfo('test', 'image', doc)
    assert info.choose_file() == doc['file']
    assert info.choose_file((200, 400)) == doc['file']


def test_mss_choose_file_with_derivatives():
    doc = {
        'id': 23,
        'file': 'original.tif',
        'width': 1000,
        'height': 2000,
        'derivatives': [
            dict(file='small.jpg', width=100, height=200),
            dict(file='medium.jpg', width=500, height=1000),
            dict(file='large.jpg', width=900, height=1800),
            dict(file='original.jpg', width=1000, height=2000),
        ]
    }
    info = MSSImageInfo('test', 'image', doc)

    assert info.choose_file() == 'original.tif'
    assert info.choose_file(None) == 'original.tif'
    assert info.choose_file((50, 100)) == 'small.jpg'
    assert info.choose_file((100, 200)) == 'small.jpg'
    assert info.choose_file((300, 600)) == 'medium.jpg'
    assert info.choose_file((500, 1000)) == 'medium.jpg'
    assert info.choose_file((684, 1368)) == 'large.jpg'
    assert info.choose_file((900, 1800)) == 'large.jpg'
    assert info.choose_file((1000, 2000)) == 'original.jpg'
    assert info.choose_file((2000, 4000)) == 'original.tif'


def create_profile(config: Config, mss_doc: Union[Exception, dict], mss_valid: bool,
                   image: Union[Exception, Path], **kwargs) -> MSSProfile:
    mock_es_handler = MagicMock(
        close=AsyncMock()
    )
    if isinstance(mss_doc, Exception):
        mock_es_handler.configure_mock(get_mss_doc=AsyncMock(side_effect=mss_doc))
    else:
        mock_es_handler.configure_mock(get_mss_doc=AsyncMock(return_value=(1, mss_doc)))

    mock_store = MagicMock(
        check_access=AsyncMock(return_value=mss_valid),
        close=AsyncMock()
    )

    @asynccontextmanager
    async def use(*a, **k):
        if isinstance(image, Exception):
            raise image
        else:
            yield image

    async def data_iter(*stream_args, **stream_kwargs):
        if isinstance(image, Exception):
            raise image
        else:
            for byte in image.read_bytes():
                yield bytes([byte])

    mock_store.configure_mock(stream=data_iter, use=use)

    with patch('iiif.profiles.mss.MSSElasticsearchHandler', Mock(return_value=mock_es_handler)):
        with patch('iiif.profiles.mss.MSSSourceStore', Mock(return_value=mock_store)):
            return MSSProfile('test', config, 'some-rights-yo', Mock(), Mock(), Mock(), **kwargs)


def create_mss_doc(emu_irn: int, file: str, width: Optional[int] = None,
                   height: Optional[int] = None, *derivatives: dict) -> dict:
    doc = {
        'id': emu_irn,
        'file': file,
    }
    if width:
        doc['width'] = width
    if height:
        doc['height'] = height
    if derivatives:
        doc['derivatives'] = list(derivatives)
    return doc


def create_derivative(width, height, file) -> dict:
    return {'width': width, 'height': height, 'file': file}


class TestGetInfo:

    async def test_no_source_required(self, config):
        mss_doc = create_mss_doc(7, 'image.jpg', 200, 300)
        profile = create_profile(config, mss_doc, True, Exception('should not need this'))
        info = await profile.get_info('the_name')
        assert info.emu_irn == 7
        assert info.width == 200
        assert info.height == 300
        assert info.original == 'image.jpg'
        assert info.derivatives == []

    async def test_source_required(self, config):
        mss_doc = create_mss_doc(7, 'image.jpg')
        image_path = create_image(config, 200, 300)
        profile = create_profile(config, mss_doc, True, image_path)
        info = await profile.get_info('the_name')
        assert info.emu_irn == 7
        assert info.width == 200
        assert info.height == 300
        assert info.original == 'image.jpg'
        assert info.derivatives == []

    async def test_doc_error(self, config):
        image = create_image(config, 200, 300)
        exception = Exception('narp')
        profile = create_profile(config, exception, True, image)
        with pytest.raises(ImageNotFound) as exc_info1:
            await profile.get_info('the_name')
        assert exc_info1.value.cause is exception

    async def test_cache(self, config):
        mss_doc = create_mss_doc(7, 'image.jpg', 200, 300)
        profile = create_profile(config, mss_doc, True, Exception('should not need this'))
        with patch.object(profile, 'get_mss_doc', wraps=profile.get_mss_doc):
            await profile.get_info('the_name')
            assert profile.get_mss_doc.call_count == 1
            await profile.get_info('the_name')
            assert profile.get_mss_doc.call_count == 1

    async def test_source_error(self, config):
        mss_doc = create_mss_doc(7, 'image.jpg')
        exception = Exception('errrrooorr!')
        profile = create_profile(config, mss_doc, True, exception)
        with pytest.raises(ImageNotFound) as exc_info:
            await profile.get_info('the_name')
        assert exc_info.value.cause is exception

    async def test_get_size_error(self, config):
        mss_doc = create_mss_doc(7, 'image.jpg')
        image = create_image(config, 200, 300)
        profile = create_profile(config, mss_doc, True, image)
        exception = Exception('nope!')
        with patch('iiif.profiles.mss.get_size', MagicMock(side_effect=exception)):
            with pytest.raises(ImageNotFound) as exc_info:
                await profile.get_info('the_name')
            assert exc_info.value.cause is exception


async def test_close(config):
    profile = create_profile(config, Exception('meh'), True, Exception('meh'))
    await profile.close()
    profile.es_handler.close.assert_called_once()
    profile.store.close.assert_called_once()
