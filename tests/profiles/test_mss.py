#!/usr/bin/env python3
# encoding: utf-8

from unittest.mock import MagicMock, AsyncMock

import orjson
import pytest
from contextlib import asynccontextmanager

from iiif.profiles import MSSProfile
from iiif.profiles.mss import MSSImageInfo
from tests.utils import create_image


def test_mss_choose_file_no_derivatives():
    doc = {
        'file': 'original.tif',
        'width': 1000,
        'height': 2000,
    }
    info = MSSImageInfo('test', 'image', doc)
    assert info.choose_file() == doc['file']
    assert info.choose_file((200, 400)) == doc['file']


def test_mss_choose_file_with_derivatives():
    doc = {
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


def create_es_mss_doc(doc):
    if doc is None:
        es_doc = {'found': False}
    else:
        es_doc = {'found': True, '_source': doc}
    return orjson.dumps(es_doc)


@asynccontextmanager
async def mock_mss_profile(config, assoc_media_count, mss_doc, aps_is_ok):
    profile = MSSProfile('test', config, 'http://creativecommons.org/licenses/by/4.0/', [''], '', 1,
                         1, ['collections'])

    count_doc = {'count': assoc_media_count}
    es_post_mock_response = AsyncMock(text=AsyncMock(return_value=orjson.dumps(count_doc)))
    es_post_mock = AsyncMock(return_value=es_post_mock_response)

    es_get_mock_response = AsyncMock(text=AsyncMock(return_value=create_es_mss_doc(mss_doc)))
    es_get_mock = AsyncMock(return_value=es_get_mock_response)

    mss_get_mock = AsyncMock(return_value=MagicMock(ok=aps_is_ok))

    original_sessions = (profile.es_session, profile.mss_session, profile.dm_session)
    profile.es_session = MagicMock(
        get=MagicMock(return_value=MagicMock(__aenter__=es_get_mock)),
        post=MagicMock(return_value=MagicMock(__aenter__=es_post_mock))
    )
    profile.mss_session = MagicMock(
        get=MagicMock(return_value=MagicMock(__aenter__=mss_get_mock)),
    )

    yield profile

    profile.es_session, profile.mss_session, profile.dm_session = original_sessions
    await profile.close()


# TODO: write more and better mss profile tests

class TestMSSProfileGetInfo:

    @pytest.mark.asyncio
    async def test_allowed(self, config):
        mss_doc = {
            'id': 1234,
            'file': 'beans.tiff',
            'width': 4000,
            'height': 1600,
        }
        async with mock_mss_profile(config, 1, mss_doc, True) as profile:
            info = await profile.get_info('1234')
            assert info is not None
            assert info.name == '1234'
            assert info.size == (4000, 1600)
            assert info.original == 'beans.tiff'

    @pytest.mark.asyncio
    async def test_missing_collections_doc(self, config):
        mss_doc = {
            'id': 1234,
            'file': 'beans.tiff',
            'width': 4000,
            'height': 1600,
        }
        async with mock_mss_profile(config, 0, mss_doc, True) as profile:
            info = await profile.get_info('1234')
            assert info is None

    @pytest.mark.asyncio
    async def test_aps_denies(self, config):
        mss_doc = {
            'id': 1234,
            'file': 'beans.tiff',
            'width': 4000,
            'height': 1600,
        }
        async with mock_mss_profile(config, 1, mss_doc, False) as profile:
            info = await profile.get_info('1234')
            assert info is None

    @pytest.mark.asyncio
    async def test_missing_mss_doc(self, config):
        async with mock_mss_profile(config, 1, None, True) as profile:
            info = await profile.get_info('1234')
            assert info is None

    @pytest.mark.asyncio
    async def test_missing_size(self, config):
        mss_doc = {
            'id': 1234,
            'file': 'beans.tiff',
        }
        async with mock_mss_profile(config, 1, mss_doc, True) as profile:
            source_path = create_image(config, 140, 504, 'mss', '1234')
            profile.fetch_source = AsyncMock(return_value=source_path)
            info = await profile.get_info('1234')
            assert info is not None
            assert info.size == (140, 504)
