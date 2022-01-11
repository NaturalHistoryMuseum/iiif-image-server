#!/usr/bin/env python3
# encoding: utf-8

from unittest.mock import AsyncMock, patch

import json
import pytest

from iiif.profiles import MSSProfile
from iiif.profiles.mss import MSSImageInfo
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


def create_es_mss_doc(doc):
    if doc is None:
        es_doc = {'found': False}
    else:
        es_doc = {'found': True, '_source': doc}
    return json.dumps(es_doc)


@pytest.fixture
def mss_profile(config):
    return MSSProfile('test', config, 'http://creativecommons.org/licenses/by/4.0/', [''], '', 1,
                      1, ['collections'])


# TODO: write more and better mss profile tests

class TestMSSProfileGetInfo:

    @pytest.mark.asyncio
    async def test_allowed(self, mss_profile):
        mss_doc = {
            'id': 1234,
            'file': 'beans.tiff',
            'width': 4000,
            'height': 1600,
        }
        mock_get_mss_doc = AsyncMock(return_value=mss_doc)
        with patch.object(mss_profile, 'get_mss_doc', mock_get_mss_doc):
            info = await mss_profile.get_info('testname')
            assert info is not None
            assert info.name == 'testname'
            assert info.size == (4000, 1600)
            assert info.original == 'beans.tiff'

    @pytest.mark.asyncio
    async def test_missing_collections_doc(self, mss_profile):
        mock_get_mss_doc = AsyncMock(return_value=None)
        with patch.object(mss_profile, 'get_mss_doc', mock_get_mss_doc):
            info = await mss_profile.get_info('1234')
            assert info is None

    @pytest.mark.asyncio
    async def test_missing_size(self, config, mss_profile):
        mss_doc = {
            'id': 1234,
            'file': 'beans.tiff',
        }
        mock_get_mss_doc = AsyncMock(return_value=mss_doc)
        with patch.object(mss_profile, 'get_mss_doc', mock_get_mss_doc):
            source_path = create_image(config, 140, 504, 'mss', 'test')
            mss_profile.fetch_source = AsyncMock(return_value=source_path)
            info = await mss_profile.get_info('test')
            assert info is not None
            assert info.size == (140, 504)
