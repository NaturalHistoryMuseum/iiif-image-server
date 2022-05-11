from contextlib import closing

import pytest
from aioresponses import aioresponses
from fastapi import HTTPException

from iiif.profiles.mss import MSSElasticsearchHandler

MOCK_HOST = 'http://not.a.real.es.host'
MSS_INDEX = 'mss'


async def test_get_mss_doc():
    mock_doc = {'a': 'doc'}
    mock_response = {'hits': {'total': 1, 'hits': [{'_source': mock_doc}]}}
    handler = MSSElasticsearchHandler([MOCK_HOST], ['index-1', 'index-2'], mss_index=MSS_INDEX)
    try:
        with aioresponses() as m:
            m.post(f'{MOCK_HOST}/{MSS_INDEX}/_search', payload=mock_response)
            total, doc = await handler.get_mss_doc('some-guid')
            assert total == 1
            assert doc == mock_doc
    finally:
        await handler.close()


async def test_get_mss_doc_none_found():
    mock_response = {'hits': {'total': 0, 'hits': []}}
    handler = MSSElasticsearchHandler([MOCK_HOST], ['index-1', 'index-2'], mss_index=MSS_INDEX)
    try:
        with aioresponses() as m:
            m.post(f'{MOCK_HOST}/{MSS_INDEX}/_search', payload=mock_response)
            total, doc = await handler.get_mss_doc('some-guid')
            assert total == 0
            assert doc is None
    finally:
        await handler.close()


async def test_get_mss_doc_many_found():
    mock_docs = [{'a': 'doc'}, {'another': 'doc'}]
    mock_response = {'hits': {'total': len(mock_docs),
                              'hits': [{'_source': mock_doc} for mock_doc in mock_docs]}}
    handler = MSSElasticsearchHandler([MOCK_HOST], ['index-1', 'index-2'], mss_index=MSS_INDEX)
    try:
        with aioresponses() as m:
            m.post(f'{MOCK_HOST}/{MSS_INDEX}/_search', payload=mock_response)
            total, doc = await handler.get_mss_doc('some-guid')
            assert total == len(mock_docs)
            assert doc == mock_docs[0]
    finally:
        await handler.close()


@pytest.mark.parametrize('count', [1, 2, 10347])
async def test_has_collection_record(count):
    mock_response = {'count': count}
    handler = MSSElasticsearchHandler([MOCK_HOST], ['index-1', 'index-2'], mss_index=MSS_INDEX)
    try:
        with aioresponses() as m:
            m.post(f'{MOCK_HOST}/{handler.collection_indices}/_count', payload=mock_response)
            assert await handler.has_collection_record('12345')
    finally:
        await handler.close()


async def test_has_collection_record_no_hits():
    mock_response = {'count': 0}
    handler = MSSElasticsearchHandler([MOCK_HOST], ['index-1', 'index-2'], mss_index=MSS_INDEX)
    try:
        with aioresponses() as m:
            m.post(f'{MOCK_HOST}/{handler.collection_indices}/_count', payload=mock_response)
            assert not await handler.has_collection_record('12345')
    finally:
        await handler.close()


async def test_cycling_hosts():
    host_1 = 'http://not.a.real.es.host1'
    mock_response_1 = {'count': 1}
    host_2 = 'http://not.a.real.es.host2'
    mock_response_2 = {'count': 0}
    handler = MSSElasticsearchHandler([host_1, host_2], ['index-1', 'index-2'],
                                      mss_index=MSS_INDEX)
    try:
        with aioresponses() as m:
            m.post(f'{host_1}/{handler.collection_indices}/_count', payload=mock_response_1)
            m.post(f'{host_2}/{handler.collection_indices}/_count', payload=mock_response_2)
            m.post(f'{host_1}/{handler.collection_indices}/_count', payload=mock_response_1)
            assert await handler.has_collection_record('12345')
            assert not await handler.has_collection_record('12345')
            assert await handler.has_collection_record('12345')
    finally:
        await handler.close()
