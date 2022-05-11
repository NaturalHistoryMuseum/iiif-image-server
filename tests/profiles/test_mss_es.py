from aioresponses import aioresponses

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


async def test_cycling_hosts():
    host_1 = 'http://not.a.real.es.host1'
    mock_response_1 = {'hits': {'total': 1, 'hits': [{'_source': {'some': 'data'}}]}}
    host_2 = 'http://not.a.real.es.host2'
    mock_response_2 = {'hits': {'total': 0, 'hits': []}}
    handler = MSSElasticsearchHandler([host_1, host_2], ['index-1', 'index-2'],
                                      mss_index=MSS_INDEX)
    try:
        with aioresponses() as m:
            m.post(f'{host_1}/{handler.mss_index}/_search', payload=mock_response_1)
            m.post(f'{host_2}/{handler.mss_index}/_search', payload=mock_response_2)
            m.post(f'{host_1}/{handler.mss_index}/_search', payload=mock_response_1)
            assert (await handler.get_mss_doc('12345'))[0] == 1
            assert (await handler.get_mss_doc('12345'))[0] == 0
            assert (await handler.get_mss_doc('12345'))[0] == 1
    finally:
        await handler.close()
