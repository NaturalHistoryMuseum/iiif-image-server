from aioresponses import aioresponses

from iiif.profiles.mss import MSSElasticsearchHandler, rebuild_data

MOCK_HOST = 'http://not.a.real.es.host'
MSS_INDEX = 'mss'


def make_es_response(*hits):
    return {
        "hits": {
            "total": {"value": len(hits), "relation": "eq"},
            "hits": [{"_source": hit} for hit in hits],
        }
    }


async def test_get_mss_doc():
    mock_doc = {'data': {'a': {'_u': 'doc', '_t': 'doc', '_k': 'doc'}}}
    mock_response = make_es_response(mock_doc)
    handler = MSSElasticsearchHandler([MOCK_HOST], ['index-1', 'index-2'], mss_index=MSS_INDEX)
    try:
        with aioresponses() as m:
            m.post(f'{MOCK_HOST}/{MSS_INDEX}/_search', payload=mock_response)
            total, doc = await handler.get_mss_doc('some-guid')
            assert total == 1
            assert doc == rebuild_data(mock_doc['data'])
    finally:
        await handler.close()


async def test_get_mss_doc_none_found():
    mock_response = make_es_response()
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
    mock_docs = [{'data': {'a': {'_u': 'doc', '_t': 'doc', '_k': 'doc'}}}, {'data': {'a': {'_u': 'doc', '_t': 'doc', '_k': 'doc'}}}]
    mock_response = make_es_response(*mock_docs)
    handler = MSSElasticsearchHandler([MOCK_HOST], ['index-1', 'index-2'], mss_index=MSS_INDEX)
    try:
        with aioresponses() as m:
            m.post(f'{MOCK_HOST}/{MSS_INDEX}/_search', payload=mock_response)
            total, doc = await handler.get_mss_doc('some-guid')
            assert total == len(mock_docs)
            assert doc == rebuild_data(mock_docs[0]['data'])
    finally:
        await handler.close()


async def test_cycling_hosts():
    host_1 = 'http://not.a.real.es.host1'
    mock_response_1 = make_es_response({'data': {'a': {'_u': 'doc', '_t': 'doc', '_k': 'doc'}}})
    host_2 = 'http://not.a.real.es.host2'
    mock_response_2 = make_es_response()
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
