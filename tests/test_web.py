import hashlib
import json
import pytest
from tornado.httpclient import HTTPClientError

from iiif.web import create_application
from tests.utils import create_image


@pytest.fixture
def app(config):
    with create_application(config) as application:
        yield application


@pytest.mark.gen_test
async def test_image_data_handler(config, http_client, base_url):
    identifier = 'vfactor:image'
    image = create_image(config, 400, 500, identifier=identifier)
    with open(image.source_path, 'rb') as f:
        original_image_hash = hashlib.sha256(f.read()).digest()

    response = await http_client.fetch(f'{base_url}/{identifier}/full/max/0/default.jpg')
    assert response.code == 200

    assert hashlib.sha256(response.body).digest() == original_image_hash


@pytest.mark.gen_test
async def test_image_info_handler(config, http_client, base_url):
    identifier = 'vfactor:image'
    create_image(config, 400, 500, identifier=identifier)

    response = await http_client.fetch(f'{base_url}/{identifier}/info.json')
    assert response.code == 200

    info = json.loads(response.body)
    assert info['id'] == f'{base_url}/{identifier}'
    assert info['@id'] == f'{base_url}/{identifier}'
    assert info['@context'] == 'http://iiif.io/api/image/3/context.json'
    assert info['type'] == 'ImageService3'
    assert info['protocol'] == 'http://iiif.io/api/image'
    assert info['rights'] == 'http://creativecommons.org/licenses/by/4.0/'
    assert info['profile'] == 'level1'
    assert info['width'] == 400
    assert info['height'] == 500
    assert info['maxWidth'] == 400
    assert info['maxHeight'] == 500
    assert info['tiles'] == [
        {'width': 512, 'scaleFactors': [1, 2, 4, 8, 16]},
        {'width': 256, 'scaleFactors': [1, 2, 4, 8, 16]},
        {'width': 1024, 'scaleFactors': [1, 2, 4, 8, 16]},
    ]
    assert info['sizes'] == [
        {"width": 400, "height": 500},
        {"width": 200, "height": 250},
    ]


@pytest.mark.gen_test
async def test_image_info_handler_no_image(http_client, base_url):
    identifier = 'vfactor:image'

    with pytest.raises(HTTPClientError) as e:
        await http_client.fetch(f'{base_url}/{identifier}/info.json')
        assert e.value.code == 404

    with pytest.raises(HTTPClientError) as e:
        await http_client.fetch(f'{base_url}/{identifier}/full/max/0/default.jpg')
        assert e.value.code == 404


@pytest.mark.gen_test
async def test_image_info_handler_invalid_type(http_client, base_url):
    identifier = 'banana:image'

    with pytest.raises(HTTPClientError) as e:
        await http_client.fetch(f'{base_url}/{identifier}/info.json')
        assert e.value.code == 404

    with pytest.raises(HTTPClientError) as e:
        await http_client.fetch(f'{base_url}/{identifier}/full/max/0/default.jpg')
        assert e.value.code == 404


@pytest.mark.gen_test
async def test_image_info_handler_unsupported_iiif_features(config, http_client, base_url):
    identifier = 'vfactor:image'
    create_image(config, 400, 500, identifier=identifier)

    with pytest.raises(HTTPClientError) as e:
        await http_client.fetch(f'{base_url}/{identifier}/full/max/90/default.jpg')
        assert e.value.code == 404

    with pytest.raises(HTTPClientError) as e:
        await http_client.fetch(f'{base_url}/{identifier}/full/max/0/bitonal.jpg')
        assert e.value.code == 404
