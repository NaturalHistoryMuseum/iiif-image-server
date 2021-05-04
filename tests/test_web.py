#!/usr/bin/env python3
# encoding: utf-8

import hashlib
import pytest
from starlette.testclient import TestClient
from unittest.mock import patch, MagicMock

from iiif.web import app
from tests.utils import create_image


@pytest.fixture
def test_client(config):
    with patch('iiif.web.load_config', MagicMock(return_value=config)):
        with TestClient(app) as client:
            yield client


def test_image_data_handler(test_client, config):
    profile = 'test'
    name = 'image'
    image = create_image(config, 400, 500, profile, name)

    with image.open('rb') as f:
        original_image_hash = hashlib.sha256(f.read()).digest()

    response = test_client.get(f'/{profile}:{name}/full/max/0/default.jpg')
    assert response.status_code == 200
    assert hashlib.sha256(response.content).digest() == original_image_hash


def test_image_info_handler(test_client, config):
    profile = 'test'
    name = 'image'
    create_image(config, 400, 500, profile, name)

    response = test_client.get(f'/{profile}:{name}/info.json')
    assert response.status_code == 200

    info = response.json()
    assert info['id'] == f'{config.base_url}/{profile}:{name}'
    assert info['@id'] == f'{config.base_url}/{profile}:{name}'
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


def test_status_succeeds(test_client):
    response = test_client.get('/status')
    assert response.status_code == 200
    assert response.json()['status'] == ':)'


def test_unrecognised_profile(test_client, config):
    profile = 'banana'
    name = 'image'
    create_image(config, 400, 500, profile, name)

    response = test_client.get(f'/{profile}:{name}/info.json')
    assert response.status_code == 404

    response = test_client.get(f'/{profile}:{name}/full/max/0/default.jpg')
    assert response.status_code == 404


def test_unsupported_iiif_features(test_client, config):
    profile = 'test'
    name = 'image'
    create_image(config, 400, 500, profile, name)

    response = test_client.get(f'/{profile}:{name}/full/max/90/default.jpg')
    assert response.status_code == 404

    response = test_client.get(f'/{profile}:{name}/full/max/90/bitonal.jpg')
    assert response.status_code == 404
