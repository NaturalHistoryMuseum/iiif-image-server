#!/usr/bin/env python3
# encoding: utf-8

import pytest

from iiif.config import Config


@pytest.fixture
def config(tmp_path):
    return Config(
        base_url='http://localhost',
        source_path=str(tmp_path / 'source'),
        cache_path=str(tmp_path / 'cache'),
        default_profile='test',
        profiles={
            'test': {
                'type': 'disk',
                'rights': 'http://creativecommons.org/licenses/by/4.0/'
            }
        }
    )
