import pytest

from iiif.config import Config


@pytest.fixture
def config(tmp_path):
    return Config(
        base_url='http://localhost',
        source_path=tmp_path / 'source',
        cache_path=tmp_path / 'cache',
        profiles={
            'test': {
                'type': 'disk',
                'rights': 'http://creativecommons.org/licenses/by/4.0/'
            }
        }
    )
