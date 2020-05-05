import pytest


# global config fixture
@pytest.fixture
def config(tmp_path, base_url):
    return {
        'base_url': base_url,
        'cache_path': tmp_path / 'cache',
        'source_path': tmp_path / 'source',
        'min_sizes_size': 200,
        'size_pool_size': 1,
        'image_pool_size': 1,
        'info_cache_size': 1,
        'image_cache_size_per_process': 1,
        'types': {
            'test': {
                'source': 'disk',
            }
        }
    }
