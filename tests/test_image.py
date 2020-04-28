import os
from unittest.mock import MagicMock

import pytest
from tornado.web import HTTPError

from iiif.image import IIIFImage


class TestIIIFImage:

    def test_colon_must_exist(self):
        with pytest.raises(HTTPError) as e:
            IIIFImage('nocolon', MagicMock(), MagicMock())

        assert e.value.status_code == 404
        assert e.value.reason == 'Identifier type not specified'

    def test_type_must_be_supported(self):
        with pytest.raises(HTTPError) as e:
            IIIFImage('beans:id', MagicMock(), MagicMock())

        assert e.value.status_code == 404
        assert e.value.reason == 'Identifier type not supported'

    def test_it_does_work(self):
        image = IIIFImage('vfactor:animage', 'root', 'cache')

        assert image.type == 'vfactor'
        assert image.name == 'animage'
        assert image.source_path == os.path.join('root', image.type, image.name)
        assert image.cache_path == os.path.join('cache', image.type, image.name)
