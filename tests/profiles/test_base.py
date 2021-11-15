from iiif.profiles.base import ImageInfo


class TestImageInfo:

    def test_size(self):
        assert ImageInfo('test', 'image1', 100, 400).size == (100, 400)

    def test_identifier(self):
        assert ImageInfo('test', 'image1', 100, 400).identifier == 'test:image1'

    def test_equality(self):
        info1 = ImageInfo('test', 'image1', 100, 400)
        info2 = ImageInfo('test', 'image1', 200, 500)
        # the identifier is the only attribute compared so width and height don't matter
        assert info1 == info2

    def test_hash(self):
        info1 = ImageInfo('test', 'image1', 100, 400)
        info2 = ImageInfo('test', 'image2', 100, 400)
        infos = {info1: 'beans'}
        assert info2 not in infos
        assert info1 in infos
