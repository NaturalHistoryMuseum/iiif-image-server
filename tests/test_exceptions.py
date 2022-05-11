from iiif.exceptions import ProfileNotFound, TooManyImages, ImageNotFound, InvalidIIIFParameter


def test_profile_not_found():
    assert ProfileNotFound('test').status_code == 404


def test_image_not_found():
    assert ImageNotFound('test', 'image').status_code == 404


def test_too_many_images():
    error = TooManyImages(1029)
    assert error.status_code == 400
    assert '1029' in error.public


def test_invalid_iiif_parameter():
    error = InvalidIIIFParameter('Goats', 'Beans')
    assert error.status_code == 400
    assert 'Goats' in error.public
    assert 'Beans' in error.public
