from iiif.exceptions import profile_not_found, too_many_images, image_not_found


def test_profile_not_found():
    assert profile_not_found().status_code == 404


def test_image_not_found():
    assert image_not_found().status_code == 404


def test_too_many_images():
    error = too_many_images(1029)
    assert error.status_code == 400
    assert '1029' in error.detail
