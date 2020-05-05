import os
from PIL import Image

from iiif.image import IIIFImage


def create_image_data(image, width, height):
    """
    Create an image on disk using pillow at the image's source path with the given dimensions.

    :param image: an IIIFImage instance
    :param width: the width of the image to create
    :param height: the height of the image to create
    :return: the pillow Image object
    """
    os.makedirs(os.path.dirname(image.source_path), exist_ok=True)
    img = Image.new('RGB', (width, height), color='red')
    img.save(image.source_path, format='jpeg')
    return img


def create_image(config, width, height, identifier='test:image'):
    """
    Create a test IIIFImage object and a real image file and return the IIIFImage object.

    :param config: the config dict
    :param width: the width of the image to create
    :param height: the height of the image to create
    :param identifier: the IIIF identifier to use for the image
    :return: the IIIFImage object
    """
    image = IIIFImage(identifier, config['source_path'], config['cache_path'])
    create_image_data(image, width, height)
    return image
