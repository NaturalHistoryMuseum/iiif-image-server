#!/usr/bin/env python3
# encoding: utf-8

import os
from PIL import Image
from concurrent.futures.process import ProcessPoolExecutor
from tornado.concurrent import Future
from tornado.httpclient import AsyncHTTPClient
from tornado.ioloop import IOLoop
from tornado.web import HTTPError


class ImageSourceSizer:

    def __init__(self, config):
        """
        :param config: the config
        """
        # initialise a process pool to run the image dimension extraction function in
        self.pool = ProcessPoolExecutor(max_workers=config['size_pool_size'])

    @staticmethod
    def _get_image_size(image):
        """
        Returns the width and height dimensions of the given image using Pillow. This should be very
        fast as Pillow should only need to read the first few bytes of the file to determine the
        size, however, if the file is not an image file or corrupt it could take longer, hence it's
        probably a good idea to run this function in a separate thread or process from the main
        event loop thread.

        This function assumes the image's source has been loaded and exists on disk at the image's
        source path.

        :param image: the Image object
        :return: a 2-tuple containing the width and the height
        """
        with Image.open(image.source_path) as pillow_image:
            return pillow_image.width, pillow_image.height

    async def get_image_size(self, image):
        """
        Returns the width and height dimensions of the given image using Pillow. Extraction is done
        asynchronously using a process pool.

        :param image: the Image object
        :return: a 2-tuple containing the width and the height
        """
        return await IOLoop.current().run_in_executor(self.pool, self._get_image_size, image)

    def stop(self):
        """
        Stop the process pool.
        """
        self.pool.shutdown()


class ImageSourceFetcher:
    """
    Provides a central place to fetch source images and ensure they're on disk.
    """

    # these are the types of image source we currently support
    supported_types = {'vfactor', 'mam'}

    def __init__(self, config):
        """
        :param config: the config
        """
        self.config = config
        # a register of the source images and their load status
        self.images = {}

    async def ensure_source_exists(self, image):
        """
        Ensures the source file for the given image is on disk. This function is safe to await
        multiple times for the same source as it will ensure the source is only retrieved once (if
        indeed it needs to be retrieved from say an external service).

        Once this function returns after being awaited it is safe to assume the source file exists.

        :param image: the Image object
        """
        if image.identifier in self.images:
            # the source file has either already been retrieved or is currently being retrieved,
            # await the original request's result
            await self.images[image.identifier]

        # we represent that state of the source file as a future, it will be resolved once we either
        # have the source file or couldn't get it and therefore need to raise an exception
        self.images[image.identifier] = Future()
        if not os.path.exists(image.source_path):
            if image.type == 'mam':
                await self._fetch_mam_image(image)
            elif image.type == 'vfactor':
                # if the vfactor image isn't on disk we have no way to retrieve so instant error!
                exception = HTTPError(status_code=404, reason=f'VFactor image not found')
                self.images[image.identifier].set_exception(exception)
                raise exception
            else:
                raise HTTPError(status_code=500, reason="Identifier type could not be retrieved")

        # complete the future as the source file should be on disk now
        self.images[image.identifier].set_result(None)

    async def _fetch_mam_image(self, image):
        """
        Fetch the image from MAM using the name as the asset ID. Currently we can only support image
        requests using the media store's "preview" size which isn't great but I guess it's better
        than nothing!

        :param image: the Image object
        """
        http_client = AsyncHTTPClient()
        response = await http_client.fetch(f'https://www.nhm.ac.uk/services/media-store/'
                                           f'asset/{image.name}/contents/preview',
                                           raise_error=False)
        if response.code != 200:
            exception = HTTPError(status_code=404, reason=f'MAM image not found ({response.code})')
            self.images[image.identifier].set_exception(exception)
            raise exception

        os.makedirs(os.path.dirname(image.source_path), exist_ok=True)
        with open(os.path.join(image.source_path), 'wb') as f:
            f.write(response.body)


class IIIFImage:
    """
    This class represents an image identified by a IIIF identifier from a request URL.
    """

    def __init__(self, identifier, root_source_path, root_cache_path):
        """
        :param identifier: the IIIF identifier, this should be in the format type:name
        :param root_source_path: the root source path for all source images
        :param root_cache_path: the root cache path for all cache images
        """
        if ':' not in identifier:
            raise HTTPError(status_code=404, reason="Identifier type not specified")

        self.identifier = identifier
        self.type, self.name = identifier.split(':', 1)

        if self.type not in ImageSourceFetcher.supported_types:
            raise HTTPError(status_code=404, reason="Identifier type not found")

        self.source_path = os.path.join(root_source_path, self.type, self.name)
        self.cache_path = os.path.join(root_cache_path, self.type, self.name)
