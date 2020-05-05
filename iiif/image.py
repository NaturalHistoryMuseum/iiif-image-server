#!/usr/bin/env python3
# encoding: utf-8
import base64
import functools
import os
import re
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

    def __init__(self, config):
        """
        :param config: the config
        """
        # the curl implementation is the fastest and best overall so force its use
        AsyncHTTPClient.configure("tornado.curl_httpclient.CurlAsyncHTTPClient",
                                  max_clients=config['max_http_fetches'])

        self.config = config
        # a register of the source images and their load status
        self.images = {}
        self.types = {}
        for supported_type, options in self.config['types'].items():
            self.types[supported_type] = (options.pop('source'), options)

    async def ensure_source_exists(self, image):
        """
        Ensures the source file for the given image is on disk. This function is safe to await
        multiple times for the same source as it will ensure the source is only retrieved once (if
        indeed it needs to be retrieved from say an external service).

        Once this function returns after being awaited it is safe to assume the source file exists.

        :param image: the Image object
        """
        if image.type not in self.types:
            raise HTTPError(status_code=400, reason='Identifier type not supported')

        if image.identifier in self.images:
            # the source file has either already been retrieved or is currently being retrieved,
            # await the original request's result
            await self.images[image.identifier]
            # get outta here, we're done!
            return

        # we represent the state of the source file as a future, it will be resolved once we either
        # have the source file or couldn't get it and therefore need to raise an exception
        self.images[image.identifier] = Future()
        if not os.path.exists(image.source_path):
            source_type, options = self.types[image.type]

            fetch_function = getattr(self, f'_fetch_{source_type}_image',
                                     functools.partial(self._source_not_supported, source_type))
            try:
                await fetch_function(image, **options)
            except Exception as exception:
                # set the exception on the future so that any future or concurrent requests for the
                # same task get the same exception
                self.images[image.identifier].set_exception(exception)
                # raise it for the current caller
                raise exception

        # complete the future as the source file is on disk now
        self.images[image.identifier].set_result(None)

    async def _source_not_supported(self, source_type, *args, **kwargs):
        """
        Default handler for requests where the source type is present in the config but doesn't have
        a corresponding fetch function. If a request gets here then there is a config error, not a
        user request error, hence the 500 response code.

        This function always raises a HTTPError error.

        :param source_type: the source type that didn't have a matching fetch function
        :raise: an HTTPError
        """
        raise HTTPError(status_code=500, reason=f'Identifier type {source_type} not supported')

    async def _fetch_disk_image(self, *args, **kwargs):
        """
        Fetch handler for already on disk images. These images are expected to have been preloaded
        into the correct location and therefore if they don't exist and we end up in this handler,
        something has gone wrong! Therefore, this function always just raises an HTTPError.

        :raise: an HTTPError
        """
        raise HTTPError(status_code=404, reason='Source image not found')

    async def _fetch_trusted_web_image(self, image, regex):
        """
        Fetch handler for "trusted web" images. This is a mechanism that allows semi-arbitrary URL
        images to be served through this server. Allowing requesters to just use a whole URL as the
        identifier and have the server just go request it would be bad for many reasons so this is
        the middle ground where a URL can be used as the identifier but it has to match the given
        regex (defined in the config) to be allowed through and actually requested by the server.

        The identifier must be a URL-safe, base64 encoded UTF-8 string.

        :param image: the IIIFImage object
        :param regex: the regex the URL must match to be fetched
        :raise: an HTTPError if the URL doesn't match the regex
        """
        url = base64.urlsafe_b64decode(image.name).decode('utf-8')
        if re.compile(regex).match(url):
            await self._fetch_web_image(image, url)
        else:
            raise HTTPError(status_code=400, reason='Type not matched')

    async def _fetch_web_image(self, image, url):
        """
        Fetch handler for web images. The name from the image parameter is used to `format` the URL
        and therefore can be used to complete a URL. This also means that the URL can be complete
        with no named "name" format placeholder.

        :param image: the IIIFImage object
        :param url: the URL to add the name to and then fetch
        """
        http_client = AsyncHTTPClient()
        response = await http_client.fetch(url.format(name=image.name), raise_error=False)
        if response.code != 200:
            raise HTTPError(status_code=404, reason=f'Source image not found ({response.code})')

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
        self.root_source_path = root_source_path
        self.root_cache_path = root_cache_path

    @property
    def source_path(self):
        return os.path.join(self.root_source_path, self.type, self.name)

    @property
    def cache_path(self):
        return os.path.join(self.root_cache_path, self.type, self.name)

    def __eq__(self, other):
        if isinstance(other, IIIFImage):
            # we can just use the paths for equivalence as they include everything
            return self.source_path == other.source_path and self.cache_path == other.cache_path
        return NotImplemented
