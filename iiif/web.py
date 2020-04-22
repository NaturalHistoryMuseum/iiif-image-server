#!/usr/bin/env python3
# encoding: utf-8

import os
import yaml
from PIL import Image
from functools import lru_cache
from itertools import count
from lru import LRU
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler, HTTPError

from iiif.image import ImageSourceFetcher, ImageSourceSizer, IIIFImage
from iiif.processing import ImageProcessingDispatcher, Task


class ImageDataHandler(RequestHandler):
    """
    Request handler for image data requests.
    """

    def initialize(self, config, dispatcher, image_source_fetcher):
        """
        Inits the handler with the global config, dispatcher object and the image manager.

        :param config: the config dict
        :param dispatcher: the ImageProcessingDispatcher instance
        :param image_source_fetcher: the ImageSourceFetcher object we can use to fetch source files
                                     with
        """
        self.config = config
        self.dispatcher = dispatcher
        self.image_source_fetcher = image_source_fetcher

    async def get(self, identifier, region, size):
        """
        Responds to IIIF image data requests.

        :param identifier: the IIIF identifier
        :param region: the requested region
        :param size: the requested size
        """
        image = IIIFImage(identifier, self.config['source_path'], self.config['cache_path'])
        # we need the image source file to exist on disk so that we can work on it
        await self.image_source_fetcher.ensure_source_exists(image)

        task = Task(image, region, size)
        # submit the task to the dispatcher and wait for it to complete
        await self.dispatcher.submit(task).wait()

        self.set_header("Content-type", "image/jpeg")
        with open(task.output_path, 'rb') as f:
            while True:
                # read the data in chunks of 64KiB
                data = f.read(65536)
                if not data:
                    break
                self.write(data)
                # flush to avoid reading the whole image into memory at once
                await self.flush()
        await self.finish()


class ImageInfoHandler(RequestHandler):
    """
    Request handler for info.json requests.
    """

    def initialize(self, config, info_cache, image_source_fetcher, image_source_sizer):
        """
        Inits the handler with the global config, info cache and info process pool instances.

        :param config: the config dict
        :param info_cache: LRU cache instance to store info.json responses in
        :param image_source_fetcher: the ImageSourceFetcher object we can use to fetch source files
                                     with
        :param image_source_sizer: the ImageSourceSizer object we can use to get the size of an
                                   image with
        """
        self.config = config
        self.info_cache = info_cache
        self.image_source_fetcher = image_source_fetcher
        self.image_source_sizer = image_source_sizer

    @staticmethod
    @lru_cache(maxsize=1024)
    def generate_sizes(width, height, min_sizes_size=200):
        """
        Produces the sizes array for the given width and height combination. Function results are
        cached.

        :param width: the width of the source image
        :param height: the height of the source image
        :param min_sizes_size: the minimum dimension size to include in the returned list
        :return: a list of sizes in descending order
        """
        # always include the original image size in the sizes list
        sizes = [{'width': width, 'height': height}]
        for i in count(1):
            factor = 2 ** i
            new_width = width // factor
            new_height = height // factor
            # stop when either dimension is smaller than
            if new_width < min_sizes_size or new_height < min_sizes_size:
                break
            sizes.append({'width': new_width, 'height': new_height})

        return sizes

    async def get(self, identifier):
        """
        Responds to IIIF info.json get requests.

        :param identifier: the IIIF image identifier
        """
        image = IIIFImage(identifier, self.config['source_path'], self.config['cache_path'])

        # if the image's info.json is not in the cache, we need to generate it and cache it
        if image.name not in self.info_cache:
            # we need the image source file to exist on disk so that we can work on it
            await self.image_source_fetcher.ensure_source_exists(image)

            width, height = await self.image_source_sizer.get_image_size(image)

            # add the complete info.json to the cache
            self.info_cache[image.name] = {
                '@context': 'http://iiif.io/api/image/3/context.json',
                'id': f'{self.config["base_url"]}/{image.identifier}',
                # mirador/openseadragon seems to need this to work even though I don't think
                # it's correct under the IIIF image API v3
                '@id': f'{self.config["base_url"]}/{image.identifier}',
                'type': 'ImageService3',
                'protocol': 'http://iiif.io/api/image',
                'width': width,
                'height': height,
                'rights': 'http://creativecommons.org/licenses/by/4.0/',
                'profile': 'level0',
                'tiles': [
                    {'width': 512, 'scaleFactors': [1, 2, 4, 8, 16]},
                    {'width': 256, 'scaleFactors': [1, 2, 4, 8, 16]},
                    {'width': 1024, 'scaleFactors': [1, 2, 4, 8, 16]},
                ],
                'sizes': self.generate_sizes(width, height, self.config['min_sizes_size']),
                # suggest to clients that upscaling isn't supported
                'maxWidth': width,
                'maxHeight': height,
            }

        # serve up the info.json (tornado automatically writes a dict out as JSON with headers etc)
        await self.finish(self.info_cache[image.name])


def main():
    """
    Main entry function for the server.
    """
    # disable DecompressionBombErrors
    # (https://pillow.readthedocs.io/en/latest/releasenotes/5.0.0.html#decompression-bombs-now-raise-exceptions)
    Image.MAX_IMAGE_PIXELS = None

    # load the config file, it should be next to this script
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(root_dir, 'config.yml'), 'r') as cf:
        config = yaml.safe_load(cf)

    # create an LRU cache to keep the most recent info.json request responses in
    info_cache = LRU(config['info_cache_size'])

    # create the dispatcher which controls how image data requests are handled
    dispatcher = ImageProcessingDispatcher()

    # create the image source managers which control image source retrieval and sizing
    image_source_fetcher = ImageSourceFetcher(config)
    image_source_sizer = ImageSourceSizer(config)

    try:
        # initialise the process pool that backs the dispatcher
        dispatcher.init_workers(config['image_pool_size'], config['image_cache_size_per_process'])

        # setup the tornado app
        app = Application([
            (r'/(?P<identifier>.+)/info.json', ImageInfoHandler,
             dict(config=config, info_cache=info_cache, image_source_fetcher=image_source_fetcher,
                  image_source_sizer=image_source_sizer)),
            (r'/(?P<identifier>.+)/(?P<region>.+)/(?P<size>.+)/0/default.jpg', ImageDataHandler,
             dict(config=config, dispatcher=dispatcher, image_source_fetcher=image_source_fetcher)),
        ])
        app.listen(config['http_port'])
        print(f'Listening on {config["http_port"]}, our pid is {os.getpid()}')
        IOLoop.current().start()
    except KeyboardInterrupt:
        print('Shutdown request received')
    finally:
        image_source_sizer.stop()
        dispatcher.stop()
        print('Shutdown complete')


if __name__ == '__main__':
    main()
