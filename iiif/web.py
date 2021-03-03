#!/usr/bin/env python3
# encoding: utf-8

import yaml
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from itertools import count
from lru import LRU
from pathlib import Path
from starlette.responses import FileResponse

from iiif.image import ImageSourceFetcher, ImageSourceSizer, IIIFImage
from iiif.processing import ImageProcessingDispatcher, Task

# disable DecompressionBombErrors
# (https://pillow.readthedocs.io/en/latest/releasenotes/5.0.0.html#decompression-bombs-now-raise-exceptions)
Image.MAX_IMAGE_PIXELS = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'OPTIONS'],
    allow_headers=['*'],
)


@app.on_event('startup')
def on_startup():
    # load the config file, it should be in the folder above this script's location
    config_path = Path(__file__).parent.parent / 'config.yml'
    with config_path.open('rb') as cf:
        config = yaml.safe_load(cf)

    # convert the path config values into Path objects
    config['source_path'] = Path(config['source_path'])
    config['cache_path'] = Path(config['cache_path'])

    # create the dispatcher which controls how image data requests are handled
    dispatcher = ImageProcessingDispatcher()
    image_source_sizer = ImageSourceSizer(config)

    # create an LRU cache to keep the most recent info.json request responses in
    info_cache = LRU(config['info_cache_size'])

    # create the image source managers which control image source retrieval and sizing
    image_source_fetcher = ImageSourceFetcher(config)

    # initialise the process pool that backs the dispatcher
    dispatcher.init_workers(config['image_pool_size'], config['image_cache_size_per_process'])

    app.state.config = config
    app.state.dispatcher = dispatcher
    app.state.image_source_sizer = image_source_sizer
    app.state.info_cache = info_cache
    app.state.image_source_fetcher = image_source_fetcher


@app.on_event('shutdown')
async def on_shutdown():
    await app.state.image_source_fetcher.stop()
    app.state.image_source_sizer.stop()
    app.state.dispatcher.stop()


@lru_cache(maxsize=1024)
def generate_sizes(width: int, height: int, min_sizes_size: int = 200):
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


@app.get("/status")
def status():
    return {'status': True}


@app.get("/{identifier:path}/info.json")
async def get_image_info(identifier: str):
    image = IIIFImage(identifier, app.state.config['source_path'], app.state.config['cache_path'])

    # if the image's info.json is not in the cache, we need to generate it and cache it
    if image.name not in app.state.info_cache:
        # we need the image source file to exist on disk so that we can work on it
        await app.state.image_source_fetcher.ensure_source_exists(image)

        width, height = await app.state.image_source_sizer.get_image_size(image)

        # add the complete info.json to the cache
        app.state.info_cache[image.name] = {
            '@context': 'http://iiif.io/api/image/3/context.json',
            'id': f'{app.state.config["base_url"]}/{image.identifier}',
            # mirador/openseadragon seems to need this to work even though I don't think
            # it's correct under the IIIF image API v3
            '@id': f'{app.state.config["base_url"]}/{image.identifier}',
            'type': 'ImageService3',
            'protocol': 'http://iiif.io/api/image',
            'width': width,
            'height': height,
            'rights': 'http://creativecommons.org/licenses/by/4.0/',
            'profile': 'level1',
            'tiles': [
                {'width': 512, 'scaleFactors': [1, 2, 4, 8, 16]},
                {'width': 256, 'scaleFactors': [1, 2, 4, 8, 16]},
                {'width': 1024, 'scaleFactors': [1, 2, 4, 8, 16]},
            ],
            'sizes': generate_sizes(width, height, app.state.config['min_sizes_size']),
            # suggest to clients that upscaling isn't supported
            'maxWidth': width,
            'maxHeight': height,
        }

    # serve up the info.json (tornado automatically writes a dict out as JSON with headers etc)
    return app.state.info_cache[image.name]


@app.get("/{identifier:path}/{region}/{size}/0/default.jpg")
async def get_image_data(identifier: str, region: str, size: str):
    image = IIIFImage(identifier, app.state.config['source_path'], app.state.config['cache_path'])
    # we need the image source file to exist on disk so that we can work on it
    await app.state.image_source_fetcher.ensure_source_exists(image)

    task = Task(image, region, size)
    # submit the task to the dispatcher and wait for it to complete
    await app.state.dispatcher.submit(task)

    return FileResponse(str(task.output_path), media_type='image/jpeg')
