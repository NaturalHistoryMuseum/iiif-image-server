#!/usr/bin/env python3
# encoding: utf-8

from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from iiif.routers import iiif, originals, simple
from iiif.state import state

# disable DecompressionBombErrors
# (https://pillow.readthedocs.io/en/latest/releasenotes/5.0.0.html#decompression-bombs-now-raise-exceptions)
Image.MAX_IMAGE_PIXELS = None

app = FastAPI(title='Data Portal Image Service')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'OPTIONS'],
    allow_headers=['*'],
)


@app.on_event('shutdown')
async def on_shutdown():
    """
    This is run on shutdown and makes sure that all of the objects we added to the app state are
    closed properly.
    """
    for profile in state.profiles.values():
        await profile.close()
    state.dispatcher.stop()


@app.get('/status')
async def status() -> dict:
    """
    Returns the status of the server along with some stats about current resource usages.

    :return: a dict
    """
    return {
        'status': ':)',
        'default_profile': state.config.default_profile_name,
        'processing': state.dispatcher.get_status(),
        'profiles': {
            profile.name: await profile.get_status()
            for profile in state.profiles.values()
        }
    }


# order matters here btw!
app.include_router(originals.router)
app.include_router(simple.router)
app.include_router(iiif.router)
