#!/usr/bin/env python3
# encoding: utf-8

import aiohttp
import platform
import time
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse

from iiif.exceptions import handler, IIIFServerException
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


@app.middleware('http')
async def add_debug_headers(request: Request, call_next):
    """
    Middleware which adds some debug headers :)

    :param request: the request
    :param call_next: the next function in the chain
    :return: the response
    """
    start_time = time.monotonic()
    response = await call_next(request)
    # add the process time
    response.headers['x-process-time'] = str(time.monotonic() - start_time)
    # add which server processed the request
    response.headers['x-served-by'] = platform.node()
    return response


@app.on_event('shutdown')
async def on_shutdown():
    """
    This is run on shutdown and makes sure that all of the objects we added to the app state are
    closed properly.
    """
    for profile in state.profiles.values():
        await profile.close()
    state.processor.stop()


@app.get('/status')
async def status(full: bool = False) -> JSONResponse:
    """
    Returns the status of the server along with some stats about current resource usages.
    \f

    :param full: boolean parameter indicating whether to provide a full status or just the
                 essentials. The full status may take longer to generate. Default: False.
    :return: a dict
    """
    body = {
        'status': ':)',
        'default_profile': state.config.default_profile_name,
        'processing': state.processor.get_status(),
        'profiles': {
            profile.name: await profile.get_status(full)
            for profile in state.profiles.values()
        }
    }
    return JSONResponse(body, headers={'cache-control': 'no-store'})


@app.get('/favicon.ico')
async def favicon() -> StreamingResponse:
    async def get():
        async with aiohttp.ClientSession() as session:
            async with session.get('https://iiif.io/favicon.ico') as response:
                while chunk := await response.content.read(4096):
                    yield chunk

    return StreamingResponse(get())


app.add_exception_handler(IIIFServerException, handler)


# order matters here btw!
app.include_router(originals.router)
app.include_router(simple.router)
app.include_router(iiif.router)
