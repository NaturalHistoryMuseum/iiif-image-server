#!/usr/bin/env python3
# encoding: utf-8

from PIL import Image
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from lru import LRU
from pathlib import Path
from starlette.responses import FileResponse, StreamingResponse
from typing import Tuple
from zipstream import AioZipStream

from iiif.config import load_config
from iiif.exceptions import image_not_found, profile_not_found, too_many_images
from iiif.processing import Task, ImageProcessingDispatcher
from iiif.profiles import ImageInfo, AbstractProfile, MSSProfile
from iiif.utils import parse_size, generate_sizes

# disable DecompressionBombErrors
# (https://pillow.readthedocs.io/en/latest/releasenotes/5.0.0.html#decompression-bombs-now-raise-exceptions)
Image.MAX_IMAGE_PIXELS = None

# this server currently supports IIIF level1
IIIF_LEVEL = 'level1'

app = FastAPI(title='Data Portal Image Service')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'OPTIONS'],
    allow_headers=['*'],
)


@app.middleware("http")
async def add_compliance_level_header(request: Request, call_next) -> Response:
    """
    Adds the IIIF compliance level this server supports as a Link header to all responses. See
    https://iiif.io/api/image/3.0/#6-compliance-level-and-profile-document.

    :param request: the request object
    :param call_next: the next function in the chain
    :return: the response
    """
    response = await call_next(request)
    response.headers['Link'] = f'<http://iiif.io/api/image/3/{IIIF_LEVEL}.json>;rel="profile"'
    return response


@app.on_event('startup')
def on_startup():
    """
    This is run on startup and just sets a series of objects on the app state so that they can be
    accessed during requests.
    """
    config = load_config()

    # load the profiles and grab the MSS profile (there should really only be one of these)
    mss = next((p for p in config.profiles.values() if isinstance(p, MSSProfile)), None)

    # create the dispatcher which controls how image data requests are handled
    dispatcher = ImageProcessingDispatcher()
    dispatcher.init_workers(config.image_pool_size, config.image_cache_size_per_process)

    # create an LRU cache to keep the most recent info.json request responses in
    info_cache = LRU(config.info_cache_size)

    # add all these objects to the app state so that they can be accessed during requests
    app.state.config = config
    app.state.dispatcher = dispatcher
    app.state.info_cache = info_cache
    app.state.mss = mss


@app.on_event('shutdown')
async def on_shutdown():
    """
    This is run on shutdown and makes sure that all of the objects we added to the app state are
    closed properly.
    """
    for profile in app.state.config.profiles.values():
        await profile.close()
    app.state.dispatcher.stop()


@app.get('/status')
async def status() -> dict:
    """
    Returns the status of the server along with some stats about current resource usages.

    :return: a dict
    """
    return {
        'status': ':)',
        'processing': app.state.dispatcher.get_status(),
        'info_cache_size': len(app.state.info_cache),
        'profiles': {
            profile.name: await profile.get_status()
            for profile in app.state.config.profiles.values()
        }
    }


@app.get('/{name}/thumbnail')
async def thumbnail(name: str) -> FileResponse:
    """
    MSS specific endpoint which returns a thumbnail version of the requested image. If the full
    image is smaller than the configured thumbnail width then a full size image is returned. The
    returned file from this endpoint is always a jpeg.

    :param name: the image name (currently the EMu IRN of the multimedia object)
    :return: a FileResponse object streaming a jpeg image
    """
    info = await get_info(app.state.mss, name)
    target_width = min(info.width, app.state.config.thumbnail_width)
    return await get_image_data(app.state.mss.name, name, 'full', f'{target_width},')


@app.get('/{name}/preview')
async def preview(name: str) -> FileResponse:
    """
    MSS specific endpoint which returns a preview version of the requested image. If the full image
    is smaller than the configured preview width then a full size image is returned. The returned
    file from this endpoint is always a jpeg.

    :param name: the image name (currently the EMu IRN of the multimedia object)
    :return: a FileResponse object streaming a jpeg image
    """
    info = await get_info(app.state.mss, name)
    target_width = min(info.width, app.state.config.preview_width)
    return await get_image_data(app.state.mss.name, name, 'full', f'{target_width},')


@app.get('/{name}/original')
async def original(name: str) -> StreamingResponse:
    """
    MSS specific endpoint which returns the original version of the requested image. This image
    won't necessarily be a jpeg (it's likely to be a tiff) as it is not processed by this server, we
    merely stream the image straight from the storage location to the requester.

    :param name: the image name (currently the EMu IRN of the multimedia object)
    :return: a StreamingResponse object streaming the original image
    """
    profile: MSSProfile = app.state.mss
    doc = await profile.get_mss_doc(name)
    if doc is None:
        raise image_not_found()

    response = StreamingResponse(
        profile.stream_original(name, chunk_size=app.state.config.download_chunk_size),
        # note the quoted file name, this avoids client-side errors if the filename contains a comma
        headers={'Content-Disposition': f'attachment; filename="{doc["file"]}"'}
    )
    return response


@app.get('/originals')
async def zip_originals(names: str, stop_on_error: bool = True,
                        use_original_filenames: bool = True) -> StreamingResponse:
    """
    MSS specific endpoint which streams a zip containing the original versions of the requested
    images. The zip is created on the fly which in theory means it could be unlimited in size,
    however, to try and keep things under control a limit set in the config.

    Any requested names that can't be found are simply not included in the downloaded zip.

    :param names: a comma separated list of EMu IRNs
    :param stop_on_error: whether to stop streaming and return an error if there is a problem while
                          streaming a file (True, the default) or finish the file and continue with
                          the next file (if the file hasn't started streaming yet then this will
                          result in an empty file, if the file has started streaming this will
                          result in a partial file)
    :param use_original_filenames: whether to use the original file names in the zip (True, the
                                   default) or name the files after the name (i.e. the EMu IRN)
    :return: a StreamingResponse object streaming a dynamically generated zip of the requested
             original files
    """
    profile: MSSProfile = app.state.mss
    chunk_size: int = app.state.config.download_chunk_size
    max_files: int = app.state.config.download_max_files

    names = [name.strip() for name in names.split(',')]
    if len(names) > max_files:
        raise too_many_images(max_files)

    # aiozipstream can't handle async generators which is a shame :(
    files = []
    for name in names:
        doc = await profile.get_mss_doc(name)
        if doc is not None:
            if use_original_filenames:
                filename = doc['file']
            else:
                filename = f'{name}{Path(doc["file"]).suffix.lower()}'
            files.append({
                'name': filename,
                'stream': profile.stream_original(name, chunk_size=chunk_size,
                                                  raise_errors=stop_on_error),
                'compression': 'deflate'
            })

    zip_stream = AioZipStream(files, chunksize=chunk_size)
    response = StreamingResponse(
        zip_stream.stream(), media_type='application/zip',
        headers={'Content-Disposition': 'attachment; filename=originals.zip'}
    )
    return response


@app.get('/{profile_name}:{name}/info.json')
async def get_image_info(profile_name: str, name: str) -> dict:
    """
    IIIF image info endpoint compliant with the specification:
    https://iiif.io/api/image/3.0/#22-image-information-request-uri-syntax.

    The profile and the name must match valid configurations from the config file to work. Together
    they must be unique.

    :param profile_name: the name of the profile to use
    :param name: the name of the image, this should be unique within the profile namespace
    :return: the info.json as a dict
    """
    profile = get_profile(profile_name)
    key = f'{profile_name}:{name}'

    # if the image's info.json is not in the cache, we need to generate it and cache it
    if key not in app.state.info_cache:
        info = await get_info(profile, name)

        id_url = f'{app.state.config.base_url}/{info.identifier}'
        # add the complete info.json to the cache
        app.state.info_cache[key] = {
            '@context': 'http://iiif.io/api/image/3/context.json',
            'id': id_url,
            # mirador/openseadragon seems to need this to work even though I don't think it's
            # correct under the IIIF image API v3
            '@id': id_url,
            'type': 'ImageService3',
            'protocol': 'http://iiif.io/api/image',
            'width': info.width,
            'height': info.height,
            'rights': profile.rights,
            'profile': IIIF_LEVEL,
            'tiles': [
                {'width': 512, 'scaleFactors': [1, 2, 4, 8, 16]},
                {'width': 256, 'scaleFactors': [1, 2, 4, 8, 16]},
                {'width': 1024, 'scaleFactors': [1, 2, 4, 8, 16]},
            ],
            'sizes': generate_sizes(info.width, info.height, app.state.config.min_sizes_size),
            # suggest to clients that upscaling isn't supported
            'maxWidth': info.width,
            'maxHeight': info.height,
        }

    # serve up the info.json (fastapi automatically writes a dict out as JSON with headers etc)
    return app.state.info_cache[key]


@app.get('/{profile_name}:{name}/{region}/{size}/0/default.jpg')
async def get_image_data(profile_name: str, name: str, region: str, size: str) -> FileResponse:
    """
    IIIF image info endpoint compliant with the specification:
    https://iiif.io/api/image/3.0/#21-image-request-uri-syntax.

    The profile and the name must match valid configurations from the config file to work. Together
    they must be unique.

    :param profile_name: the name of the profile to use
    :param name: the name of the image, this should be unique within the profile namespace
    :param region: the rectangular portion of the underlying image content to be returned
    :param size: the dimensions to which the extracted region, which might be the full image, is to
                 be scaled
    :return: a FileResponse object streaming the image data in jpeg format
    """
    profile, info = await get_profile_and_info(profile_name, name)

    # the logic here allows profiles to try and find the smallest image derivative available that
    # can do the job as this will produce the fastest response time (smaller images are faster to
    # retrieve (if they need to be retrieved) and process!). MSS is the only profile that currently
    # does this. Note that we can only do this optimisation if the region is the entire image.
    # TODO: we could also check the user hasn't requested the exact full width and height
    target = parse_size(size, info.width, info.height) if region == 'full' else None

    # ensure a source file is available to process
    source_path = await profile.fetch_source(info, target)

    output_path = app.state.config.cache_path / profile_name / name / region / f'{size}.jpg'
    if not output_path.exists():
        task = Task(source_path, output_path, region, size)
        # submit the task to the dispatcher and wait for it to complete
        await app.state.dispatcher.submit(task)

    return FileResponse(output_path, media_type='image/jpeg')


def get_profile(profile_name: str) -> AbstractProfile:
    """
    Helper function that gets the AbstractProfile object associated with the given name and returns
    it. If one cannot be found then an error is raised.

    :param profile_name: the profile name
    :return: the profile object (this will be a subclass of the AbstractProfile abstract class)
    """
    profile = app.state.config.profiles.get(profile_name, None)
    if profile is None:
        raise profile_not_found()
    return profile


async def get_info(profile: AbstractProfile, name: str) -> ImageInfo:
    """
    Helper function that gets the ImageInfo from the profile and returns it. If the info isn't
    available then an error is raised.

    :param profile: the profile object (this must be a subclass of the AbstractProfile abstract
                    class)
    :param name: the image name
    :return: the info object (this will be a subclass of the ImageInfo abstract class)
    """
    info = await profile.get_info(name)
    if info is None:
        raise image_not_found()
    return info


async def get_profile_and_info(profile_name: str, name: str) -> Tuple[AbstractProfile, ImageInfo]:
    """
    Helper function that gets the profile and the info at the same time.

    :param profile_name: the name of the profile
    :param name: the name of the image
    :return: a 2-tuple containing the profile object and the info object
    """
    profile = get_profile(profile_name)
    return profile, await get_info(profile, name)
