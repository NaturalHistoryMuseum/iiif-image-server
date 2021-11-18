from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from starlette.responses import FileResponse

from iiif.ops import IIIF_LEVEL, parse_params
from iiif.processing import Task
from iiif.state import state
from iiif.utils import get_mimetype

router = APIRouter()


@router.get('/{identifier}/info.json')
async def get_image_info(identifier: str) -> ORJSONResponse:
    """
    IIIF image info endpoint compliant with the specification:
    https://iiif.io/api/image/3.0/#22-image-information-request-uri-syntax.

    The profile and the name must match valid configurations from the config file to work. Together
    they must be unique.
    \f

    :param identifier: the image identifier
    :return: the info.json as a dict
    """
    profile, info = await state.get_profile_and_info(identifier)
    info_json = await profile.generate_info_json(info, IIIF_LEVEL)
    # add a cache-control header and iiif header
    headers = {
        'cache-control': f'max-age={profile.cache_for}',
        'link': f'<http://iiif.io/api/image/3/level{IIIF_LEVEL}.json>;rel="profile"'
    }
    return ORJSONResponse(content=info_json, headers=headers)


@router.get('/{identifier}/{region}/{size}/{rotation}/{quality}.{fmt}')
async def get_image_data(identifier: str, region: str, size: str, rotation: str, quality: str,
                         fmt: str) -> FileResponse:
    """
    IIIF image info endpoint compliant with the specification:
    https://iiif.io/api/image/3.0/#21-image-request-uri-syntax.

    The profile and the name must match valid configurations from the config file to work. Together
    they must be unique.
    \f

    :param identifier: the image identifier
    :param region: the rectangular portion of the underlying image content to be returned
    :param size: the dimensions to which the extracted region, which might be the full image, is to
                 be scaled
    :param rotation: the rotation to apply to the image
    :param quality: the quality of the image to return
    :param fmt: the format to return the image in
    :return: a FileResponse object streaming the image data in the requested format (currently
             always jpeg)
    """
    profile, info = await state.get_profile_and_info(identifier)
    # parse the IIIF parts of the request to assert parameter correctness and define what how we're
    # going to manipulate the image when we process it
    ops = parse_params(info, region, size, rotation, quality, fmt)
    output_path = state.config.cache_path / info.profile_name / info.name / ops.location

    # only do work if there is no cached version of the requested file
    if not output_path.exists():
        # this is a performance enhancement. By providing a hint at the size of the image we need to
        # serve up, we can (sometimes!) use a smaller source image thus reducing processing time
        if ops.region.full:
            # the full image region is selected so we can take the hint from the size parameter
            size_hint = (ops.size.w, ops.size.h)
        else:
            # a region has been specified, we'll have to use the whole thing
            # TODO: can we do better than this?
            size_hint = None

        # ensure a source file is available to process, passing the hint
        source_path = await profile.fetch_source(info, size_hint)

        task = Task(source_path, output_path, ops)
        # submit the task to the dispatcher and wait for it to complete
        await state.dispatcher.submit(task)

    # add a cache-control header and iiif header
    headers = {
        'cache-control': f'max-age={profile.cache_for}',
        'link': f'<http://iiif.io/api/image/3/level{IIIF_LEVEL}.json>;rel="profile"'
    }
    return FileResponse(output_path, media_type=get_mimetype(output_path), headers=headers)
