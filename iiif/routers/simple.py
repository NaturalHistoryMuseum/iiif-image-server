from fastapi import APIRouter
from starlette.responses import FileResponse, StreamingResponse

from iiif.exceptions import image_not_found
from iiif.routers.iiif import get_image_data
from iiif.state import state
from iiif.utils import parse_identifier, get_mimetype

router = APIRouter()
default_iiif_params = dict(rotation='0', quality='default', fmt='jpg')


@router.get('/{identifier}')
async def image(identifier: str) -> FileResponse:
    """
    Simple endpoint for an image which returns the preview size version of it.
    \f

    :param identifier: the image identifier
    :return: a FileResponse object streaming a jpeg image
    """
    return await preview(identifier)


@router.get('/{identifier}/thumbnail')
async def thumbnail(identifier: str) -> FileResponse:
    """
    Endpoint which returns a thumbnail version of the requested image. If the full image is smaller
    than the configured thumbnail width then a full size image is returned. The returned file from
    this endpoint is always a jpeg.
    \f

    :param identifier: the image identifier
    :return: a FileResponse object streaming a jpeg image
    """
    profile, info = await state.get_profile_and_info(identifier)
    target_width = min(info.width, state.config.thumbnail_width)
    return await get_image_data(identifier=info.identifier, region='full', size=f'{target_width},',
                                **default_iiif_params)


@router.get('/{identifier}/preview')
async def preview(identifier: str) -> FileResponse:
    """
    Endpoint which returns a preview version of the requested image. If the full image is smaller
    than the configured preview width then a full size image is returned. The returned file from
    this endpoint is always a jpeg.
    \f

    :param identifier: the image identifier
    :return: a FileResponse object streaming a jpeg image
    """
    profile, info = await state.get_profile_and_info(identifier)
    target_width = min(info.width, state.config.preview_width)
    return await get_image_data(identifier=info.identifier, region='full', size=f'{target_width},',
                                **default_iiif_params)


@router.get('/{identifier}/original')
async def original(identifier: str) -> StreamingResponse:
    """
    Endpoint which returns the original version of the requested image. This image won't necessarily
    be a jpeg (e.g. it could be a tiff) as it is not processed by this server, we merely stream the
    image straight from the storage location to the requester.
    \f

    :param identifier: the image identifier
    :return: a StreamingResponse object streaming the original image
    """
    profile_name, name = parse_identifier(identifier)
    profile = state.get_profile(profile_name)
    filename = await profile.resolve_filename(name)
    if filename is None:
        raise image_not_found()
    response = StreamingResponse(
        profile.stream_original(name, chunk_size=state.config.download_chunk_size),
        media_type=get_mimetype(filename),
        # note the quoted file name, this avoids client-side errors if the filename contains a comma
        headers={'content-disposition': f'attachment; filename="{filename}"'}
    )
    return response
