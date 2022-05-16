from fastapi import APIRouter
from pathlib import Path
from starlette.responses import StreamingResponse
from typing import Tuple, Iterable
from zipstream import AioZipStream

from iiif.exceptions import TooManyImages
from iiif.profiles import AbstractProfile
from iiif.state import state
from iiif.utils import parse_identifier

router = APIRouter()


def parse_identifiers(identifiers: str) -> Iterable[Tuple[AbstractProfile, str]]:
    """
    Given a comma separated list of identifiers, split them up and yield unique (profile, name)
    pairs. Each identifier in the string can be either a full identifier <profile>:<name> or one
    using the default profile <name>. Only unique identifiers are returned to avoid downloading the
    same file multiple times for no reason.

    :param identifiers: the identifiers, comma separated
    :return: iterable of 2-tuples containing the AbstractProfile and the name
    """
    seen = set()
    for identifier in identifiers.split(','):
        profile_name, name = parse_identifier(identifier)
        profile_and_name = (state.get_profile(profile_name), name)
        if profile_and_name not in seen:
            seen.add(profile_and_name)
            yield profile_and_name


@router.get('/originals')
async def zip_originals(identifiers: str, use_original_filenames: bool = True) -> StreamingResponse:
    """
    Endpoint which streams a zip containing the original versions of the requested images.
    The zip is created on the fly which in theory means it could be unlimited in size, however, to
    try and keep things under control and because of implementation shortcomings, a limit is set in
    the config.

    Any requested identifiers that can't be found are simply not included in the downloaded zip.
    \f

    :param identifiers: a comma separated list of identifiers (<profile name>:<name>) and/or just
                        names in which case the default profile is used
    :param use_original_filenames: whether to use the original file names in the zip (True, the
                                   default) or name the files after the image name
    :return: a StreamingResponse object streaming a dynamically generated zip of the requested
             original files
    """
    chunk_size = state.config.download_chunk_size
    profiles_and_names = list(parse_identifiers(identifiers))
    if len(profiles_and_names) > state.config.download_max_files:
        raise TooManyImages(state.config.download_max_files)

    # aiozipstream can't handle async generators which is a shame :(
    files = []
    for profile, name in profiles_and_names:
        filename = await profile.resolve_filename(name)
        if filename is not None:
            if not use_original_filenames:
                filename = f'{name}{Path(filename).suffix.lower()}'
            files.append({
                'name': filename,
                'stream': profile.stream_original(name, chunk_size=chunk_size),
                'compression': 'deflate'
            })

    zip_stream = AioZipStream(files, chunksize=state.config.download_chunk_size)
    response = StreamingResponse(
        zip_stream.stream(), media_type='application/zip',
        headers={'Content-Disposition': 'attachment; filename=originals.zip'}
    )
    return response
