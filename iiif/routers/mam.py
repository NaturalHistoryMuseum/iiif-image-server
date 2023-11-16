from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import RedirectResponse

from iiif.profiles.mss import MSSProfile
from iiif.state import state

router = APIRouter()


@router.get('/mam/{asset_id}')
async def mam_redirect(request: Request, asset_id: str) -> RedirectResponse:
    """
    When an old MAM URL is requested, it is now redirected to this endpoint. This endpoint looks up
    the old asset ID and then redirects base simple image endpoint using the GUID instead of the
    asset ID.
    If the MSS is the default profile then the mss: is omitted, if not then it is included.
    \f

    :param request: the request object
    :param asset_id: the MAM asset ID
    :return: a RedirectResponse to the MSS preview endpoint
    """
    mss_profile: MSSProfile = state.get_profile('mss')
    # convert the asset ID into a GUID
    guid = await mss_profile.convert_guid_to_asset_id(asset_id)

    if state.config.default_profile_name == 'mss':
        # if the default profile is the mss profile, redirect to just guid for nice clean URLs
        identifier = guid
    else:
        # otherwise, create the full identifier with profile name
        identifier = f'mss:{guid}'
    # this seems to be the easiest way to ensure we redirect to a sensible path given we may be
    # under some custom subpath via a proxy
    path = request.url.path.replace(f'/mam/{asset_id}', f'/{identifier}')
    return RedirectResponse(path)
