from collections import Counter
from concurrent.futures import Executor

import asyncio
import json
import logging
import shutil
import tempfile
import time
from aiohttp import ClientResponse, ClientError, ClientTimeout
from cachetools import TTLCache
from contextlib import asynccontextmanager
from dataclasses import dataclass
from elasticsearch_dsl import Search
from functools import partial
from itertools import cycle
from pathlib import Path
from typing import List, Union
from typing import Tuple, Optional
from urllib.parse import quote

from iiif.config import Config
from iiif.exceptions import Timeout, IIIFServerException, ImageNotFound, log_error
from iiif.profiles.base import AbstractProfile, ImageInfo
from iiif.utils import Locker, convert_image, create_client_session, FetchCache, Fetchable
from iiif.utils import get_size


class MSSAccessDenied(ImageNotFound):

    def __init__(self, profile: str, name: str, emu_irn: int):
        super().__init__(profile, name,
                         log=f"MSS denied access to multimedia IRN {emu_irn} [guid: {name}]")
        self.emu_irn = emu_irn


class MSSDocDuplicates(ImageNotFound):

    def __init__(self, profile: str, name: str, total: int):
        super().__init__(profile, name, log=f"Found {total} MSS docs for the guid {name}")
        self.total = total


class MSSDocNotFound(ImageNotFound):

    def __init__(self, profile: str, name: str):
        super().__init__(profile, name, log=f"No MSS doc found for the guid {name}")


class MSSStoreFailure(IIIFServerException):

    def __init__(self, profile: str, name: str, cause: 'StoreStreamError'):
        super().__init__(f'Failed to retrieve the requested image data for {name}', status_code=503,
                         log=f'Failed to stream the source file for {profile}:{name} from '
                             f'{cause.url} due to {cause.cause}')


class AssetIDNotFound(IIIFServerException):

    def __init__(self, asset_id: str):
        super().__init__(f"Asset ID {asset_id} not found", status_code=404)


class AssetIDDuplicateGUIDs(IIIFServerException):

    def __init__(self, asset_id: str, total: int):
        super().__init__(f"Asset ID {asset_id} matched multiple images", status_code=404,
                         log=f"Asset ID {asset_id} matched multiple {total} GUIDs")
        self.total = total


class MSSStoreNoLength(IIIFServerException):

    def __init__(self, profile: str, name: str):
        super().__init__(f'Failed to get data length for {name} from the {profile} backend.')


class MSSConversionFailure(IIIFServerException):

    def __init__(self, source: 'MSSSourceFile', cause: Exception):
        super().__init__(f'Failed to convert source image',
                         log=f'Failed to convert {source.file} ({source.emu_irn}) due to {cause}')


class MSSImageInfo(ImageInfo):
    """
    MSS variant of the ImageInfo class.
    """

    def __init__(self, profile_name: str, name: str, doc: dict):
        """
        :param profile_name: the name of the profile
        :param name: the name of the image, this will be a GUID
        :param doc: the image's doc from the elasticsearch MSS index
        """
        super().__init__(profile_name, name, doc.get('width', None), doc.get('height', None))
        self.doc = doc
        self.emu_irn = doc['id']
        # the name of the original file as it appears on the actual filesystem EMu is using
        self.original = doc['file']
        # the old MAM asset ID value, if there is one
        self.old_asset_id = doc.get('old_asset_id')
        # a list of the EMu generated derivatives of the original file. The list should already be
        # in the ascending (width, height) order because the import process sorts it
        self.derivatives = doc.get('derivatives', [])

    def choose_file(self, target_size: Optional[Tuple[int, int]] = None) -> str:
        """
        Given a target size, retrieve the smallest file which contains the target size. If no target
        size is provided or there are no derivatives available, then the original image file is
        returned.

        :param target_size: the target size as a 2-tuple of ints, or None
        :return: the name of the chosen file
        """
        if target_size is not None and self.derivatives:
            target_width, target_height = target_size
            for derivative in self.derivatives:
                if target_width <= derivative['width'] and target_height <= derivative['height']:
                    return derivative['file']
        return self.original


class MSSProfile(AbstractProfile):

    def __init__(self,
                 name: str,
                 config: Config,
                 pool: Executor,
                 rights: str,
                 es_hosts: List[str],
                 mss_url: str,
                 mss_ssl: bool = True,
                 mss_index: str = 'data-mss-latest',
                 mss_limit: int = 10,
                 es_limit: int = 10,
                 info_cache_size: int = 100_000,
                 info_cache_ttl: float = 43_200,
                 info_lock_ttl: float = 60,
                 source_cache_size: int = 1024 * 1024 * 256,
                 source_cache_ttl: float = 12 * 60 * 60,
                 convert_quality: int = 85,
                 convert_subsampling: str = '4:2:0',
                 dams_limit: int = 4,
                 dams_ssl: bool = True,
                 **kwargs
                 ):
        """
        :param name: the name of this profile
        :param config: the Config object
        :param pool: the general purpose pool for offloading processing if necessary
        :param rights: the rights url for images served by this profile
        :param es_hosts: a list of elasticsearch hosts to use
        :param mss_url: mss base url
        :param mss_ssl: boolean indicating whether ssl certificates should be checked when making
                        requests to mss
        :param mss_index: the MSS index name in elasticsearch
        :param mss_limit: the maximum number of simultaneous connections that can be made to the MSS
        :param es_limit: the maximum number of simultaneous connections that can be made to
                         elasticsearch
        :param info_cache_size: max size of the caches related to the metadata of the image
        :param info_cache_ttl: ttl of the data in the caches related to the metadata of the image
        :param info_lock_ttl: how long to lock for when gathering the metadata about an image from
                              the various sources. Note that this needs to cover Elasticsearch and
                              MSS requests as well as potentially downloading the source image (in
                              order to find out the size)
        :param source_cache_size: max size in bytes of the source cache on disk
        :param source_cache_ttl: ttl of each source image in the cache
        :param convert_quality: quality to use when converting a source to a jpeg
        :param convert_subsampling: subsampling value to use when converting a source to a jpeg
        :param dams_limit: the maximum number of simultaneous connections that can be made to the
                           old dams service (to retrieve files stored at the damsurlfile value)
        :param dams_ssl: boolean indicating whether ssl certificates should be checked when making
                         requests to dams
        :param kwargs: extra kwargs for the AbstractProfile base class __init__
        """
        super().__init__(name, config, pool, rights, **kwargs)
        self.info_cache = TTLCache(info_cache_size, info_cache_ttl)
        self.get_info_locker = Locker(default_timeout=info_lock_ttl)

        self.get_mss_doc_locker = Locker(default_timeout=info_lock_ttl)
        self.mss_doc_cache = TTLCache(maxsize=info_cache_size, ttl=info_cache_ttl)

        self.es_handler = MSSElasticsearchHandler(es_hosts, es_limit, mss_index)
        self.store = MSSSourceStore(self.source_path, pool, mss_url, source_cache_size,
                                    source_cache_ttl, mss_limit, dams_limit, mss_ssl,
                                    dams_ssl, convert_quality, convert_subsampling)

    async def get_info(self, name: str) -> MSSImageInfo:
        """
        Given an image name (a GUID) returns a MSSImageInfo object or raise an exception if the
        image can't be found/isn't allowed to be accessed. If the image doesn't have width and
        height stored in the elasticsearch index for whatever reason then the original source image
        will be retrieved and the size extracted.

        :param name: the GUID of the image
        :return: an MSSImageInfo instance
        """
        if name in self.info_cache:
            return self.info_cache[name]

        try:
            async with self.get_info_locker.acquire(name):
                try:
                    # double check that the cache hasn't been filled while we waited for the lock
                    if name in self.info_cache:
                        return self.info_cache[name]

                    doc = await self.get_mss_doc(name)
                    info = MSSImageInfo(self.name, name, doc)
                    if info.width is None or info.height is None:
                        async with self.use_source(info) as source_path:
                            info.width, info.height = get_size(source_path)
                    self.info_cache[name] = info
                    return info
                except IIIFServerException:
                    raise
                except Exception as cause:
                    e = ImageNotFound(
                        self.name, f'An error occurred while processing an info request for {name}',
                        cause=cause, level=logging.ERROR
                    )
                    raise e from cause
        except asyncio.TimeoutError as cause:
            raise Timeout(cause=cause, log=f'Timeout while waiting for get_info lock on {name} in '
                                           f'profile {self.name}')

    @asynccontextmanager
    async def use_source(self, info: MSSImageInfo, size: Optional[Tuple[int, int]] = None) -> Path:
        """
        Given an MSSImageInfo object, retrieve a source image and yield the path where it is stored.
        This is an async context manager and the while the context exists the source image will
        remain on disk. If no target size is provided then the size of the image is used as the
        target size, this could result in either the original image being retrieved or a derivative
        of the same size (sometimes, it seems, EMu generates a jpeg derivative of the same size as
        the original). If the target size is provided then the smallest image available in MSS that
        can fulfill the request will be used.

        :param info: an MSSImageInfo instance
        :param size: a target size 2-tuple or None
        :return: the path to the source image on disk
        """
        # if the target size isn't provided, fill it in using the full size of the image. This
        # provides useful functionality as it allows choose_file to return a full size derivative
        # of the image instead of the original if one is available
        if size is None:
            size = info.size
        file = info.choose_file(size)
        source = MSSSourceFile(info.emu_irn, file, info.original == file)

        try:
            async with self.store.use(source) as path:
                yield path
        except StoreStreamError as cause:
            raise MSSStoreFailure(self.name, info.name, cause)

    async def get_mss_doc(self, name: str) -> dict:
        """
        Retrieves an MSS doc and ensures it's should be accessible. For a doc to be returned:

            - the GUID (i.e. the name) must be unique and exist in the mss elasticsearch index
            - the EMu IRN that the GUID maps to must be valid according the to the MSS (specifically
              the APS)

        :param name: the image name (a GUID)
        :return: the mss doc as a dict
        """
        if name in self.mss_doc_cache:
            return self.mss_doc_cache[name]

        try:
            async with self.get_mss_doc_locker.acquire(name):
                try:
                    # double check that the cache hasn't been filled while we waited for the lock
                    if name in self.mss_doc_cache:
                        return self.mss_doc_cache[name]

                    # first, get the document from the mss index
                    total, doc = await self.es_handler.get_mss_doc(name)
                    if total > 1:
                        raise MSSDocDuplicates(self.name, name, total)
                    elif total == 0:
                        raise MSSDocNotFound(self.name, name)

                    emu_irn = int(doc['id'])

                    # check with mss that the irn is valid
                    if not await self.store.check_access(emu_irn):
                        raise MSSAccessDenied(self.name, name, emu_irn)

                    self.mss_doc_cache[name] = doc
                    return doc
                except IIIFServerException:
                    raise
                except Exception as cause:
                    e = ImageNotFound(self.name, name, cause=cause, level=logging.ERROR)
                    raise e from cause
        except asyncio.TimeoutError as cause:
            raise Timeout(cause=cause, log=f'Timeout while waiting for get_mss_doc lock on {name} '
                                           f'in profile {self.name}')

    async def resolve_filename(self, name: str) -> str:
        """
        Given an image name (i.e. a GUID), return the original filename or None if the image can't
        be found.

        :param name: the image name (in this case the GUID)
        :return: the filename
        """
        doc = await self.get_mss_doc(name)
        return doc['file']

    async def resolve_original_size(self, name: str) -> int:
        """
        Given an image name (i.e. a GUID), return the size of the original image. This relies on the
        server which is actually serving up the original file data telling us how big the file is
        through a content-length header. This means we're using the actual filesize, not relying on
        the data EMu has to hand (which, if the file has been modified directly on disk, may not be
        correct and would cause issues if we presented an incorrect content-length).

        :param name: the image name (in this case the GUID)
        :return: the size of the original image in bytes
        """
        try:
            doc = await self.get_mss_doc(name)
            source = MSSSourceFile(int(doc['id']), doc['file'], True)
            return await self.store.get_file_size(source)
        except StoreStreamNoLength:
            raise MSSStoreNoLength(self.name, name)
        except StoreStreamError as cause:
            raise MSSStoreFailure(self.name, name, cause)

    async def stream_original(self, name: str, chunk_size: int = 4096):
        """
        Async generator which yields the bytes of the original image for the given image name (EMu
        IRN). If the image isn't available then nothing is yielded.

        :param name: the name of the image (EMu IRN)
        :param chunk_size: the size of the chunks to yield
        """
        try:
            doc = await self.get_mss_doc(name)
            source = MSSSourceFile(int(doc['id']), doc['file'], True, chunk_size)
            async for chunk in self.store.stream(source):
                yield chunk
        except StoreStreamError as cause:
            e = MSSStoreFailure(self.name, name, cause)
            # it's highly likely this error won't get surfaced through the exception handler because
            # the response has likely already begun, so log it before raising it to ensure we can
            # see what happened
            log_error(e)
            raise e

    async def convert_guid_to_asset_id(self, asset_id: str) -> str:
        """
        Given an old MAM asset ID, see if we can convert it into a GUID.

        :param asset_id: the old MAM asset ID
        :return: the matching GUID
        """
        total, guid = await self.es_handler.lookup_guid(asset_id)
        if total == 0:
            raise AssetIDNotFound(asset_id)
        elif total > 1:
            raise AssetIDDuplicateGUIDs(asset_id, total)
        return guid

    async def close(self):
        """
        Close down this profile.
        """
        await self.es_handler.close()
        await self.store.close()

    async def get_status(self) -> dict:
        status = await super().get_status()
        status['source_cache'] = await self.store.get_status()
        status['es'] = await self.es_handler.get_status()
        return status


def rebuild_data(parsed_data: dict) -> dict:
    """
    Rebuild the original data that Splitgill has encoded in Elasticsearch.

    :param parsed_data: the parsed dict
    :return: the rebuilt data dict
    """
    # this doesn't need _ checks because you can't currently have parsed types at the
    # root level of the data dict
    return {key: rebuild_dict_or_list(value) for key, value in parsed_data.items()}


def rebuild_dict_or_list(
    value: Union[dict, list]
) -> Union[int, str, bool, float, dict, list, None]:
    """
    Rebuild a dict or a list inside the parsed dict.

    :param value: a dict which can either be for structure or a value, or a list of
                  either value or structure dicts
    :return: a dict, list, or value
    """
    if isinstance(value, dict):
        if "_u" in value:
            # this is a value dict, return the original value
            return value["_u"]
        else:
            # this is a structural dict, pass each value through this function but
            # filter out fields that start with an underscore, unless they are the
            # special _id field
            return {
                key: rebuild_dict_or_list(value)
                for key, value in value.items()
                if not key.startswith("_") or key == "_id"
            }
    elif isinstance(value, list):
        # pass each element of the list through this function
        return [rebuild_dict_or_list(element) for element in value]
    else:
        # failsafe: just return the value. This should only really happen with lists
        # containing Nones (which is technically allowed)
        return value


class MSSElasticsearchHandler:
    """
    Class that handles requests to Elasticsearch for the MSS profile.
    """

    def __init__(self, es_hosts: List[str], limit: int = 20, mss_index: str = 'data-mss-latest'):
        """
        :param es_hosts: a list of elasticsearch hosts to use
        :param limit: the maximum number of simultaneous connections that can be made to
                         elasticsearch
        :param mss_index: the MSS index name in elasticsearch
        """
        self.es_hosts = cycle(es_hosts)
        self.es_session = create_client_session(limit, ssl=False)
        self.mss_index = mss_index

    async def get_mss_doc(self, guid: str) -> Tuple[int, Optional[dict]]:
        """
        Given a GUID, return the number of MSS doc hits and the first hit from the MSS index.

        :param guid: the GUID of the image
        :return: the total number of hits and the first hit's source
        """
        search_url = f'{next(self.es_hosts)}/{self.mss_index}/_search'
        search = Search().filter('term', **{'data.guid._k': guid}).extra(size=1, track_total_hits=True)
        async with self.es_session.post(search_url, json=search.to_dict()) as response:
            result = await response.json(encoding='utf-8')
            total = result['hits']['total']['value']
            first_doc = next((doc['_source']['data'] for doc in result['hits']['hits']), None)
            if first_doc:
                first_doc = rebuild_data(first_doc)
            return total, first_doc

    async def lookup_guid(self, asset_id: str) -> Tuple[int, Optional[str]]:
        """
        Given an old MAM asset ID, lookup the associated GUID.

        :param asset_id: the old MAM asset ID
        :return: the total hits and the GUID (or None if there are no hits)
        """
        search_url = f'{next(self.es_hosts)}/{self.mss_index}/_search'
        search = Search().filter('term', **{'data.old_asset_id._k': asset_id}).extra(size=1, track_total_hits=True)
        async with self.es_session.post(search_url, json=search.to_dict()) as response:
            result = await response.json(encoding='utf-8')
            total = result['hits']['total']['value']
            first_doc = next((doc['_source']['data'] for doc in result['hits']['hits']), None)
            if first_doc:
                guid = first_doc['guid']['_u']
            else:
                guid = None
            return total, guid

    async def get_status(self) -> dict:
        """
        Returns a dict describing the Elasticsearch cluster health.

        :return: a dict of status info
        """
        try:
            health_url = f'{next(self.es_hosts)}/_cluster/health'
            start_time = time.monotonic()
            async with self.es_session.get(health_url,
                                           timeout=ClientTimeout(total=5)) as response:
                return {
                    'status': (await response.json())['status'],
                    'response_time': time.monotonic() - start_time
                }
        except Exception as e:
            return {
                'status': 'unreachable',
                'error': str(e),
            }

    async def close(self):
        await self.es_session.close()


@dataclass
class MSSSourceFile(Fetchable):
    """
    Fetchable subclass representing an image in the MSS that can be retrieved.
    """
    emu_irn: int
    file: str
    is_original: bool
    chunk_size: int = 4096

    @property
    def public_name(self) -> str:
        return str(self.emu_irn)

    @property
    def store_path(self) -> Path:
        return Path(str(self.emu_irn), self.file)

    @property
    def url(self) -> str:
        return f'/nhmlive/{self.emu_irn}/{quote(self.file)}'

    @property
    def dams_url(self) -> str:
        return f'/nhmlive/{self.emu_irn}/damsurl'

    @staticmethod
    def check_url(emu_irn: int) -> str:
        return f'/nhmlive/{emu_irn}'


class StoreStreamError(Exception):

    def __init__(self, source: MSSSourceFile, url: str, cause: ClientError):
        super().__init__()
        self.source = source
        self.url = url
        self.cause = cause


class StoreStreamNoLength(Exception):
    pass


class MSSSourceStore(FetchCache):
    """
    Class that controls fetching source files from the MSS and then storing them in a cache and
    streaming them to users directly.
    """

    def __init__(self, root: Path, pool: Executor, mss_url: str, max_size: int, ttl: float,
                 mss_limit: int = 20, dams_limit: int = 5, mss_ssl: bool = True,
                 dams_ssl: bool = True, quality: int = 85, subsampling: str = '4:2:0'):
        """
        :param root: the path to store the data
        :param pool: the general purpose pool for offloading processing if necessary
        :param mss_url: the MSS base URL
        :param max_size: maximum size that the cache can grow to
        :param ttl: the maximum time a source file can be unused before it is removed
        :param mss_limit: the maximum number of simultaneous connections allowed to the MSS
        :param dams_limit: the maximum number of simultaneous connections allowed to the dams
        :param mss_ssl: whether to use SSL for MSS connections
        :param dams_ssl: whether to use SSL for the dams connections
        :param quality: jpeg quality of converted source files
        :param subsampling: jpeg subsampling value of converted source files
        """
        super().__init__(root, ttl, max_size)
        self.pool = pool
        self.mss_session = create_client_session(mss_limit, mss_ssl, mss_url)
        self.dm_session = create_client_session(dams_limit, dams_ssl)
        self._convert = partial(convert_image, quality=quality, subsampling=subsampling)
        self.stream_errors = Counter()

    async def _fetch(self, source: MSSSourceFile):
        """
        Fetch a file from the MSS and store it in the cache.

        :param source: the source
        """
        with tempfile.NamedTemporaryFile() as f:
            image_path = Path(f.name)
            async for chunk in self.stream(source):
                f.write(chunk)
            f.flush()

            # convert the image file, saving the data in a temp file but then moving it
            # to the source_path after the conversion is complete
            with tempfile.NamedTemporaryFile(delete=False) as g:
                target_path = Path(g.name)

                convert = partial(self._convert, image_path, target_path)
                try:
                    await asyncio.get_running_loop().run_in_executor(self.pool, convert)
                except Exception as cause:
                    raise MSSConversionFailure(source, cause)

                cache_path = self.root / source.store_path
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(target_path, cache_path)

    @asynccontextmanager
    async def _open_stream(self, source: MSSSourceFile) -> ClientResponse:
        """
        Opens a stream to the requested source data file. This is returned in the form of an aiohttp
        ClientResponse object which will be correctly closed once this context manager exits. The
        response may come from the MSS or it may come from the old dams service (via the damsurl
        file).

        :param source: the source file requested
        :return: a ClientResponse object
        """
        check_dams = False
        current_url = source.url
        stage = 'mss_direct'
        try:
            async with self.mss_session.get(source.url) as mss_response:
                if mss_response.status == 404:
                    check_dams = True
                else:
                    mss_response.raise_for_status()
                    yield mss_response

            if check_dams:
                current_url = source.dams_url
                stage = 'mss_indirect'
                async with self.mss_session.get(source.dams_url) as mss_response:
                    mss_response.raise_for_status()
                    # load the url in the response and fetch it
                    dams_url = await mss_response.text(encoding='utf-8')

                current_url = dams_url
                stage = 'dams'
                async with self.dm_session.get(dams_url) as dams_response:
                    dams_response.raise_for_status()
                    yield dams_response
        except ClientError as e:
            self.stream_errors[stage] += 1
            raise StoreStreamError(source, current_url, e)

    async def stream(self, source: MSSSourceFile):
        """
        Stream the given source file directly from the MSS/dams.

        :param source: the source
        :return: yields chunks of bytes
        """
        async with self._open_stream(source) as response:
            while chunk := await response.content.read(source.chunk_size):
                yield chunk

    async def get_file_size(self, source: MSSSourceFile) -> int:
        """
        Returns the file size of the given source by connecting to the backend service that can
        provide the source and returning the content-length. No body data is read, just the headers.

        :param source: the source
        :return: the size of the file in bytes
        """
        async with self._open_stream(source) as response:
            size = response.headers.get('content-length')
            if size is None:
                raise StoreStreamNoLength('The backend returned no content-length')
            return int(size)

    async def check_access(self, emu_irn: int) -> bool:
        """
        Check whether the EMu multimedia IRN is valid according to the APS (which is a part of the
        MSS).

        :param emu_irn: the EMu multimedia IRN
        :return: True if it's ok, False if not
        """
        async with self.mss_session.get(MSSSourceFile.check_url(emu_irn)) as response:
            return response.ok

    async def get_status(self) -> dict:
        """
        Add an MSS specific status to the basic stats returned by the FetchCache super
        implementation.

        :return: a dict of status information
        """
        status = await super().get_status()
        status['error_breakdown'] = self.stream_errors
        try:
            start_time = time.monotonic()
            async with self.mss_session.get('/nhmlive/status',
                                            timeout=ClientTimeout(total=5)) as response:
                status['mss_status'] = {
                    **(await response.json()),
                    'response_time': time.monotonic() - start_time,
                }
        except Exception as e:
            status['mss_status'] = {
                'status': 'unreachable',
                'error': str(e),
            }
        return status

    async def close(self):
        """
        Close down the store, this will simply close out our sessions and pools, all the files are
        left on the disk.
        """
        await self.mss_session.close()
        await self.dm_session.close()
