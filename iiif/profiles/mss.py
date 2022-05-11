from concurrent.futures import ProcessPoolExecutor, Executor

import asyncio
import json
import logging
import shutil
import tempfile
import time
from cachetools import TTLCache
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from elasticsearch_dsl import Search
from fastapi import HTTPException
from functools import partial
from itertools import cycle
from pathlib import Path
from typing import List
from typing import Tuple, Optional
from urllib.parse import quote

from iiif.config import Config
from iiif.exceptions import Timeout, IIIFServerException, ImageNotFound
from iiif.profiles.base import AbstractProfile, ImageInfo
from iiif.utils import Locker, convert_image, create_client_session, FetchCache, Fetchable
from iiif.utils import get_size


class MissingCollectionRecord(ImageNotFound):

    def __init__(self, profile: str, name: str, emu_irn: int):
        super().__init__(profile, name, log=f"Couldn't find collection record associated with "
                                            f"multimedia IRN {emu_irn} [guid: {name}]")
        self.emu_irn = emu_irn


class MSSAccessDenied(ImageNotFound):

    def __init__(self, profile: str, name: str, emu_irn: int):
        super().__init__(profile, name, log=f"MSS denied access to multimedia IRN {emu_irn} "
                                            f"[guid: {name}]")
        self.emu_irn = emu_irn


class MSSDocDuplicates(ImageNotFound):

    def __init__(self, profile: str, name: str, total: int):
        super().__init__(profile, name, log=f"Found {total} MSS docs for the guid {name}")
        self.total = total


class MSSDocNotFound(ImageNotFound):

    def __init__(self, profile: str, name: str):
        super().__init__(profile, name, log=f"No MSS doc found for the guid {name}")


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
                 rights: str,
                 es_hosts: List[str],
                 collection_indices: List[str],
                 mss_url: str,
                 mss_ssl: bool = True,
                 mss_index: str = 'mss',
                 mss_limit: int = 10,
                 es_limit: int = 10,
                 info_cache_size: int = 100_000,
                 info_cache_ttl: float = 43_200,
                 info_lock_ttl: float = 60,
                 source_cache_size: int = 1024 * 1024 * 256,
                 source_cache_ttl: float = 12 * 60 * 60,
                 convert_fast_pool_size: int = 1,
                 convert_slow_pool_size: int = 1,
                 convert_quality: int = 85,
                 convert_subsampling: str = '4:2:0',
                 dams_limit: int = 4,
                 dams_ssl: bool = True,
                 **kwargs
                 ):
        """
        :param name: the name of this profile
        :param config: the Config object
        :param rights: the rights url for images served by this profile
        :param es_hosts: a list of elasticsearch hosts to use
        :param collection_indices: the indices to search to confirm the images can be accessed
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
        :param convert_fast_pool_size: how many processes to use for converting "fast" images
        :param convert_slow_pool_size: how many processes to use for converting "slow" images
        :param convert_quality: quality to use when converting a source to a jpeg
        :param convert_subsampling: subsampling value to use when converting a source to a jpeg
        :param dams_limit: the maximum number of simultaneous connections that can be made to the
                           old dams service (to retrieve files stored at the damsurlfile value)
        :param dams_ssl: boolean indicating whether ssl certificates should be checked when making
                         requests to dams
        :param kwargs: extra kwargs for the AbstractProfile base class __init__
        """
        super().__init__(name, config, rights, **kwargs)
        self.info_cache = TTLCache(info_cache_size, info_cache_ttl)
        self.get_info_locker = Locker(default_timeout=info_lock_ttl)

        self.get_mss_doc_locker = Locker(default_timeout=info_lock_ttl)
        self.mss_doc_cache = TTLCache(maxsize=info_cache_size, ttl=info_cache_ttl)

        self.es_handler = MSSElasticsearchHandler(es_hosts, collection_indices, es_limit, mss_index)
        self.store = MSSSourceStore(self.source_path, mss_url, source_cache_size,
                                    source_cache_ttl, mss_limit, dams_limit, mss_ssl,
                                    dams_ssl, convert_slow_pool_size, convert_fast_pool_size,
                                    convert_quality, convert_subsampling)

    async def get_info(self, name: str) -> MSSImageInfo:
        """
        Given an image name (a GUID) returns a MSSImageInfo object or raise an HTTPException if the
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
        except asyncio.TimeoutError:
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
        source = MSSSourceFile(str(info.emu_irn), file, info.original == file)
        async with self.store.use(source) as path:
            yield path

    async def get_mss_doc(self, name: str) -> dict:
        """
        Retrieves an MSS doc and ensures it's should be accessible. For a doc to be returned:

            - the GUID (i.e. the name) must be unique and exist in the mss elasticsearch index
            - the EMu IRN that the GUID maps to must be valid according the to the MSS (specifically
              the APS)
            - the GUID (i.e. the name) must be found in either the specimen, index lot or
              artefacts indices as an associated media item

        :param name: the image name (a GUID)
        :return: the mss doc as a dict or None
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

                    # TODO: I think we can get rid of this requirement now that we're using GUIDs
                    #       instead of irns - you can't guess a GUID
                    # check if there's an associated collection record
                    if not await self.es_handler.has_collection_record(emu_irn):
                        raise MissingCollectionRecord(self.name, name, emu_irn)

                    # finally, check with mss that the irn is valid
                    if not await self.store.check_access(emu_irn):
                        raise MSSAccessDenied(self.name, name, emu_irn)

                    self.mss_doc_cache[name] = doc
                    return doc
                except IIIFServerException:
                    raise
                except Exception as cause:
                    e = ImageNotFound(self.name, name, cause=cause, level=logging.ERROR)
                    raise e from cause
        except asyncio.TimeoutError:
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

    async def stream_original(self, name: str, chunk_size: int = 4096, raise_errors=True):
        """
        Async generator which yields the bytes of the original image for the given image name (EMu
        IRN). If the image isn't available then nothing is yielded.

        :param name: the name of the image (EMu IRN)
        :param chunk_size: the size of the chunks to yield
        :param raise_errors: whether to raise errors when they happen or simply stop, swallowing the
                             error
        """
        try:
            doc = await self.get_mss_doc(name)
            source = MSSSourceFile(doc['id'], doc['file'], True, chunk_size)
            async for chunk in self.store.stream(source):
                yield chunk
        except Exception as e:
            if raise_errors:
                raise e

    async def close(self):
        """
        Close down this profile.
        """
        await self.es_handler.close()
        await self.store.close()


class MSSElasticsearchHandler:
    """
    Class that handles requests to Elasticsearch for the MSS profile.
    """

    def __init__(self, es_hosts: List[str], collection_indices: List[str], limit: int = 20,
                 mss_index: str = 'mss'):
        """
        :param es_hosts: a list of elasticsearch hosts to use
        :param collection_indices: the indices to search to confirm the images can be accessed
        :param limit: the maximum number of simultaneous connections that can be made to
                         elasticsearch
        :param mss_index: the MSS index name in elasticsearch
        """
        self.es_hosts = cycle(es_hosts)
        self.es_session = create_client_session(limit, ssl=False)
        self.collection_indices = ','.join(collection_indices)
        self.mss_index = mss_index

    async def get_mss_doc(self, guid: str) -> Tuple[int, Optional[dict]]:
        """
        Given a GUID, return the number of MSS doc hits and the first hit from the MSS index.

        :param guid: the GUID of the image
        :return: the total number of hits and the first hit's source
        """
        search_url = f'{next(self.es_hosts)}/{self.mss_index}/_search'
        search = Search().filter('term', **{'guid.keyword': guid}).extra(size=1)
        async with self.es_session.post(search_url, json=search.to_dict()) as response:
            text = await response.text(encoding='utf-8')
            result = json.loads(text)
            total = result['hits']['total']
            first_doc = next((doc['_source'] for doc in result['hits']['hits']), None)
            return total, first_doc

    async def has_collection_record(self, emu_irn: int) -> bool:
        """
        Check whether the given image IRN is associated with a collection record.

        :param emu_irn: the EMu IRN
        :return: True if it is, False if not
        """
        count_url = f'{next(self.es_hosts)}/{self.collection_indices}/_count'
        search = Search() \
            .filter('term', **{'data.associatedMedia._id': emu_irn}) \
            .filter('term', **{'meta.versions': int(time.time() * 1000)})
        async with self.es_session.post(count_url, json=search.to_dict()) as response:
            text = await response.text(encoding='utf-8')
            return json.loads(text)['count'] > 0

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


class MSSSourceStore(FetchCache):
    """
    Class that controls fetching source files from the MSS and then storing them in a cache and
    streaming them to users directly.
    """

    def __init__(self, root: Path, mss_url: str, max_size: int, ttl: float,
                 mss_limit: int = 20, dams_limit: int = 5, mss_ssl: bool = True,
                 dams_ssl: bool = True, slow_pool_size: int = 1, fast_pool_size: int = 1,
                 quality: int = 85, subsampling: str = '4:2:0'):
        """
        :param root: the path to store the data
        :param mss_url: the MSS base URL
        :param max_size: maximum size that the cache can grow to
        :param ttl: the maximum time a source file can be unused before it is removed
        :param mss_limit: the maximum number of simultaneous connections allowed to the MSS
        :param dams_limit: the maximum number of simultaneous connections allowed to the dams
        :param mss_ssl: whether to use SSL for MSS connections
        :param dams_ssl: whether to use SSL for the dams connections
        :param slow_pool_size: size of the "slow" conversion pool
        :param fast_pool_size: size of the "fast" conversion pool
        :param quality: jpeg quality of converted source files
        :param subsampling: jpeg subsampling value of converted source files
        """
        super().__init__(root, ttl, max_size)
        self.mss_session = create_client_session(mss_limit, mss_ssl, mss_url)
        self.dm_session = create_client_session(dams_limit, dams_ssl)
        self._fast_pool = ProcessPoolExecutor(max_workers=slow_pool_size)
        self._slow_pool = ProcessPoolExecutor(max_workers=fast_pool_size)
        self._convert = partial(convert_image, quality=quality, subsampling=subsampling)

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

                pool = self._choose_convert_pool(source.file)
                convert = partial(self._convert, image_path, target_path)
                await asyncio.get_running_loop().run_in_executor(pool, convert)

                cache_path = self.root / source.store_path
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(target_path, cache_path)

    async def stream(self, source: MSSSourceFile):
        """
        Stream the given source file directly from the MSS.

        :param source: the source
        :return: yields chunks of bytes
        """
        async with AsyncExitStack() as stack:
            response = await stack.enter_async_context(self.mss_session.get(source.url))

            if response.status == 401:
                raise HTTPException(status_code=401, detail=f'Access denied')

            if response.status == 404 and source.is_original:
                # check for a damsurl file
                response = await stack.enter_async_context(self.mss_session.get(source.dams_url))
                response.raise_for_status()

                # load the url in the response and fetch it
                damsurl = await response.text(encoding='utf-8')
                response = await stack.enter_async_context(self.dm_session.get(damsurl))

            response.raise_for_status()

            while chunk := await response.content.read(source.chunk_size):
                yield chunk

    def _choose_convert_pool(self, file: str) -> Executor:
        """
        Pick the pool to use for converting the given file. JPEGs are converted in the fast pool
        whereas everything else is done in the slow pool.

        To avoid the conversion pool sitting there just converting loads of giant tiffs and blocking
        anything else from going through, we have two pools with different priorities (this could
        have been implemented as a priority queue of some kind but this is easier and time is money,
        yo! Might as well let the OS do the scheduling).

        :param file: the file name
        :return: which pool to convert in
        """
        if Path(file).suffix.lower() in ('.jpeg', '.jpg'):
            return self._fast_pool
        else:
            return self._slow_pool

    async def check_access(self, emu_irn: int) -> bool:
        """
        Check whether the EMu multimedia IRN is valid according to the APS (which is a part of the
        MSS).

        :param emu_irn: the EMu multimedia IRN
        :return: True if it's ok, False if not
        """
        async with self.mss_session.get(MSSSourceFile.check_url(emu_irn)) as response:
            return response.ok

    async def close(self):
        """
        Close down the store, this will simply close out our sessions and pools, all the files are
        left on the disk.
        """
        await self.mss_session.close()
        await self.dm_session.close()
        self._fast_pool.shutdown()
        self._slow_pool.shutdown()
