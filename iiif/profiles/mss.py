import asyncio
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, List, Optional
from urllib.parse import quote

import aiocron as aiocron
import aiohttp
import orjson
import shutil
import tempfile
import time
from contextlib import AsyncExitStack
from elasticsearch_dsl import Search
from fastapi import HTTPException
from itertools import cycle, chain

from iiif.config import Config
from iiif.profiles.base import AbstractProfile, ImageInfo
from iiif.utils import OnceRunner, convert_image, get_size


class MSSImageInfo(ImageInfo):
    """
    MSS variant of the ImageInfo class.
    """

    def __init__(self, profile_name: str, name: str, doc: dict):
        """
        :param profile_name: the name of the profile
        :param name: the name of the image, this will be EMu IRN
        :param doc: the image's doc from the elasticsearch MSS index
        """
        super().__init__(profile_name, name, doc.get('width', None), doc.get('height', None))
        self.doc = doc
        # the name of the original file as it appears on SCALE
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
    """
    Profile for the MSS service which provides access to the images stored in EMu.
    """

    def __init__(self,
                 name: str,
                 config: Config,
                 rights: str,
                 es_hosts: List[str],
                 mss_url: str,
                 ic_fast_pool_size: int,
                 ic_slow_pool_size: int,
                 collection_indices: List[str],
                 mss_index: str = 'mss',
                 es_limit: int = 100,
                 doc_cache_size: int = 1_000_000,
                 doc_exception_timeout: int = 0,
                 mss_limit: int = 20,
                 fetch_cache_size: int = 1_000_000,
                 fetch_exception_timeout: int = 0,
                 ic_quality: int = 85,
                 ic_subsampling: str = '4:2:0',
                 dm_limit: int = 4,
                 mss_ssl: bool = True,
                 dm_ssl: bool = True,
                 **kwargs
                 ):
        """
        :param name: the name of this profile
        :param config: the config object
        :param rights: the rights url for images served by this profile
        :param es_hosts: a list of elasticsearch hosts to use
        :param mss_url: mss base url
        :param ic_fast_pool_size: the size of the fast source image conversion pool (currently,
                                  source images that are already jpegs are put in this pool)
        :param ic_slow_pool_size: the size of the slow source image conversion pool (currently,
                                  source images that are not jpegs (most likely tiffs) are put in
                                  this pool)
        :param collection_indices: the indices to search to confirm the images can be accessed
        :param mss_index: the name of the MSS index
        :param es_limit: the number of elasticsearch requests that can be active simultaneously
        :param doc_cache_size: the size of the cache for doc results
        :param doc_exception_timeout: timeout for exceptions that occur during doc retrieval
        :param mss_limit: the number of mss requests that can be active simultaneously
        :param fetch_cache_size: the size of the cache for fetch results
        :param fetch_exception_timeout: timeout for exceptions that occur during fetching
        :param ic_quality: the jpeg quality to use when converting images
        :param ic_subsampling: the jpeg subsampling to use when converting images
        :param dm_limit: the number of dams requests that can be active simultaneously
        :param mss_ssl: boolean indicating whether ssl certificates should be checked when making
                        requests to mss
        :param dm_ssl: boolean indicating whether ssl certificates should be checked when making
                       requests to dams
        """
        super().__init__(name, config, rights, **kwargs)
        self.loop = asyncio.get_event_loop()
        # runners
        self.doc_runner = OnceRunner('doc', doc_cache_size, doc_exception_timeout)
        self.fetch_runner = OnceRunner('fetch', fetch_cache_size, fetch_exception_timeout)
        # elasticsearch
        self.es_hosts = cycle(es_hosts)
        self.es_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=es_limit))
        self.collection_indices = ','.join(collection_indices)
        self.mss_index = mss_index
        # MSS
        self.mss_url = mss_url
        self.mss_ssl = None if mss_ssl else False
        self.mss_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=mss_limit,
                                                                                ssl=mss_ssl))
        # dams
        self.dm_ssl = None if dm_ssl else False
        self.dm_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=dm_limit,
                                                                               ssl=dm_ssl))
        # image conversion
        self.ic_fast_pool = ProcessPoolExecutor(max_workers=ic_slow_pool_size)
        self.ic_slow_pool = ProcessPoolExecutor(max_workers=ic_fast_pool_size)
        self.ic_quality = ic_quality
        self.ic_subsampling = ic_subsampling
        # start a cron to make sure we check the access for each image we have cached every hour
        self.clean_up_cron = aiocron.crontab('0 * * * *', func=self.clean_up)

    async def clean_up(self) -> int:
        """
        Call this function to clean up cached data for images that shouldn't be accessible any more.
        This could be called on the same basis as the data importer but instead we simplify and call
        it way more, i.e. every hour (see the clean_up_cron attr defined above).

        Cached image data, source image data, image info metadata and cached info.json dicts are all
        removed if an image is no longer accessible.

        :return: the number of images that had data removed
        """
        removed = 0
        # use the cache dir, source dir and info.json cache as sources for the names to remove. We
        # could also use the fetch runner but given that we're using the source dir already and
        # that's where the fetch runner writes to it seems unnecessary
        name_sources = chain(self.cache_path.iterdir(), self.source_path.iterdir(),
                             self.info_json_cache.keys())
        names = {path.name for path in name_sources}
        for name in names:
            doc = await self.get_mss_doc(name, refresh=True)
            if doc is None:
                shutil.rmtree(self.cache_path / name)
                shutil.rmtree(self.source_path / name)
                self.fetch_runner.expire_matching(lambda task_id: task_id.startswith(f'{name}::'))
                self.info_json_cache.pop(name, None)
                removed += 1

        self.logger.warning(f'Clean up removed {removed} images that are no longer available')
        return removed

    async def get_info(self, name: str) -> Optional[MSSImageInfo]:
        """
        Given an image name (an EMu IRN) returns a MSSImageInfo object or None if the image can't be
        found/isn't allowed to be accessed. If the image doesn't have width and height stored in the
        elasticsearch index for whatever reason then the image will be retrieved and the size
        extracted.

        :param name: the EMu IRN of the image
        :return: an MSSImageInfo instance or None
        """
        doc = await self.get_mss_doc(name)
        if doc is None:
            return None

        info = MSSImageInfo(self.name, name, doc)

        if info.width is None or info.height is None:
            source_path = await self.fetch_source(info)
            info.width, info.height = get_size(source_path)

        return info

    async def fetch_source(self, info: MSSImageInfo,
                           target_size: Optional[Tuple[int, int]] = None) -> Path:
        """
        Given a MSSImageInfo object retrieve a source image that can be used to fulfill future image
        data requests. If no target size is provided then the size of the image is used as the
        target size, this could result in either the original image being retrieved or a derivative
        of the same size (sometimes, it seems, EMu generates a jpeg derivative of the same size as
        the original). If the target size is provided then the smallest image available in MSS that
        can fulfill the request will be used.

        :param info: an MSSImageInfo instance
        :param target_size: a target size 2-tuple or None
        :return: the path to the source image on disk
        """
        # if the target size isn't provided, fill it in using the full size of the image. This
        # provides useful functionality as it allows choose_file to return a full size derivative
        # of the image instead of the original if one is available
        if target_size is None:
            target_size = info.size
        file = info.choose_file(target_size)

        source_path = self.source_path / info.name / file
        if source_path.exists():
            return source_path

        task_id = f'{info.name}::{file}'
        is_original = file == info.original

        async def download():
            with tempfile.NamedTemporaryFile() as f:
                async for chunk in self._fetch_file(info.name, file, is_original):
                    f.write(chunk)
                f.flush()

                # to avoid the conversion pool sitting there just converting loads of giant tiffs
                # and blocking anything else from going through, we have two pools with different
                # priorities (this could have been implemented as a priority queue of some kind but
                # this is easier and time is money, yo! Might as well let the OS do the scheduling).
                if source_path.suffix.lower() in {'.jpeg', '.jpg'}:
                    # jpegs are quick to convert
                    pool = self.ic_fast_pool
                else:
                    # anything else is slower, tiffs for example
                    pool = self.ic_slow_pool

                # we've downloaded the file, convert it on a separate thread
                await self.loop.run_in_executor(pool, convert_image, Path(f.name), source_path,
                                                self.ic_quality, self.ic_subsampling)

        # we only want to be downloading a source file once so use the runner
        await self.fetch_runner.run(task_id, download)
        return source_path

    async def get_mss_doc(self, name: str, refresh: bool = False) -> Optional[dict]:
        """
        Retrieves a MSS doc and ensures it's should be accessible. For a doc to be returned instead
        of None:

            - the EMu IRN must be valid according the to MSS (i.e. the APS)
            - the doc must exist in the mss index in elasticsearch
            - the EMu IRN (i.e. the name) must be found in either the specimen, index lot or
              artefacts indices as an associated media item

        :param name: the image name (the EMu IRN)
        :param refresh: whether to enforce a refresh of the doc from elasticsearch rather than using
                        the cache
        :return: the mss doc as a dict or None
        """

        async def get_doc() -> Optional[dict]:
            # first, check that we have a document in the mss index
            doc_url = f'{next(self.es_hosts)}/{self.mss_index}/_doc/{name}'
            async with self.es_session.get(doc_url) as response:
                text = await response.text(encoding='utf-8')
                info = orjson.loads(text)
                if not info['found']:
                    return None

            # next, check that the irn is associated with a record in the collection datasets
            count_url = f'{next(self.es_hosts)}/{self.collection_indices}/_count'
            search = Search() \
                .filter('term', **{'data.associatedMedia._id': name}) \
                .filter('term', **{'meta.versions': int(time.time() * 1000)})
            async with self.es_session.post(count_url, json=search.to_dict()) as response:
                text = await response.text(encoding='utf-8')
                if orjson.loads(text)['count'] == 0:
                    return None

            # finally, check with mss that the irn is valid
            async with self.mss_session.get(f'{self.mss_url}/{name}') as response:
                if not response.ok:
                    return None

            # if we get here then all 3 checks have passed
            return info['_source']

        if refresh:
            self.doc_runner.expire(name)
        return await self.doc_runner.run(name, get_doc)

    async def resolve_filename(self, name: str) -> Optional[str]:
        """
        Given an image name (i.e. IRN), return the original filename or None if the image can't be
        found.

        :param name: the image name (in this case the IRN)
        :return: the filename or None
        """
        doc = await self.get_mss_doc(name)
        return doc['file'] if doc is not None else None

    async def stream_original(self, name: str, chunk_size: int = 4096, raise_errors=True):
        """
        Async generator which yields the bytes of the original image for the given image name (EMu
        IRN). If the image isn't available then nothing is yielded.

        :param name: the name of the image (EMu IRN)
        :param chunk_size: the size of the chunks to yield
        :param raise_errors: whether to raise errors when they happen or simply stop, swallowing the
                             error
        """
        doc = await self.get_mss_doc(name)
        if doc is not None:
            try:
                async for chunk in self._fetch_file(name, doc['file'], True, chunk_size):
                    yield chunk
            except Exception as e:
                if raise_errors:
                    raise e

    async def _fetch_file(self, name: str, file: str, is_original: bool, chunk_size: int = 4096):
        """
        Fetches a file from MSS or, if the file is the original and doesn't exist in MSS, the old
        dams servers. Once a source for the requested file is located, the bytes are yielded in
        chunks of chunk_size.

        :param name: the name of the file (the EMu IRN)
        :param file: the name of the file to retrieve
        :param is_original: whether the file is an original image (if it is and the file doesn't
                            exist in MSS we'll try looking for a damsurl file)
        """
        async with AsyncExitStack() as stack:
            file_url = f'{self.mss_url}/{name}/{quote(file)}'
            response = await stack.enter_async_context(self.mss_session.get(file_url))

            if response.status == 401:
                raise HTTPException(status_code=401, detail=f'Access denied')

            if response.status == 404 and is_original:
                # check for a damsurl file
                damsurl_file = f'{self.mss_url}/{name}/damsurl'
                response = await stack.enter_async_context(self.mss_session.get(damsurl_file))
                response.raise_for_status()

                # load the url in the response and fetch it
                damsurl = await response.text(encoding='utf-8')
                response = await stack.enter_async_context(self.dm_session.get(damsurl))

            response.raise_for_status()

            while chunk := await response.content.read(chunk_size):
                yield chunk

    async def close(self):
        """
        Close down this profile.
        """
        await self.es_session.close()
        await self.mss_session.close()
        await self.dm_session.close()
        self.ic_fast_pool.shutdown()
        self.ic_slow_pool.shutdown()

    async def get_status(self, full: bool = False) -> dict:
        """
        Returns some nice stats about what the runners are up to and such.

        :return: a dict of stats
        """
        status = await super().get_status(full)
        runners = (self.doc_runner, self.fetch_runner)
        status['runners'] = {
            runner.name: await runner.get_status() for runner in runners
        }
        # TODO: add pool stats
        return status
