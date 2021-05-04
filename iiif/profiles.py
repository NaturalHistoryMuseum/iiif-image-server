#!/usr/bin/env python3
# encoding: utf-8

import abc
import aiocron as aiocron
import aiohttp
import asyncio
import logging
import orjson
import shutil
import tempfile
import time
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import AsyncExitStack
from elasticsearch_dsl import Search
from itertools import cycle, chain
from lru import LRU
from pathlib import Path
from typing import Tuple, List, Optional, Any, Union
from urllib.parse import quote

from iiif.config import Config, registry
from iiif.utils import OnceRunner, get_path_stats, convert_image, get_size, get_mss_base_url_path, \
    generate_sizes, create_logger


class ImageInfo:
    """
    Base info class which holds the basic details of an image.
    """

    def __init__(self, profile_name: str, name: str, width: int, height: int):
        """
        :param profile_name: the name of the profile this image belongs to
        :param name: the name of image (this should be unique within the profile's jurisdiction)
        :param width: the width of the image
        :param height: the height of the image
        """
        self.profile_name = profile_name
        self.name = name
        self.width = width
        self.height = height
        self.identifier = f'{self.profile_name}:{self.name}'

    @property
    def size(self) -> Tuple[int, int]:
        """
        Returns width and height as a 2-tuple.
        :return: a 2-tuple containing the width and height
        """
        return self.width, self.height

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ImageInfo):
            return self.identifier == other.identifier
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.identifier)


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


class AbstractProfile(abc.ABC):
    """
    Abstract base class for all profiles. Each subclass defines a profile which can be used to
    retrieve the source image data required to complete a IIIF request (either info.json or actual
    data). Once a profile has been used to retrieve a source image and make it real on disk as a
    jpeg file, it can be processed in a common way (see the processing module).
    """

    def __init__(self, name: str, config: Config, rights: str, info_json_cache_size: int = 1000,
                 log_level: Union[int, str] = logging.WARNING):
        """
        :param name: the name of the profile, should be unique across profiles
        :param config: the config object
        :param rights: the rights definition for all images handled by this profile
        :param info_json_cache_size: the size of the info.json cache
        """
        self.name = name
        self.config = config
        # this is where all of our source images will be stored
        self.source_path = config.source_path / name
        # this is where all of our processed images will be stored
        self.cache_path = config.cache_path / name
        self.rights = rights
        self.source_path.mkdir(exist_ok=True)
        self.cache_path.mkdir(exist_ok=True)
        self.info_json_cache = LRU(info_json_cache_size)
        self.logger = create_logger(name, log_level)

    @abc.abstractmethod
    async def get_info(self, name: str) -> Optional[ImageInfo]:
        """
        Returns an instance of ImageInfo or one of it's subclasses for the given name. Note that
        this function does not generate the info.json, see generate_info_json below for that!

        :param name: the name of the image
        :return: an info object or None if the image wasn't available (for any reason)
        """
        pass

    @abc.abstractmethod
    async def fetch_source(self, info: ImageInfo,
                           target_size: Optional[Tuple[int, int]] = None) -> Optional[Path]:
        """
        Ensures the source file required for the optional target size is available

        :param info:
        :param target_size:
        :return:
        """
        pass

    async def generate_info_json(self, info: ImageInfo, iiif_level: str) -> dict:
        """
        Generates an info.json dict for the given image. The info.json is cached locally in this
        profile's attributes.

        :param info: the ImageInfo object to create the info.json dict for
        :param iiif_level: the IIIF image server compliance level to include in the info.json
        :return: the generated or cached info.json dict for the image
        """
        # if the image's info.json isn't cached, create and add the complete info.json to the cache
        if info.name not in self.info_json_cache:
            id_url = f'{self.config.base_url}/{info.identifier}'
            self.info_json_cache[info.name] = {
                '@context': 'http://iiif.io/api/image/3/context.json',
                'id': id_url,
                # mirador/openseadragon seems to need this to work even though I don't think it's
                # correct under the IIIF image API v3
                '@id': id_url,
                'type': 'ImageService3',
                'protocol': 'http://iiif.io/api/image',
                'width': info.width,
                'height': info.height,
                'rights': self.rights,
                'profile': iiif_level,
                'tiles': [
                    {'width': 512, 'scaleFactors': [1, 2, 4, 8, 16]},
                    {'width': 256, 'scaleFactors': [1, 2, 4, 8, 16]},
                    {'width': 1024, 'scaleFactors': [1, 2, 4, 8, 16]},
                ],
                'sizes': generate_sizes(info.width, info.height, self.config.min_sizes_size),
                # suggest to clients that upscaling isn't supported
                'maxWidth': info.width,
                'maxHeight': info.height,
            }

        return self.info_json_cache[info.name]

    async def close(self):
        """
        Close down the profile ensuring any resources are released. This will be called before
        server shutdown.
        """
        pass

    async def get_status(self) -> dict:
        """
        Returns some stats about the profile.

        :return: a dict of stats
        """
        return {
            'name': self.name,
            'info_json_cache_size': len(self.info_json_cache),
            'sources': get_path_stats(self.source_path),
            'cache': get_path_stats(self.cache_path),
        }


@registry.register('disk')
class OnDiskProfile(AbstractProfile):
    """
    A profile representing source files that are already on disk and don't need to be fetched from
    an external source.
    """

    async def get_info(self, name: str) -> Optional[ImageInfo]:
        """
        Given an image name, returns an info object for it. If the image doesn't exist on disk then
        None is returned.

        :param name: the image name
        :return: None if the image doesn't exist on disk, or an ImageInfo instance
        """
        source = self._get_source(name)
        if not source.exists():
            return None
        else:
            size = get_size(self._get_source(name))
            return ImageInfo(self.name, name, *size)

    async def fetch_source(self, info: ImageInfo,
                           target_size: Optional[Tuple[int, int]] = None) -> Optional[Path]:
        """
        Given an info object returns the path to the on disk image source. The target size is
        ignored by this function because we only have the full size original and nothing else.

        :param info: the image info object
        :param target_size: a target size - this is ignored by this function
        :return: the path to the source image on disk
        """
        source_path = self._get_source(info.name)
        return source_path if source_path.exists() else None

    def _get_source(self, name: str) -> Path:
        """
        Returns the path to the given name in this profile.

        :param name: the name of the image
        :return: the path to the image
        """
        return self.source_path / name


@registry.register('mss')
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
                 info_json_cache_size: int = 1000,
                 log_level: Union[int, str] = logging.WARNING,
                 mss_index: str = 'mss',
                 es_limit: int = 100,
                 doc_cache_size: int = 1_000_000,
                 doc_exception_timeout: int = 0,
                 mss_limit: int = 20,
                 fetch_cache_size: int = 1_000_000,
                 fetch_exception_timeout: int = 0,
                 ic_quality: int = 80,
                 ic_subsampling: int = 0,
                 dm_limit: int = 4
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
        :param info_json_cache_size: the size of the info.json cache for this profile
        :param log_level: the log level for this profile
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
        """
        super().__init__(name, config, rights, info_json_cache_size, log_level)
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
        self.mss_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=mss_limit))
        # dams
        self.dm_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=dm_limit))
        # image conversion
        self.ic_fast_pool = ProcessPoolExecutor(max_workers=ic_slow_pool_size)
        self.ic_slow_pool = ProcessPoolExecutor(max_workers=ic_fast_pool_size)
        self.ic_quality = ic_quality
        self.ic_subsampling = ic_subsampling
        # start a cron to make sure we check the access for each image we have cached every hour
        self.clean_up_cron = aiocron.crontab('0 * * * *', func=self.clean_up)

    async def clean_up(self) -> int:
        removed = 0
        names = {path.name for path in chain(self.cache_path.iterdir(), self.source_path.iterdir())}
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

            - the doc must exist in the mss index in elasticsearch
            - the EMu IRN (i.e. the name) must be found in either the specimen, index lot or
              artefacts indices as an associated media item

        :param name: the image name (the EMu IRN)
        :param refresh: whether to enforce a refresh of the doc from elasticsearch rather than using
                        the cache
        :return: the mss doc as a dict or None
        """

        async def get_doc() -> Optional[dict]:
            count_url = f'{next(self.es_hosts)}/{self.collection_indices}/_count'
            search = Search() \
                .filter('term', **{'data.associatedMedia._id': name}) \
                .filter('term', **{'meta.versions': int(time.time() * 1000)})
            async with self.es_session.post(count_url, json=search.to_dict()) as response:
                text = await response.text(encoding='utf-8')
                if orjson.loads(text)['count'] == 0:
                    return None

            doc_url = f'{next(self.es_hosts)}/{self.mss_index}/_doc/{name}'
            async with self.es_session.get(doc_url) as response:
                text = await response.text(encoding='utf-8')
                info = orjson.loads(text)
                if info['found']:
                    return info['_source']
                else:
                    return None

        if refresh:
            self.doc_runner.expire(name)
        return await self.doc_runner.run(name, get_doc)

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
        base = get_mss_base_url_path(name)

        async with AsyncExitStack() as stack:
            file_url = f'{self.mss_url}/{base}/{quote(file)}'
            response = await stack.enter_async_context(self.mss_session.get(file_url))

            if response.status == 404 and is_original:
                # check for a damsurl file
                damsurl_file = f'{self.mss_url}/{base}/damsurl'
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

    async def get_status(self) -> dict:
        """
        Returns some nice stats about what the runners are up to and such.

        :return: a dict of stats
        """
        status = await super().get_status()
        runners = (self.doc_runner, self.fetch_runner)
        status['runners'] = {
            runner.name: await runner.get_status() for runner in runners
        }
        # TODO: add pool stats
        return status
