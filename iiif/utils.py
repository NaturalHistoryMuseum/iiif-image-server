#!/usr/bin/env python3
# encoding: utf-8
from collections import OrderedDict, Counter

import abc
import aiohttp
import asyncio
import humanize
import io
import logging
import mimetypes
from PIL import Image
from contextlib import suppress, asynccontextmanager
from functools import lru_cache
from itertools import count
from jpegtran import JPEGImage
from pathlib import Path
from typing import Optional, Tuple, Union, Any
from wand.exceptions import WandException
from wand.image import Image as WandImage

mimetypes.init()

# this is all assuming we're using uvicorn...
uvicorn_logger = logging.getLogger('uvicorn.error')
# use this logger to dump stuff into the same channels as the uvicorn logs
logger = uvicorn_logger.getChild('iiif')


class Locker:
    """
    A normal async Lock surrounded by a timeout.
    """

    def __init__(self, default_timeout: float = 0):
        """
        :param default_timeout: how long to timeout the locks for by default (defaults to 0 which
                                means don't timeout, just lock forever).
        """
        self._locks = {}
        self.default_timeout = default_timeout

    @asynccontextmanager
    async def acquire(self, key: Any, timeout: Optional[float] = None):
        if timeout is None:
            timeout = self.default_timeout

        lock = self._locks.setdefault(key, asyncio.Lock())
        if timeout > 0:
            acquired = False
            try:
                acquired = await asyncio.wait_for(lock.acquire(), timeout=timeout)
                yield
            except asyncio.TimeoutError:
                raise
            finally:
                if acquired:
                    lock.release()
        else:
            async with lock:
                yield

    def is_locked(self, key: Any) -> bool:
        """
        Check if the given key is locked.

        :param key: the key
        :return: True if the key is locked, False if not
        """
        return key in self._locks and self._locks[key].locked()


def get_size(path: Path) -> Tuple[int, int]:
    """
    Returns the size of the image at the given path.

    :param path: the image path on disk
    :return: a 2-tuple containing the width and height
    """
    with Image.open(path) as pillow_image:
        return pillow_image.width, pillow_image.height


@lru_cache(maxsize=65536)
def generate_sizes(width: int, height: int, minimum_size: int = 200):
    """
    Produces the sizes array for the given width and height combination. Function results are
    cached.

    :param width: the width of the source image
    :param height: the height of the source image
    :param minimum_size: the minimum dimension size to include in the returned list
    :return: a list of sizes in descending order
    """
    # always include the original image size in the sizes list
    sizes = [{'width': width, 'height': height}]
    for i in count(1):
        factor = 2 ** i
        new_width = width // factor
        new_height = height // factor
        # stop when either dimension is smaller than
        if new_width < minimum_size or new_height < minimum_size:
            break
        sizes.append({'width': new_width, 'height': new_height})

    return sizes


def convert_image(image_path: Path, target_path: Path, quality: int = 80,
                  subsampling: str = '4:2:0'):
    """
    Given the path to an image, outputs the image to the target path in jpeg format. This should
    happen to all images that will have processing done on them subsequently as it means we can use
    a common approach to all files - namely using jpegtran on them.

    :param image_path: the path to the source image
    :param target_path: the path to output the jpeg version of the image
    :param quality: the jpeg quality setting to use
    :param subsampling: the jpeg subsampling to use
    """
    # given this is usually run in a separate process, make sure we have disabled bomb errors
    disable_bomb_errors()
    try:
        with WandImage(filename=str(image_path)) as image:
            if image.format.lower() == 'jpeg':
                # if it's a jpeg, remove the orientation exif tag. We do this because through trial
                # and error it seems the dimensions provided by EMu are non-orientated and therefore
                # we need to work on the images without their orientation too and serve them up
                # without it otherwise things start to go awry
                image.strip()
            else:
                image.format = 'jpeg'

            image.compression_quality = quality
            image.options['jpeg:sampling-factor'] = subsampling

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open('wb') as f:
                image.save(file=f)
    except WandException:
        with Image.open(image_path) as image:
            if image.format.lower() == 'jpeg':
                exif = image.getexif()
                # this is the orientation tag, remove it if it's there
                exif.pop(0x0112, None)
                image.info['exif'] = exif.tobytes()

            target_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(target_path, format='jpeg', quality=quality, subsampling=subsampling)


def parse_identifier(identifier: str) -> Tuple[Optional[str], str]:
    """
    Parse the identifier into a profile name and image name. The profile name may be omitted in
    which case a (None, <name>) is returned.

    :param identifier: the image identifier
    :return: a 2-tuple of 2 strings or None and a string
    """
    if ':' in identifier:
        return tuple(identifier.split(':', 1))
    else:
        return None, identifier


def to_pillow(image: JPEGImage) -> Image:
    """
    Convert the given JPEGImage to a Pillow image.

    :param image: a JPEGImage object
    :return: a Pillow image object
    """
    return Image.open(io.BytesIO(image.as_blob()))


def to_jpegtran(image: Image) -> JPEGImage:
    """
    Convert the given Pillow image to a JPEGImage image.

    :param image: a Pillow image object
    :return: a JPEGImage object
    """
    output = io.BytesIO()
    image.save(output, format='jpeg')
    output.seek(0)
    return JPEGImage(blob=output.read())


def get_mimetype(filename: Union[str, Path]) -> str:
    """
    Given a filename, guesses the mime type and returns it. If no sensible guess can be made then
    application/octet-stream is returned.

    :param filename: the name of the file
    :return: a mime type
    """
    guess = mimetypes.guess_type(filename)[0]
    # use octet stream as the default
    return guess if guess is not None else 'application/octet-stream'


def create_client_session(limit: int, ssl: bool = True,
                          base_url: Optional[str] = None) -> aiohttp.ClientSession:
    """
    Convenience function to create a new aiohttp session object.

    :param limit: the maximum number of simultaneous connections allowed
    :param ssl: whether to verify SSL certs or not
    :param base_url: the base URL for all requests made with this session
    :return: a new aiohttp.ClientSession object
    """
    # if we want to use ssl this parameter needs to be set to None, if not then False is used
    ssl = None if ssl else False
    return aiohttp.ClientSession(base_url=base_url,
                                 connector=aiohttp.TCPConnector(limit=limit, ssl=ssl))


class Fetchable(abc.ABC):
    """
    Abstract base class of something that can be fetched by the FetchCache.
    """

    @property
    @abc.abstractmethod
    def public_name(self) -> str:
        """
        A name to use in errors.
        :return: a name that is acceptable for public users.
        """
        ...

    @property
    @abc.abstractmethod
    def store_path(self) -> Path:
        """
        Where this fetchable will be stored in the store. This should be relative to the store root.
        :return: the relative path where this fetchable should be stored
        """
        ...


class FetchCache(abc.ABC):
    """
    A cache with TTL and LRU functionality which allows custom fetching of the data put in it.
    """

    def __init__(self, root: Path, ttl: float, max_size: float, clean_empty_dirs: bool = True,
                 fetch_timeout: float = 120):
        """
        Note that this init will automatically call self.load() and therefore populate the cache.
        This could take time if the cache is enormous.

        :param root: the root under which all data will be stored
        :param ttl: how long untouched files can stay in the cache before being removed
        :param max_size: the maximum number of bytes that can be stored in the cache
        :param clean_empty_dirs: whether to delete empty parent dirs when removing expired files
        :param fetch_timeout: how long to wait for the _fetch function to complete
        """
        self.root = root
        self.ttl = ttl
        self.max_size = max_size
        self.clean_empty_dirs = clean_empty_dirs
        self.fetch_timeout = fetch_timeout
        self.total_size = 0
        self._in_use = Counter()
        self._sizes = {}
        self._locker = Locker()
        self._cleaners = OrderedDict()
        self.requests = 0
        self.completed = 0
        self.errors = 0
        # let's see what currently exists
        self.load()

    def load(self):
        """
        Scan the root dir and add any found files into the cache.
        """
        for path in self.root.rglob('*'):
            if path.is_file():
                size = path.stat().st_size
                self._sizes[path] = size
                self._cleaners[path] = self._schedule_clean_up(path)
                self.total_size += size
        logger.info(f'Found {len(self._sizes)} files in {self.root} [{self.pct}]')

    @property
    def pct(self) -> str:
        """
        Returns the percentage usage of the cache as a number with 2 decimal points. This is mainly
        a convenience for logging.

        :return: the percentage the cache is in use as a formatted string
        """
        return f'{((self.total_size / self.max_size) * 100):.2f}%'

    def _clean_up(self, path: Path):
        """
        Callback scheduled to clean up paths when their TTL expires.

        :param path: the path to clean up
        """
        with suppress(KeyError):
            self._cleaners.pop(path)

        if path not in self._in_use:
            if path.exists():
                with suppress(Exception):
                    path.unlink()
                    if self.clean_empty_dirs:
                        for parent in path.parents:
                            if parent != self.root and not any(parent.iterdir()):
                                parent.rmdir()
                            else:
                                break
            self.total_size -= self._sizes.pop(path, 0)
            logger.info(f'Cleaned up {path} [{self.pct}]')

    def _schedule_clean_up(self, path: Path) -> asyncio.TimerHandle:
        """
        Add a callback to remove the given path after the TTL.

        :param path: the path to remove
        :return: a TimerHandle object
        """
        return asyncio.get_running_loop().call_later(self.ttl, self._clean_up, path)

    def __contains__(self, relative_path: Path) -> bool:
        """
        Checks whether the given path is in use or not. This only checks the in use dict, not the
        cleaners dict.

        :param relative_path: the path (relative to the root)
        :return: True if the path is in use, False if not
        """
        return self.root / relative_path in self._in_use

    @abc.abstractmethod
    async def _fetch(self, fetchable: Fetchable):
        """
        Abstract method which, when called, puts the requested fetchable file on disk.

        :param fetchable: the fetchable
        """
        ...

    @asynccontextmanager
    async def use(self, fetchable: Fetchable) -> Path:
        """
        Async context manager which when entered ensures the given fetchable is on disk and provides
        a path to it. Once the context manager exits the path is volatile and will expire according
        to the TTL (once it is no longer in use by any other coroutines).

        :param fetchable: the fetchable
        :return: the full path to the file
        """
        self.requests += 1
        path = self.root / fetchable.store_path

        if path not in self._in_use:
            if path in self._cleaners:
                self._cleaners.pop(path).cancel()
            elif not path.exists():
                async with self._locker.acquire(path, timeout=self.fetch_timeout):
                    if not path.exists():
                        try:
                            await self._fetch(fetchable)
                        except Exception:
                            self.errors += 1
                            raise
                        self._sizes[path] = path.stat().st_size
                        self.total_size += self._sizes[path]

                        times = 0
                        while self._cleaners and self.total_size > self.max_size and times < 10:
                            path_to_clean_up, handle = self._cleaners.popitem(last=False)
                            handle.cancel()
                            self._clean_up(path_to_clean_up)
                            times += 1

        self._in_use[path] += 1
        try:
            yield path
        finally:
            self._in_use[path] -= 1

            if self._in_use[path] == 0:
                del self._in_use[path]
                self._cleaners[path] = self._schedule_clean_up(path)

            self.completed += 1

    async def get_status(self) -> dict:
        """
        Returns some basic stats about the cache as a dict.

        :return: a dict of stats
        """
        return {
            'requests': self.requests,
            'completed': self.completed,
            'errors': self.errors,
            'cache_size': humanize.naturalsize(self.total_size, binary=True),
            'max_size': humanize.naturalsize(self.max_size, binary=True),
            'percentage_used': self.pct,
            'in_use': len(self._in_use),
            'waiting_clean_up': len(self._cleaners),
        }


def disable_bomb_errors():
    """
    Disables DecompressionBombErrors that are thrown by Pillow when an image we're processing is too
    large.
    Details: https://pillow.readthedocs.io/en/latest/releasenotes/5.0.0.html#decompression-bombs-now-raise-exceptions
    """
    # disable DecompressionBombErrors
    # ()
    Image.MAX_IMAGE_PIXELS = None
