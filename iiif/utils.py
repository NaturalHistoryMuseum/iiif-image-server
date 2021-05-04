#!/usr/bin/env python3
# encoding: utf-8

from asyncio import Future

import asyncio
import humanize
from PIL import Image
from contextlib import contextmanager
from functools import lru_cache
from itertools import count
from lru import LRU
from pathlib import Path
from typing import Callable, Awaitable, Optional, Tuple


class OnceRunner:
    """
    This class runs tasks and caches the results. It ensures that a task is only actually run the
    first time it is submitted and any later calls use the cached result of the first run.
    """

    def __init__(self, name: str, size: int, exception_timeout: Optional[float] = 0):
        """
        :param name: the name of the runner
        :param size: the maximum size of the LRU cache used to store task results
        :param exception_timeout: minimum time in seconds to wait before trying a task again if it
                                  raises an exception. Pass 0 to never try the task again (default).
        """
        self.name = name
        self.results = LRU(size=size)
        self.exception_timeout = exception_timeout
        self.waiting = 0
        self.working = 0
        self.loop = asyncio.get_event_loop()

    async def run(self, task_id: str, task: Callable[..., Awaitable], *args, **kwargs):
        """
        Given a task and it's unique identifier, either run the task if it's the first time we've
        seen it or return the cached result of a previous run. If the task passed is currently
        running then the result is awaited and returned once the first instance of the task
        completes.

        :param task_id: the task's id
        :param task: the task itself, this should be a reference to an async function
        :param args: the args to pass to the async function when running it
        :param kwargs: the kwargs to pass to the async function when running it
        :return: the task's return value
        """
        if task_id in self.results:
            with self.wait():
                # this will either resolve instantaneously because the future in the results cache
                # is done or it will wait for the task that is populating the future to complete
                return await self.results[task_id]

        with self.work():
            self.results[task_id] = Future()
            try:
                result = await task(*args, **kwargs)
                self.results[task_id].set_result(result)
                return result
            except Exception as exception:
                # set the exception on the future so that any future or concurrent requests for the
                # same task get the same exception when they await it
                self.results[task_id].set_exception(exception)

                if self.exception_timeout:
                    self.loop.call_later(self.exception_timeout, self.expire, task_id)

                # raise it for the current caller
                raise exception

    def expire(self, task_id: str) -> bool:
        """
        Force the expiry of the given task by popping it from the cache.

        :param task_id: the id of the task to expire
        :return: True if the task was evicted, False if it was already expired
        """
        return self.results.pop(task_id, None) is not None

    def expire_matching(self, filter_function: Callable[[str], bool]) -> int:
        """
        Expire the task results that match the given filter function. Returns the number of task
        results that were expired.

        :param filter_function: a function that when passed a str task ID returns True if the task
                                result should be removed and False if not
        :return: the number of tasks removed by the filter function
        """
        return sum(map(self.expire, filter(filter_function, list(self.results))))

    async def get_status(self) -> dict:
        """
        Returns a status dict about the state of the runner.

        :return: a dict of details
        """
        return {
            'size': len(self.results),
            'waiting': self.waiting,
            'working': self.working,
        }

    @contextmanager
    def wait(self):
        """
        Convenience context manager that increases and then decreases the waiting count around
        whatever code it wraps.
        """
        self.waiting += 1
        yield
        self.waiting -= 1

    @contextmanager
    def work(self):
        """
        Convenience context manager that increases and then decreases the working count around
        whatever code it wraps.
        """
        self.working += 1
        yield
        self.working -= 1


def parse_size(size: str, width: int, height: int):
    """
    Parses the IIIF size parameter and returns the width and a height requested. Note that this only
    currently supports parsing level 1 compliant size parameters (see
    https://iiif.io/api/image/3.0/compliance/#32-size).

    :param size: the size parameter value
    :param width: the full width of the image
    :param height: the full height of the image
    :return: the requested width and height
    """
    if size == 'max':
        return width, height

    w, h = (float(v) if v != '' else v for v in size.split(','))
    if h == '':
        h = height * w / width
    elif w == '':
        w = width * h / height

    return int(w), int(h)


def get_path_stats(path: Path) -> dict:
    """
    Calculates some statistics about the on disk path's files.

    :param path: the path to investigate
    :return: a dict of stats
    """
    sizes = [f.stat().st_size for f in path.glob('**/*') if f.is_file()]
    size = sum(sizes)
    return {
        'count': len(sizes),
        'size_bytes': size,
        'size_pretty': humanize.naturalsize(size, binary=True),
    }


def get_mss_base_url_path(name: str):
    """
    Given the name of an image in the MSS system (i.e. an EMu IRN) return the URL path part that
    contains the its images in MSS.

    This is directly related to how EMu actually stores the files associated with a multimedia
    record on disk. Through reverse engineering, it has been determined that it stores the files
    under a folder path where two folders are created the second one is the (padded) last 3 digits
    of the IRN and the first one is the everything up to those last 3 digits. If the padded IRN is
    exactly 3 digits then a single 0 is used. Examples:

        - 1 -> 0/001
        - 14 -> 0/014
        - 305 -> 0/305
        - 9217 -> 9/217
        - 2389749823 -> 2389749/823

    :param name: the EMu IRN of the image
    :return: the path part as a string
    """
    padded = str(name).zfill(4)
    return f'{padded[:-3]}/{padded[-3:]}'


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


def convert_image(image_path: Path, target_path: Path, quality: int = 80, subsampling: int = 0):
    """
    Given the path to an image, outputs the image to the target path in jpeg format. This should
    happen to all images that will have processing done on them subsequently as it means we can use
    a common approach to all files - namely using jpegtran on them.

    :param image_path: the path to the source image
    :param target_path: the path to output the jpeg version of the image
    :param quality: the jpeg quality setting to use
    :param subsampling: the jpeg subsampling to use
    """
    image = Image.open(image_path)

    # if it's a jpeg, remove the orientation exif tag. We do this because through trial and error it
    # seems the dimensions provided by EMu are non-orientated and therefore we need to work on the
    # images without their orientation too and serve them up without it otherwise things start to go
    # awry
    if image.format.lower() == 'jpeg':
        exif = image.getexif()
        # this is the orientation tag, remove it if it's there
        exif.pop(0x0112, None)
        image.info['exif'] = exif.tobytes()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(target_path, format='jpeg', quality=quality, subsampling=subsampling)
