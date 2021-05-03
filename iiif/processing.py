#!/usr/bin/env python3
# encoding: utf-8
from asyncio import Future
from collections import defaultdict, namedtuple

import asyncio
import io
import multiprocessing as mp
import random
from PIL import Image
from fastapi import HTTPException
from jpegtran import JPEGImage
from lru import LRU
from multiprocessing.context import Process
from pathlib import Path
from threading import Thread
from typing import Any, Optional

from iiif.utils import parse_size

Crop = namedtuple('Crop', ('x', 'y', 'w', 'h'))


class Task:
    """
    Class representing an image processing task as defined by a IIIF based request.
    """

    def __init__(self, source_path: Path, output_path: Path, region: str, size: str):
        """
        :param region: the IIIF region request parameter
        :param size: the IIIF size request parameter
        """
        self.source_path = source_path
        self.output_path = output_path
        self.region = region
        self.size = size

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.source_path == other.source_path and self.region == other.region and \
                   self.size == other.size
        return NotImplemented


def process_region(image: JPEGImage, region: str) -> JPEGImage:
    """
    Processes a IIIF region parameter which essentially involves cropping the image.

    :param image: a jpegtran JPEGImage object
    :param region: the IIIF region parameter
    :return: a jpegtran JPEGImage object
    """
    if region == 'full':
        # no crop required!
        return image

    # cache these dimensions as jpegtran has to work to get them
    width = image.width
    height = image.height

    if region == 'square':
        if width < height:
            # the image is portrait, we need to crop out a centre square the size of the width
            crop = Crop(0, int((height - width) / 2), width, width)
        elif width > height:
            # the image is landscape, we need to crop out a centre square the size of the height
            crop = Crop(int((width - height) / 2), 0, height, height)
        else:
            # the image is already square, return the whole thing
            return image
    else:
        crop = Crop(*map(int, region.split(',')))

    # jpegtran can't handle crops that don't have an origin divisible by 16 therefore we're going to
    # do the crop in pillow, however, we're going to crop down to the next lowest number below each
    # of x and y that is divisible by 16 using jpegtran and then crop off the remaining pixels in
    # pillow to get to the desired size. This is all for performance, jpegtran is so much quicker
    # than pillow hence it's worth this hassle
    if crop.x % 16 or crop.y % 16:
        # work out how far we need to shift the x and y to get to the next lowest numbers divisible
        # by 16
        x_shift = crop.x % 16
        y_shift = crop.y % 16
        # crop the image using the shifted x and y and the shifted width and height
        image = image.crop(crop.x - x_shift, crop.y - y_shift, crop.w + x_shift, crop.h + y_shift)
        # convert the jpegtran image object to a pillow image object
        pillow_image = Image.open(io.BytesIO(image.as_blob()))
        # do the final crop to get us the desired size
        pillow_image = pillow_image.crop((x_shift, y_shift, x_shift + crop.w, y_shift + crop.h))
        # write the image to memory
        output = io.BytesIO()
        pillow_image.save(output, format='jpeg')
        output.seek(0)
        # read the image written out by pillow back in as a jpegtran JPEGImage object and return
        return JPEGImage(blob=output.read())
    else:
        # if the crop has an origin divisible by 16 then we can just use jpegtran directly
        return image.crop(*crop)


def process_size(image: JPEGImage, size: str) -> JPEGImage:
    """
    Processes a IIIF size parameter which essentially involves resizing the image.

    :param image: a jpegtran JPEGImage object
    :param size: the IIIF size parameter
    :return: a jpegtran JPEGImage object
    """
    # cache these dimensions as jpegtran has to work to get them
    width, height = image.width, image.height

    target_width, target_height = parse_size(size, width, height)

    if size == 'max' or (target_width == width and target_height == height):
        return image

    if width < target_width or height < target_height:
        raise HTTPException(400, detail='Size greater than extracted region without specifying ^')

    return image.downscale(target_width, target_height)


def process_image_requests(worker_id: Any, task_queue: mp.Queue, result_queue: mp.Queue,
                           cache_size: int):
    """
    Processes a given task queue, putting tasks on the given results queue once complete. This
    function is blocking and should be run in a separate process.

    Due to the way JPEGImage handles file data we use the LRU cache to avoid rereading source files
    if possible. When initialised, JPEGImage loads the entire source file into memory but is then
    immutable when using the various operation functions (crop, downscale etc). This means it's most
    efficient for us to load the file once and reuse the JPEGImage object over and over again, hence
    the LRU image cache.

    :param worker_id: the worker id associated with this process
    :param task_queue: a multiprocessing Queue of Task objects
    :param result_queue: a multiprocessing Queue to put the completed Task objects on
    :param cache_size: the size to use for the LRU cache for loaded source images
    """
    image_cache = LRU(cache_size)

    try:
        # wait for tasks until we get a sentinel (in this case None)
        for task in iter(task_queue.get, None):
            try:
                if task.source_path not in image_cache:
                    # the JPEGImage init function reads the entire source file into memory
                    image_cache[task.source_path] = JPEGImage(task.source_path)

                image = image_cache[task.source_path]

                image = process_region(image, task.region)
                image = process_size(image, task.size)

                # ensure the full cache path exists
                task.output_path.parent.mkdir(parents=True, exist_ok=True)
                # write the processed image to disk
                with task.output_path.open('wb') as f:
                    f.write(image.as_blob())

                # put our worker id, the task and None on the result queue to indicate to the main
                # process that it's done and we encountered no exceptions
                result_queue.put((worker_id, task, None))
            except Exception as e:
                # if we get a keyboard interrupt we need to stop!
                if isinstance(e, KeyboardInterrupt):
                    raise e
                # put our worker id, the task and the exception on the result queue to indicate to
                # the main process that it's done and we encountered an exception
                result_queue.put((worker_id, task, e))
    except KeyboardInterrupt:
        pass


class Worker:
    """
    Class representing an image processing worker process.
    """

    def __init__(self, worker_id: Any, result_queue: mp.Queue, cache_size: int):
        """
        :param worker_id: the worker's id, handy for debugging and not really used otherwise
        :param result_queue: the multiprocessing Queue that should be used by the worker to indicate
                             task completions
        :param cache_size: the requested size of the worker's image cache
        """
        self.worker_id = worker_id

        # create a multiprocessing Queue for just this worker's tasks
        self.task_queue = mp.Queue()
        # create the process
        self.process = Process(target=process_image_requests, args=(worker_id, self.task_queue,
                                                                    result_queue, cache_size))
        self.queue_size = 0
        # this LRU cache holds the source file paths that should be in the process's image cache at
        # the time the last task on the task queue is processed and therefore allows us to use it as
        # a heuristic when determining which worker to assign a task (we want to hit the image cache
        # as much as possible!)
        self.predicted_cache = LRU(cache_size)
        self.process.start()

    def add(self, task: Task):
        """
        Adds the given task to this worker's task queue.

        :param task: the Task object
        """
        self.queue_size += 1
        self.predicted_cache[task.source_path] = True
        # this will almost always be instantaneous but does have the chance to block up the entire
        # asyncio thread
        self.task_queue.put(task)

    def done(self, task: Task):
        """
        Call this to notify this worker that it completed the given task.

        :param task: the task that was completed
        """
        self.queue_size -= 1

    def stop(self):
        """
        Requests that this worker stops. This is a blocking call and will wait until the worker has
        completed all currently queued tasks.
        """
        # send the sentinel
        self.task_queue.put(None)
        self.process.join()

    def is_warm_for(self, task: Task) -> bool:
        """
        Determines whether the worker is warmed up for a given task. This just checks to see whether
        the source image will be in the worker's LRU cache when it is processed if it is added to
        the queue now.

        :param task: the task
        :return: True if the source path is warm on this worker or False if not
        """
        return task.source_path in self.predicted_cache


class ImageProcessingDispatcher:
    """
    Class controlling the image processing workers.
    """

    def __init__(self):
        # keep a reference to the correct asyncio loop so that we can correctly call task completion
        # callbacks from the result thread
        self.loop = asyncio.get_event_loop()
        # a dict of the Worker objects we're dispatching the requests to keyed by their worker ids
        self.workers = {}
        # a register of the processed image paths and tornado Event objects indicating whether they
        # have been processed yet, we deliberately don't pre-populate this in case the cache
        # directory is large and leave it to be lazily built as requests come in (see submit method)
        self.output_paths = {}
        # the multiprocessing result queue used by all workers to notify the main process that a
        # task has been completed
        self.result_queue = mp.Queue()
        self.result_thread = Thread(target=self.result_listener)
        self.result_thread.start()

    def result_listener(self):
        """
        This function should be run in a separate thread to avoid blocking the asyncio loop. It
        listens for results to be put on the result queue by workers and adds a callback into the
        main ayncio loop to notify all waiting coroutines.
        """
        for result in iter(self.result_queue.get, None):
            self.loop.call_soon_threadsafe(self.finish_task, *result)

    def init_workers(self, worker_count: int, worker_cache_size: int):
        """
        Initialises the required number of workers.

        :param worker_count: the number of workers to create
        :param worker_cache_size: the size of each worker's image cache
        """
        for i in range(worker_count):
            self.workers[i] = Worker(i, self.result_queue, worker_cache_size)

    async def submit(self, task: Task):
        """
        Submits the given task to be processed on one of our worker processes. If the task has
        already been completed (this is determined by the existence of the task's output path) then
        the task will not be reprocessed. Tornado Future objects are used to determine if a task has
        been completed or not. If the task has already been completed, this function will return
        immediately when awaited. If a task is requested again whilst it is already being processed,
        the Future object created for the in progress task will be awaited on by this function for
        the new processing request. This results in all tasks resolving at the same time upon the
        first task's completion.

        :param task: the task object
        """
        if task.output_path not in self.output_paths:
            # we haven't processed this task before, create a Future and add it to the output_paths
            processed_future = Future()
            self.output_paths[task.output_path] = processed_future
            if task.output_path.exists():
                # if the path exists the task was created before this server started up, set the
                # result to indicate the task is complete
                processed_future.set_result(None)
            else:
                # otherwise, choose a worker and add it to it
                worker = self.choose_worker(task)
                worker.add(task)

        await self.output_paths[task.output_path]

    def choose_worker(self, task: Task) -> Worker:
        """
        Select a worker for the given task. Workers are chosen by giving them a score and then
        randomly choosing the worker from the group with highest score.

        Workers which will have the source image loaded into their image caches are prioritised as
        are workers with a queue size shorter than the number of workers (for lack of a better value
        to be less than).

        :param task: the task
        :return: a Worker object
        """
        buckets = defaultdict(list)
        for worker in self.workers.values():
            # higher is better
            score = 0

            if worker.queue_size <= len(self.workers):
                # you get a point if your queue is smaller than the current number of workers
                score += 1
                if worker.queue_size == 0:
                    # and an extra point if you have no tasks on your queue
                    score += 1

            if worker.is_warm_for(task):
                # you get a point if you are warmed up for the task
                score += 1

            # add the worker to the appropriate bucket
            buckets[score].append(worker)

        # choose the bucket with the highest score and pick a worker at random from it
        return random.choice(buckets[max(buckets.keys())])

    def finish_task(self, worker_id: Any, task: Task, exception: Optional[Exception]):
        """
        Called by the result thread to signal that a task has finished processing. This function
        simply retrieves the the Future object associated with the task's output path and sets its
        result/exception.

        :param worker_id: the id of the worker that completed the task
        :param task: the task object that is complete
        :param exception: an exception that occurred during processing, if no exception was
                          encountered this will be None
        """
        self.workers[worker_id].done(task)
        if exception is None:
            self.output_paths[task.output_path].set_result(None)
        else:
            self.output_paths[task.output_path].set_exception(exception)

    def stop(self):
        """
        Signals all worker processes to stop. This function will block until all workers have
        finished processing their queues.
        """
        for worker in self.workers.values():
            worker.stop()
        self.result_queue.put(None)
        self.result_thread.join()

    def get_status(self) -> dict:
        """
        Returns some basic stats info as a dict.

        :return: a dict of stats
        """
        return {
            'results_queue_size': self.result_queue.qsize(),
            'worker_queue_sizes': {
                worker.worker_id: worker.queue_size for worker in self.workers.values()
            }
        }
