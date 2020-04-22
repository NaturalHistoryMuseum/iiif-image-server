#!/usr/bin/env python3
# encoding: utf-8
from collections import defaultdict

import multiprocessing as mp
import os
import random
from jpegtran import JPEGImage
from lru import LRU
from multiprocessing.context import Process
from threading import Thread
from tornado.ioloop import IOLoop
from tornado.locks import Event


def process_image_request(worker_id, task_queue, result_queue, cache_size):
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

    # TODO: handle worker errors properly
    # TODO: jpegtran can't upscale, might want to prevent that from being asked for or use pillow
    #       for just those ops?

    try:
        # wait for tasks until we get a sentinel (in this case None)
        for task in iter(task_queue.get, None):
            if task.image.name not in image_cache:
                # the JPEGImage init function reads the entire source file into memory
                image_cache[task.image.name] = JPEGImage(task.image.source_path)

            image = image_cache[task.image.name]

            if task.region != 'full':
                x, y, w, h = map(int, task.region.split(','))
                image = image.crop(x, y, w, h)

            if task.size != 'max':
                image_width = image.width
                image_height = image.height
                w, h = (float(v) if v != '' else v for v in task.size.split(','))
                if h == '':
                    h = image_height * w / image_width
                elif w == '':
                    w = image_width * h / image_height
                image = image.downscale(int(w), int(h))

            # ensure the full cache path exists
            os.makedirs(os.path.dirname(task.output_path), exist_ok=True)
            # write the processed image to disk
            with open(task.output_path, 'wb') as f:
                f.write(image.as_blob())

            # put our worker id and the task on the result queue to indicate to the main process
            # that it's done
            result_queue.put((worker_id, task))
    except KeyboardInterrupt:
        pass


class Worker:
    """
    Class representing an image processing worker process.
    """

    def __init__(self, worker_id, result_queue, cache_size):
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
        self.process = Process(target=process_image_request, args=(worker_id, self.task_queue,
                                                                   result_queue, cache_size))
        self.queue_size = 0
        # this LRU cache holds the source file paths that should be in the process's image cache at
        # the time the last task on the task queue is processed and therefore allows us to use it as
        # a heuristic when determining which worker to assign a task (we want to hit the image cache
        # as much as possible!)
        self.predicted_cache = LRU(cache_size)
        self.process.start()

    def add(self, task):
        """
        Adds the given task to this worker's task queue.

        :param task: the Task object
        """
        self.queue_size += 1
        self.predicted_cache[task.image.source_path] = True
        # this will almost always be instantaneous but does have the chance to block up the entire
        # asyncio thread
        self.task_queue.put(task)

    def done(self, task):
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

    def is_warm_for(self, task):
        """
        Determines whether the worker is warmed up for a given task. This just checks to see whether
        the source image will be in the worker's LRU cache when it is processed if it is added to
        the queue now.

        :param task: the task
        :return: True if the source path is warm on this worker or False if not
        """
        return task.image.source_path in self.predicted_cache


class Task:
    """
    Class representing an image processing task as defined by a IIIF based request.
    """

    def __init__(self, image, region, size):
        """
        :param image: the Image object to work on
        :param region: the IIIF region request parameter
        :param size: the IIIF size request parameter
        """
        self.image = image
        self.region = region
        self.size = size
        # the output path is formed by using the image's cache path and then each part of the
        # request as a folder in the path
        self.output_path = os.path.join(image.cache_path, region, f'{size}.jpg')


class ImageProcessingDispatcher:
    """
    Class controlling the image processing workers.
    """

    def __init__(self):
        # keep a reference to the correct tornado io loop so that we can correctly call task
        # completion callbacks from the result thread
        self.loop = IOLoop.current()
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
            self.loop.add_callback(self.finish_task, *result)

    def init_workers(self, worker_count, worker_cache_size):
        """
        Initialises the required number of workers.

        :param worker_count: the number of workers to create
        :param worker_cache_size: the size of each worker's image cache
        """
        for i in range(worker_count):
            self.workers[i] = Worker(i, self.result_queue, worker_cache_size)

    def submit(self, task):
        """
        Processes the given task on one of our worker processes. If the task has already been
        completed (this is determined by the existence of the task's output path) then the task will
        not be reprocessed. Tornado Event objects are used to determine if a task has been completed
        or not and should be awaited on. If the task has already been completed a switched on Event
        object will be returned by this function which will immediately return when awaited. If a
        running task is requested again whilst it is being processed, the same Event object will be
        returned by this function for the processing request and the new requests. This results in
        all tasks resolving at the same time upon the first task's completion.

        :param task: the task object
        :return: a tornado Event object to await on
        """
        if task.output_path in self.output_paths:
            # this task has either already been completed prior to this request or is currently
            # being processed, just return the Event object associated with it
            return self.output_paths[task.output_path]

        # we haven't processed this task before, create an event and add it to the output_paths
        processed_event = Event()
        self.output_paths[task.output_path] = processed_event
        if os.path.exists(task.output_path):
            # if the path exists the task was created before this server started up, set it to
            # indicate the task is complete
            processed_event.set()
        else:
            # otherwise, choose a worker and add it to it
            worker = self.choose_worker(task)
            worker.add(task)

        # return the Event object
        return processed_event

    def choose_worker(self, task):
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

    def finish_task(self, worker_id, task):
        """
        Called by the result thread to signal that a task has finished processing. This function
        simply sets the Event object associated with the task's output path.

        :param worker_id: the id of the worker that completed the task
        :param task: the task object that is complete
        """
        self.workers[worker_id].done(task)
        self.output_paths[task.output_path].set()

    def stop(self):
        """
        Signals all worker processes to stop. This function will block until all workers have
        finished processing their queues.
        """
        for worker in self.workers.values():
            worker.stop()
        self.result_queue.put(None)
        self.result_thread.join()
