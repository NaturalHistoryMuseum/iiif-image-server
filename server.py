#!/usr/bin/env python3
# encoding: utf-8

import multiprocessing as mp
import os
import random
import yaml
from PIL import Image
from concurrent.futures.process import ProcessPoolExecutor
from functools import lru_cache
from itertools import count
from jpegtran import JPEGImage
from lru import LRU
from multiprocessing.context import Process
from threading import Thread
from tornado.ioloop import IOLoop
from tornado.locks import Event
from tornado.web import Application, RequestHandler, HTTPError

# disable DecompressionBombErrors
# (https://pillow.readthedocs.io/en/latest/releasenotes/5.0.0.html#decompression-bombs-now-raise-exceptions)
Image.MAX_IMAGE_PIXELS = None


def process_image_request(task_queue, result_queue, cache_size):
    """
    Processes a given task queue, putting tasks on the given results queue once complete. This
    function is blocking and should be run in a separate process.

    Due to the way JPEGImage handles file data we use the LRU cache to avoid rereading source files
    if possible. When initialised, JPEGImage loads the entire source file into memory but is then
    immutable when using the various operation functions (crop, downscale etc). This means it's most
    efficient for us to load the file once and reuse the JPEGImage object over and over again, hence
    the LRU image cache.

    :param task_queue: a multiprocessing Queue of Task objects
    :param result_queue: a multiprocessing Queue to put the completed Task objects on
    :param cache_size: the size to use for the LRU cache for loaded source images
    """
    image_cache = LRU(cache_size)

    try:
        # wait for tasks until we get a sentinel (in this case None)
        for task in iter(task_queue.get, None):
            if task.image_name not in image_cache:
                # the JPEGImage init function reads the entire source file into memory
                image_cache[task.image_name] = JPEGImage(task.source_path)

            image = image_cache[task.image_name]

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
            os.makedirs(os.path.dirname(task.cached_path), exist_ok=True)
            # write the processed image to disk
            with open(task.cached_path, 'wb') as f:
                f.write(image.as_blob())

            # put the task on the result queue to indicate to the main process that it's done
            result_queue.put(task)
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
        self.cache_size = cache_size

        # create a multiprocessing Queue for just this worker's tasks
        self.task_queue = mp.Queue()
        # create the process
        self.process = Process(target=process_image_request, args=(self.task_queue, result_queue,
                                                                   self.cache_size))
        # this LRU cache holds the source file paths that should be in the process's image cache at
        # the time the last task on the task queue is processed and therefore allows us to use it as
        # a heuristic when determining which worker to assign a task (we want to hit the image cache
        # as much as possible!)
        self.predicted_cache = LRU(self.cache_size)
        self.process.start()

    @property
    def queue_size(self):
        return self.task_queue.qsize()

    def add(self, task):
        """
        Adds the given task to this worker's task queue.

        :param task: the Task object
        """
        self.predicted_cache[task.source_path] = True
        # this will almost always be instantaneous but does have the chance to block up the entire
        # asyncio thread
        self.task_queue.put(task)

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
        return task.source_path in self.predicted_cache


class Task:
    """
    Class representing an image processing task as defined by a IIIF based request.
    """

    def __init__(self, source_dir, cache_dir, image_name, region, size):
        """
        :param source_dir: the directory to retrieve the source image for this task from
        :param cache_dir: the directory to cache the image produced by this task in
        :param image_name: the name of the source image
        :param region: the IIIF region request parameter
        :param size: the IIIF size request parameter
        """
        self.image_name = image_name
        self.region = region
        self.size = size
        # the cached path is formed by using each part of the request as a folder in the path
        self.cached_path = os.path.join(cache_dir, image_name, region, f'{size}.jpg')
        self.source_path = os.path.join(source_dir, image_name)


class ImageProcessingDispatcher:
    """
    Class controlling the image processing workers.
    """

    def __init__(self):
        # keep a reference to the correct tornado io loop so that we can correctly call task
        # completion callbacks from the result thread
        self.loop = IOLoop.current()
        # a list of the Worker objects we're dispatching the requests to
        self.workers = []
        # a register of the processed image paths and tornado Event objects indicating whether they
        # have been processed yet, we deliberately don't pre-populate this in case the cache
        # directory is enormous and leave it to be lazily built as requests come in (see process)
        self.cached_paths = {}
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
        for task in iter(self.result_queue.get, None):
            self.loop.add_callback(self.finish_task, task)

    def init_workers(self, worker_count, worker_cache_size):
        """
        Initialises the required number of workers.

        :param worker_count: the number of workers to create
        :param worker_cache_size: the size of each worker's image cache
        """
        for i in range(worker_count):
            self.workers.append(Worker(i, self.result_queue, worker_cache_size))

    def process(self, task):
        """
        Processes the given task on one of our worker processes. If the task has already been
        completed (this is determined by the existence of the task's cached path) then the task will
        not be reprocessed. Tornado Event objects are used to determine if a task has been completed
        or not and should be awaited. If the task has already been completed a switched on Event
        object will be returned by this function which will immediately return when awaited. If a
        running task is requested again whilst it is being processed, the same Event object will be
        returned by this function for the processing request and the new requests. This results in
        all tasks resolving at the same time upon the first task's completion.

        :param task: the task object
        :return: a tornado Event object to await on
        """
        if task.cached_path in self.cached_paths:
            # this task has either already been completed prior to this request or is currently
            # being processed, just return the Event object associated with it
            return self.cached_paths[task.cached_path]

        # we haven't processed this task before, create an event and add it to the cached_paths
        loaded_event = Event()
        self.cached_paths[task.cached_path] = loaded_event
        if os.path.exists(task.cached_path):
            # if the path exists the task was created before this server started up, set it to
            # indicate the task is complete
            loaded_event.set()
        else:
            # otherwise, choose a worker and add it to it
            worker = self.choose_worker(task)
            worker.add(task)

        # return the Event object
        return loaded_event

    def choose_worker(self, task):
        """
        Select a worker for the given task.

        Workers which will have the source image loaded into their image caches are prioritised. If
        no worker is found that meets this criteria.

        :param task:
        :return:
        """
        # select a random worker to start with
        selected_worker = random.choice(self.workers)
        for worker in self.workers:
            # TODO: take queue size into consideration rather than just choosing the first warm one
            # if we find a warm worker, choose them
            if worker.is_warm_for(task):
                return worker
            # if this worker isn't warm, compare it's queue size so that if no workers are warm we
            # send the task to the worker with the smallest queue
            if selected_worker.queue_size < worker.queue_size:
                selected_worker = worker
        return selected_worker

    def finish_task(self, task):
        """
        Called by the result thread to signal that a task has finished processing. This function
        simply sets the cached path's Event object.

        :param task: the task object that is complete
        """
        self.cached_paths[task.cached_path].set()

    def stop(self):
        """
        Signals all worker processes to stop. This function will block until all workers have
        finished processing their queues.
        """
        for worker in self.workers:
            worker.stop()
        self.result_queue.put(None)
        self.result_thread.join()


class ImageDataHandler(RequestHandler):
    """
    Request handler for image data requests.
    """

    def initialize(self, config, dispatcher):
        """
        Inits the handler with the global config and dispatcher objects.

        :param config: the config dict
        :param dispatcher: the ImageProcessingDispatcher instance
        """
        self.config = config
        self.dispatcher = dispatcher

    async def get(self, image_name, region, size):
        """
        Responds to IIIF image data get requests.

        :param image_name: the image name (identifier)
        :param region: the requested region
        :param size: the requested size
        """
        task = Task(self.config['source_dir'], self.config['cache_dir'], image_name, region, size)
        await self.dispatcher.process(task).wait()

        self.set_header("Content-type", "image/jpeg")
        with open(task.cached_path, 'rb') as f:
            while True:
                # read the data in chunks of 64KiB
                data = f.read(65536)
                if not data:
                    break
                self.write(data)
                # flush to avoid reading the whole image into memory at once
                await self.flush()
        await self.finish()


class ImageInfoHandler(RequestHandler):
    """
    Request handler for info.json requests.
    """

    def initialize(self, config, info_cache, info_pool):
        """
        Inits the handler with the global config, info cache and info process pool instances.

        :param config: the config dict
        :param info_cache: LRU cache instance to store info.json responses in
        :param info_pool: process pool for retrieving source image sizes
        """
        self.config = config
        self.info_cache = info_cache
        self.info_pool = info_pool

    @staticmethod
    def get_image_size(source_dir, image_name):
        """
        Function that retrieves the width and height (as a tuple) of the given source image. This
        function should be fast enough to run in the main asyncio event loop, however, to be sure it
        doesn't block up the server we run it in a process pool.

        :param source_dir: the source image directory path
        :param image_name: the name of the source image
        :return: the width and height as a tuple
        """
        with Image.open(os.path.join(source_dir, image_name)) as image:
            return image.width, image.height

    @staticmethod
    @lru_cache(maxsize=1024)
    def generate_sizes(width, height, min_sizes_size=200):
        """
        Produces the sizes array for the given width and height combination. Function results are
        cached for speed.

        :param width: the width of the source image
        :param height: the height of the source image
        :param min_sizes_size: the minimum dimension size to include in the returned list
        :return: a list of sizes in descending order
        """
        # always include the original image size in the sizes list
        sizes = [{'width': width, 'height': height}]
        for i in count(1):
            factor = 2 ** i
            new_width = width // factor
            new_height = height // factor
            # stop when either dimension is smaller than
            if new_width < min_sizes_size or new_height < min_sizes_size:
                break
            sizes.append({'width': new_width, 'height': new_height})

        return sizes

    async def get(self, image_name):
        """
        Responds to IIIF info.json get requests.

        :param image_name: the image name (identifier)
        """
        if image_name not in self.info_cache:
            # the image's info.json is not in the cache, generate it and store it
            try:
                # use the process pool to extract the width and height of the source image
                width, height = await IOLoop.current().run_in_executor(self.info_pool,
                                                                       self.get_image_size,
                                                                       self.config['source_dir'],
                                                                       image_name)

                # add the complete info.json to the cache
                self.info_cache[image_name] = {
                    '@context': 'http://iiif.io/api/image/3/context.json',
                    # mirador/openseadragon seems to need this to work even though I don't think
                    # it's correct under the IIIF image API v3
                    '@id': f'{self.config["base_url"]}/{image_name}',
                    'id': f'{self.config["base_url"]}/{image_name}',
                    'type': 'ImageService3',
                    'protocol': 'http://iiif.io/api/image',
                    'width': width,
                    'height': height,
                    'rights': 'http://creativecommons.org/licenses/by/4.0/',
                    'profile': 'level0',
                    'tiles': [
                        {'width': 512, 'scaleFactors': [1, 2, 4, 8, 16]},
                        {'width': 256, 'scaleFactors': [1, 2, 4, 8, 16]},
                        {'width': 1024, 'scaleFactors': [1, 2, 4, 8, 16]},
                    ],
                    'sizes': self.generate_sizes(width, height, self.config['min_sizes_size']),
                }
            except FileNotFoundError:
                raise HTTPError(status_code=404, reason="Image not found")

        # serve up the info.json (tornado automatically writes a dict out as JSON with headers etc)
        await self.finish(self.info_cache[image_name])


def main():
    """
    Main entry function for the server.
    """
    # load the config file, it should be next to this script
    root_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(root_dir, 'config.yml'), 'r') as cf:
        config = yaml.safe_load(cf)

    # initialise a process pool to get source image dimensions
    info_pool = ProcessPoolExecutor(max_workers=config['info_pool_size'])
    # create an LRU cache to keep the most recent info.json request responses in
    info_cache = LRU(config['info_cache_size'])
    # create the dispatcher which controls how image data requests are handled
    dispatcher = ImageProcessingDispatcher()

    try:
        # initialise the process pool that backs the dispatcher
        dispatcher.init_workers(config['image_pool_size'], config['image_cache_size_per_process'])

        # setup the tornado app
        app = Application([
            (r'/(?P<image_name>.+)/info.json', ImageInfoHandler,
             dict(config=config, info_cache=info_cache, info_pool=info_pool)),
            (r'/(?P<image_name>.+)/(?P<region>.+)/(?P<size>.+)/0/default.jpg', ImageDataHandler,
             dict(config=config, dispatcher=dispatcher)),
        ])
        app.listen(config['http_port'])
        print(f'Listening on {config["http_port"]}, our pid is {os.getpid()}')
        IOLoop.current().start()
    except KeyboardInterrupt:
        print('Shutdown request received')
    finally:
        info_pool.shutdown()
        dispatcher.stop()
        print('Shutdown complete')


if __name__ == '__main__':
    main()
