#!/usr/bin/env python3
# encoding: utf-8

from asyncio import Future

import hashlib
import io
import multiprocessing as mp
import os
import pytest
from PIL import Image
from fastapi import HTTPException
from queue import Queue
from unittest.mock import patch, MagicMock, call

from iiif.ops import parse_params
from iiif.processing import Task, process_image_requests, Worker, ImageProcessingDispatcher
from iiif.profiles.base import ImageInfo
from tests.utils import create_image

default_image_width = 4000
default_image_height = 5000


@pytest.fixture
def source_path(config):
    return create_image(config, default_image_width, default_image_height)


@pytest.fixture
def cache_path(config):
    return config.cache_path / 'test' / 'image'


@pytest.fixture
def info():
    return ImageInfo('test_profile', 'test_image', default_image_width, default_image_height)


@pytest.fixture
def task_queue():
    # a real queue but not a multiprocessing one
    return Queue()


@pytest.fixture
def result_queue():
    # a real queue but not a multiprocessing one
    return Queue()


def check_size(task, width, height):
    with Image.open(task.output_path) as image:
        assert image.width == width
        assert image.height == height


def check_result(task: Task, op_function):
    with Image.open(task.source_path) as img:
        cropped_source = io.BytesIO()
        img = op_function(img)
        img.save(cropped_source, format='jpeg')
        cropped_source.seek(0)

        with open(task.output_path, 'rb') as f:
            assert (hashlib.sha256(f.read()).digest() ==
                    hashlib.sha256(cropped_source.read()).digest())


class TestProcessImageRequestsLevel0:
    """
    Test the process_image_requests function for IIIF Image API v3 level 0 compliance.
    See: https://iiif.io/api/image/3.0/compliance/.

    Note that we implicitly don't support rotations other than 0, quality other than and formats
    other than jpg and therefore we don't need to test for them.
    """

    def test(self, source_path, cache_path, info, task_queue, result_queue):
        # this is all that is expected at level 0
        task = Task(source_path, cache_path, parse_params(info))
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert task.output_path.exists()
        check_size(task, default_image_width, default_image_height)
        check_result(task, lambda img: img)


class TestProcessImageRequestsLevel1:
    """
    Test the process_image_requests function for IIIF Image API v3 level 1 compliance.
    See: https://iiif.io/api/image/3.0/compliance/.
    """

    def test_regionByPx_jpegtran(self, source_path, cache_path, info, task_queue, result_queue):
        x, y, w, h = 0, 0, 1024, 1024
        task = Task(source_path, cache_path, parse_params(info, f'{x},{y},{w},{h}'))
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, 1024, 1024)
        check_result(task, lambda img: img.crop((x, y, x + w, y + h)))

    def test_regionByPx_any(self, source_path, cache_path, info, task_queue, result_queue):
        x, y, w, h = 6, 191, 1002, 1053
        task = Task(source_path, cache_path, parse_params(info, f'{x},{y},{w},{h}'))
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, 1002, 1053)
        check_result(task, lambda img: img.crop((x, y, x + w, y + h)))

    def test_regionSquare_a_square(self, config, cache_path, task_queue, result_queue):
        width = 500
        height = 500
        source = create_image(config, width, height)
        info = ImageInfo('test_profile', 'test_image', width, height)
        task = Task(source, cache_path, parse_params(info, 'square'))
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, height)
        check_result(task, lambda img: img)

    def test_regionSquare_a_portrait_jpegtran(self, config, cache_path, task_queue, result_queue):
        width = 512
        height = 768
        source = create_image(config, width, height)
        info = ImageInfo('test_profile', 'test_image', width, height)
        task = Task(source, cache_path, parse_params(info, 'square'))
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, width)
        check_result(task, lambda img: img.crop((0, 128, 512, 640)))

    def test_regionSquare_a_landscape_jpegtran(self, config, cache_path, task_queue, result_queue):
        width = 768
        height = 512
        source = create_image(config, width, height)
        info = ImageInfo('test_profile', 'test_image', width, height)
        task = Task(source, cache_path, parse_params(info, 'square'))
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, height, height)
        check_result(task, lambda img: img.crop((128, 0, 640, 512)))

    def test_regionSquare_a_portrait_any(self, config, cache_path, task_queue, result_queue):
        width = 500
        height = 700
        source = create_image(config, width, height)
        info = ImageInfo('test_profile', 'test_image', width, height)
        task = Task(source, cache_path, parse_params(info, 'square'))
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, width)
        check_result(task, lambda img: img.crop((0, 100, 500, 600)))

    def test_regionSquare_a_landscape_any(self, config, cache_path, task_queue, result_queue):
        width = 700
        height = 500
        source = create_image(config, width, height)
        info = ImageInfo('test_profile', 'test_image', width, height)
        task = Task(source, cache_path, parse_params(info, 'square'))
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, height, height)
        check_result(task, lambda img: img.crop((100, 0, 600, 500)))

    def test_sizeByW(self, source_path, cache_path, info, task_queue, result_queue):
        width = 512
        expected_height = int(default_image_height * width / default_image_width)
        task = Task(source_path, cache_path, parse_params(info, size=f'{width},'))
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, expected_height)
        check_result(task, lambda img: img.resize((width, expected_height)))

    def test_sizeByH(self, source_path, cache_path, info, task_queue, result_queue):
        height = 512
        expected_width = int(default_image_width * height / default_image_height)
        task = Task(source_path, cache_path, parse_params(info, size=f',{height}'))
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, expected_width, height)
        check_result(task, lambda img: img.resize((expected_width, height)))

    def test_sizeByWh(self, source_path, cache_path, info, task_queue, result_queue):
        width = 400
        height = 600
        task = Task(source_path, cache_path, parse_params(info, size=f'{width},{height}'))
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, height)
        check_result(task, lambda img: img.resize((width, height)))


class TestProcessImageRequestsMisc:

    def test_region_and_size_precise(self, source_path, cache_path, info, task_queue, result_queue):
        # simple check to make sure region and size play nicely together
        task = Task(source_path, cache_path, parse_params(info, '100,200,600,891', '256,401'))
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, 256, 401)
        check_result(task, lambda img: img.crop((100, 200, 600, 891)).resize((256, 401)))

    def test_region_and_size_inferred(self, config, cache_path, task_queue, result_queue):
        # check to make sure region and size play nicely when they're both relying on image
        # dimension ratios
        width = 500
        height = 700
        source = create_image(config, width, height)
        info = ImageInfo('test_profile', 'test_image', width, height)
        task = Task(source, cache_path, parse_params(info, 'square', ',400'))
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, 400, 400)
        check_result(task, lambda img: img.crop((0, 100, 500, 600)).resize((400, 400)))

    def test_upscale_errors_when_not_specified(self, source_path, cache_path, info, task_queue,
                                               result_queue):
        task = Task(source_path, cache_path, parse_params(info, '0,0,400,400', '500,500'))
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        worker_id, result_task, exception = result_queue.get()
        assert worker_id == 0
        assert result_task == task
        assert isinstance(exception, HTTPException)
        assert exception.status_code == 400
        assert exception.detail == 'Size greater than extracted region without specifying ^'


class TestWorker:

    @pytest.fixture
    def worker(self, result_queue):
        worker = Worker(0, result_queue, 1)
        worker.stop()
        return worker

    @pytest.fixture
    def task(self, source_path, cache_path, info):
        return Task(source_path, cache_path, parse_params(info))

    @pytest.fixture
    def result_queue(self):
        return mp.Queue()

    def test_add(self, worker, task):
        assert worker.queue_size == 0

        worker.add(task)

        assert worker.queue_size == 1
        assert task.source_path in worker.predicted_cache
        assert worker.task_queue.qsize() == 1

    def test_done(self, worker, task):
        assert worker.queue_size == 0
        worker.add(task)
        assert worker.queue_size == 1
        worker.done(task)
        assert worker.queue_size == 0

    def test_stop(self, result_queue):
        worker = Worker(0, result_queue, 1)
        worker.stop()
        assert worker.task_queue.qsize() == 0
        assert not worker.process.is_alive()

    def test_is_warm_for(self, config, result_queue):
        worker = Worker(0, result_queue, 2)

        image1_source = create_image(config, 100, 200, profile='test', name='image1')
        info1 = ImageInfo('t1', 't1', 100, 200)
        image2_source = create_image(config, 100, 200, profile='test', name='image2')
        info2 = ImageInfo('t1', 't2', 100, 200)
        image3_source = create_image(config, 100, 200, profile='test', name='image3')
        info3 = ImageInfo('t1', 't3', 100, 200)
        image4_source = create_image(config, 100, 200, profile='test', name='image4')
        info4 = ImageInfo('t1', 't4', 100, 200)
        image1_cache = config.cache_path / 'test' / 'image1'
        image2_cache = config.cache_path / 'test' / 'image2'
        image3_cache = config.cache_path / 'test' / 'image3'
        image4_cache = config.cache_path / 'test' / 'image4'

        task1 = Task(image1_source, image1_cache, parse_params(info1))
        task2 = Task(image2_source, image2_cache, parse_params(info2))
        task3 = Task(image3_source, image3_cache, parse_params(info3))
        task4 = Task(image4_source, image4_cache, parse_params(info4))

        worker.add(task1)
        assert any(worker.is_warm_for(task) for task in (task1,))
        assert not any(worker.is_warm_for(task) for task in (task2, task3, task4))

        worker.add(task2)
        assert any(worker.is_warm_for(task) for task in (task1, task2))
        assert not any(worker.is_warm_for(task) for task in (task3, task4))

        worker.add(task3)
        assert any(worker.is_warm_for(task) for task in (task2, task3))
        assert not any(worker.is_warm_for(task) for task in (task1, task4))

        worker.add(task4)
        assert any(worker.is_warm_for(task) for task in (task3, task4))
        assert not any(worker.is_warm_for(task) for task in (task1, task2))

        worker.stop()

    def test_result_queue_is_used(self, config, cache_path, result_queue):
        worker = Worker(0, result_queue, 2)
        source_path = create_image(config, 100, 200)
        info = ImageInfo('t1', 't1', 100, 200)
        task = Task(source_path, cache_path, parse_params(info))

        worker.add(task)
        worker.stop()

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)


class TestTask:

    def test_equality(self, config):
        image1_source = create_image(config, 100, 200, profile='test', name='image1')
        image2_source = create_image(config, 100, 200, profile='test', name='image2')
        image1_cache = config.cache_path / 'test' / 'image1'
        image2_cache = config.cache_path / 'test' / 'image2'

        assert Task(image1_source, image1_cache, 'full', 'max') == Task(image1_source, image1_cache,
                                                                        'full', 'max')
        assert Task(image1_source, image1_cache, 'full', 'max') != Task(image2_source, image2_cache,
                                                                        'full', 'max')
        assert Task(image2_source, image2_cache, 'full', 'max') != Task(image2_source, image2_cache,
                                                                        'full', '256,')
        assert Task(image2_source, image2_cache, 'square', 'max') != Task(image2_source,
                                                                          image2_cache, 'full',
                                                                          'max')
        # semantically these are the same but we don't recognise it
        assert Task(image1_source, image1_cache, 'full', 'max') != Task(image1_source, image1_cache,
                                                                        '0,0,100,200', 'max')
        assert Task(image1_source, image1_cache, 'full', 'max') != Task(image1_source, image1_cache,
                                                                        'full', '100,200')


@pytest.fixture
def dispatcher(event_loop):
    with patch('iiif.processing.asyncio.get_event_loop', MagicMock(return_value=event_loop)):
        dispatcher = ImageProcessingDispatcher()
        yield dispatcher
        dispatcher.stop()


class TestImageProcessingDispatcher:

    def test_result_listener(self):
        mock_loop = MagicMock()
        with patch('iiif.processing.asyncio.get_event_loop', MagicMock(return_value=mock_loop)):
            dispatcher = ImageProcessingDispatcher()
            mock_result = ('some', 'mock', 'data')
            dispatcher.result_queue.put(mock_result)
            dispatcher.result_queue.put(None)
            dispatcher.result_thread.join()

        mock_loop.call_soon_threadsafe.assert_called_once_with(dispatcher.finish_task, *mock_result)

    def test_init_workers(self, dispatcher):
        worker_mock = MagicMock()
        with patch('iiif.processing.Worker', worker_mock):
            dispatcher.init_workers(2, 3)

        worker_mock.assert_has_calls([
            call(0, dispatcher.result_queue, 3),
            call(1, dispatcher.result_queue, 3)
        ])
        assert len(dispatcher.workers) == 2

    def test_stop(self):
        dispatcher = ImageProcessingDispatcher()
        dispatcher.init_workers(2, 3)

        worker0 = dispatcher.workers[0]
        worker1 = dispatcher.workers[1]

        with patch.object(worker0, 'stop', wraps=worker0.stop) as stop0:
            with patch.object(worker1, 'stop', wraps=worker1.stop) as stop1:
                dispatcher.stop()
                stop0.assert_called_once()
                stop1.assert_called_once()

        assert dispatcher.result_queue.empty()
        assert not dispatcher.result_thread.is_alive()

    def test_finish_task_success(self, dispatcher, config, cache_path):
        worker_id = 0
        source = create_image(config, 100, 200)
        task = Task(source, cache_path, 'full', 'max')
        mock_future = MagicMock()
        mock_worker = MagicMock()
        dispatcher.workers[worker_id] = mock_worker
        dispatcher.output_paths[task.output_path] = mock_future

        dispatcher.finish_task(worker_id, task, None)

        mock_worker.done.assert_called_once_with(task)
        mock_future.set_result.called_once_with(None)
        mock_future.set_exception.assert_not_called()

    def test_finish_task_failure(self, dispatcher, config, cache_path):
        worker_id = 0
        source = create_image(config, 100, 200)
        task = Task(source, cache_path, 'full', 'max')
        mock_future = MagicMock()
        mock_worker = MagicMock()
        mock_exception = MagicMock()
        dispatcher.workers[worker_id] = mock_worker
        dispatcher.output_paths[task.output_path] = mock_future

        dispatcher.finish_task(worker_id, task, mock_exception)

        mock_worker.done.assert_called_once_with(task)
        mock_future.set_exception.called_once_with(mock_exception)
        mock_future.set_result.assert_not_called()

    def test_choose_worker_all_empty(self, dispatcher):
        worker0 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=False))
        worker1 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=False))
        worker2 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=False))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        dispatcher.workers[2] = worker2

        mock_choice = MagicMock()
        with patch('iiif.processing.random', MagicMock(choice=mock_choice)):
            dispatcher.choose_worker(MagicMock())

        mock_choice.assert_called_once_with([worker0, worker1, worker2])

    def test_choose_worker_all_really_full(self, dispatcher):
        worker0 = MagicMock(queue_size=100, is_warm_for=MagicMock(return_value=False))
        worker1 = MagicMock(queue_size=100, is_warm_for=MagicMock(return_value=False))
        worker2 = MagicMock(queue_size=100, is_warm_for=MagicMock(return_value=False))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        dispatcher.workers[2] = worker2

        mock_choice = MagicMock()
        with patch('iiif.processing.random', MagicMock(choice=mock_choice)):
            dispatcher.choose_worker(MagicMock())

        mock_choice.assert_called_once_with([worker0, worker1, worker2])

    def test_choose_worker_some_warm(self, dispatcher):
        worker0 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=False))
        worker1 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=True))
        worker2 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=True))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        dispatcher.workers[2] = worker2

        mock_choice = MagicMock()
        with patch('iiif.processing.random', MagicMock(choice=mock_choice)):
            dispatcher.choose_worker(MagicMock())

        mock_choice.assert_called_once_with([worker1, worker2])

    def test_choose_worker_scenario_1(self, dispatcher):
        """
        Worker states:
            - 0: empty, not warm
            - 1: empty, warm

        Worker 1 should be chosen as warm and empty is better than not warm and empty.
        """
        worker0 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=False))
        worker1 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=True))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        assert worker1 == dispatcher.choose_worker(MagicMock())

    def test_choose_worker_scenario_2(self, dispatcher):
        """
        Worker states:
            - 0: full, not warm
            - 1: full, warm

        Worker 1 should be chosen as warm and full is better than not warm and full.
        """
        worker0 = MagicMock(queue_size=3, is_warm_for=MagicMock(return_value=False))
        worker1 = MagicMock(queue_size=3, is_warm_for=MagicMock(return_value=True))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        assert worker1 == dispatcher.choose_worker(MagicMock())

    def test_choose_worker_scenario_3(self, dispatcher):
        """
        Worker states:
            - 0: some tasks, not warm
            - 1: some tasks, warm

        Worker 1 should be chosen as warm with some tasks is better than not warm and some tasks.
        """
        worker0 = MagicMock(queue_size=1, is_warm_for=MagicMock(return_value=False))
        worker1 = MagicMock(queue_size=1, is_warm_for=MagicMock(return_value=True))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        assert worker1 == dispatcher.choose_worker(MagicMock())

    def test_choose_worker_scenario_4(self, dispatcher):
        """
        Worker states:
            - 0: full, not warm
            - 1: empty, not warm

        Worker 1 should be chosen as empty but not warm is better than full and not warm
        """
        worker0 = MagicMock(queue_size=3, is_warm_for=MagicMock(return_value=False))
        worker1 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=False))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        assert worker1 == dispatcher.choose_worker(MagicMock())

    def test_choose_worker_scenario_5(self, dispatcher):
        """
        Worker states:
            - 0: some tasks, not warm
            - 1: empty, not warm
            - 2: full, not warm

        Worker 1 should be chosen as empty but not warm is better than full and not warm as well as
        some tasks and not warm.
        """
        worker0 = MagicMock(queue_size=3, is_warm_for=MagicMock(return_value=False))
        worker1 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=False))
        worker2 = MagicMock(queue_size=1, is_warm_for=MagicMock(return_value=False))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        dispatcher.workers[2] = worker2
        assert worker1 == dispatcher.choose_worker(MagicMock())

    def test_choose_worker_scenario_6(self, dispatcher):
        """
        Worker states:
            - 0: some tasks, warm
            - 1: empty, warm
            - 2: full, warm

        Worker 1 should be chosen as empty but warm is better than full and warm as well as some
        tasks and warm.
        """
        worker0 = MagicMock(queue_size=3, is_warm_for=MagicMock(return_value=True))
        worker1 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=True))
        worker2 = MagicMock(queue_size=1, is_warm_for=MagicMock(return_value=True))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        dispatcher.workers[2] = worker2
        assert worker1 == dispatcher.choose_worker(MagicMock())

    def test_choose_worker_scenario_7(self, dispatcher):
        """
        Worker states:
            - 0: full, warm
            - 1: empty, not warm

        Worker 1 should be chosen as empty but not warm is better than full and warm
        """
        worker0 = MagicMock(queue_size=3, is_warm_for=MagicMock(return_value=True))
        worker1 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=False))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        assert worker1 == dispatcher.choose_worker(MagicMock())

    def test_choose_worker_scenario_8(self, dispatcher):
        """
        Worker states:
            - 0: some tasks, warm
            - 1: empty, not warm

        A random choice between worker 0 and worker 1 should be made as warm but with some tasks is
        equal to empty but not warm.
        """
        worker0 = MagicMock(queue_size=1, is_warm_for=MagicMock(return_value=True))
        worker1 = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=False))
        dispatcher.workers[0] = worker0
        dispatcher.workers[1] = worker1
        mock_choice = MagicMock()
        with patch('iiif.processing.random', MagicMock(choice=mock_choice)):
            dispatcher.choose_worker(MagicMock())
        mock_choice.assert_called_once_with([worker0, worker1])

    @pytest.mark.asyncio
    async def test_submit_already_in_progress_success(self, event_loop, dispatcher, config,
                                                      cache_path):
        source = create_image(config, 100, 100)
        image_task = Task(source, cache_path, 'full', 'max')

        # create a future to mimic what would happen if a request for this task had already been
        # submitted
        previous_request_future = Future()
        dispatcher.output_paths[image_task.output_path] = previous_request_future

        # launch a task to which submits the task, this should await the result of the previous
        # request future created above
        task = event_loop.create_task(dispatcher.submit(image_task))
        # indicate the the future is complete
        previous_request_future.set_result(None)
        # make sure our task is done
        await task
        # assert it
        assert task.done()

    @pytest.mark.asyncio
    async def test_submit_already_in_progress_failure(self, event_loop, dispatcher, config,
                                                      cache_path):
        source = create_image(config, 100, 100)
        image_task = Task(source, cache_path, 'full', 'max')

        # create a future to mimic what would happen if a request for this task had already been
        # submitted
        previous_request_future = Future()
        dispatcher.output_paths[image_task.output_path] = previous_request_future

        # launch a task to which submits the task, this should await the result of the previous
        # request future created above
        task = event_loop.create_task(dispatcher.submit(image_task))
        # indicate the the future is complete but failed
        mock_exception = Exception('test!')
        previous_request_future.set_exception(mock_exception)

        with pytest.raises(Exception) as e:
            await task

        assert e.value == mock_exception

    @pytest.mark.asyncio
    async def test_submit_already_finished_success(self, event_loop, dispatcher, config,
                                                   cache_path):
        source = create_image(config, 100, 100)
        image_task = Task(source, cache_path, 'full', 'max')

        # create a future to mimic what would happen if a request for this task had already been
        # submitted
        previous_request_future = Future()
        previous_request_future.set_result(None)
        dispatcher.output_paths[image_task.output_path] = previous_request_future

        # launch a task to which submits the task, this should finish almost immediately
        task = event_loop.create_task(dispatcher.submit(image_task))
        # make sure our task is done
        await task
        # assert it
        assert task.done()

    @pytest.mark.asyncio
    async def test_submit_already_finished_failure(self, dispatcher, config, cache_path):
        source = create_image(config, 100, 100)
        image_task = Task(source, cache_path, 'full', 'max')

        # create a future to mimic what would happen if a request for this task had already been
        # submitted
        previous_request_future = Future()
        mock_exception = Exception('test!')
        previous_request_future.set_exception(mock_exception)
        dispatcher.output_paths[image_task.output_path] = previous_request_future

        with pytest.raises(Exception) as e:
            await dispatcher.submit(image_task)

        assert e.value == mock_exception

    @pytest.mark.asyncio
    async def test_submit_path_exists(self, dispatcher, config, cache_path):
        source = create_image(config, 100, 100)
        image_task = Task(source, cache_path, 'full', 'max')

        # make sure the path exists and write some data there
        image_task.output_path.parent.mkdir(parents=True, exist_ok=True)
        with image_task.output_path.open('w') as f:
            f.write('something!')

        mock_worker = MagicMock()
        dispatcher.workers[0] = mock_worker

        await dispatcher.submit(image_task)

        assert dispatcher.output_paths[image_task.output_path].done()
        mock_worker.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_path_does_not_exist(self, event_loop, dispatcher, config, cache_path):
        source = create_image(config, 100, 100)
        image_task = Task(source, cache_path, 'full', 'max')

        def mock_add(t):
            # just immediately complete the task
            dispatcher.output_paths[t.output_path].set_result(None)

        mock_worker = MagicMock(queue_size=0, is_warm_for=MagicMock(return_value=True),
                                add=MagicMock(side_effect=mock_add))
        dispatcher.workers[0] = mock_worker

        # launch a task to which submits the task, this should finish almost immediately
        task = event_loop.create_task(dispatcher.submit(image_task))

        # make sure our task is done
        await task
        # assert it
        assert task.done()

        assert dispatcher.output_paths[image_task.output_path].done()
        mock_worker.add.assert_called_once_with(image_task)
