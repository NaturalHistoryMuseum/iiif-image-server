import hashlib
import io
import multiprocessing as mp
import os
import pytest
from PIL import Image
from queue import Queue
from tornado.web import HTTPError
from unittest.mock import patch, MagicMock, call

from iiif.image import IIIFImage
from iiif.processing import Task, process_image_requests, Worker, ImageProcessingDispatcher

default_image_width = 4000
default_image_height = 5000


def create_image(tmp_path, width, height, identifier='vfactor:image'):
    image = IIIFImage(identifier, tmp_path / 'source', tmp_path / 'cache')
    os.makedirs(os.path.dirname(image.source_path), exist_ok=True)
    img = Image.new('RGB', (width, height), color='red')
    img.save(image.source_path, format='jpeg')
    return image


@pytest.fixture
def image(tmp_path):
    return create_image(tmp_path, default_image_width, default_image_height)


def check_size(task, width, height):
    with Image.open(task.output_path) as image:
        assert image.width == width
        assert image.height == height


def check_result(task, op_function):
    with Image.open(task.image.source_path) as img:
        cropped_source = io.BytesIO()
        img = op_function(img)
        img.save(cropped_source, format='jpeg')
        cropped_source.seek(0)

        with open(task.output_path, 'rb') as f:
            assert (hashlib.sha256(f.read()).digest() ==
                    hashlib.sha256(cropped_source.read()).digest())


class TestProcessImageRequests:
    """
    This base class just provides a couple of fixtures specific to these tests, namely a real Queue
    but not a multiprocessing Queue as is actually used but a standard one.
    """

    @pytest.fixture
    def task_queue(self):
        # a real queue but not a multiprocessing one
        return Queue()

    @pytest.fixture
    def result_queue(self):
        # a real queue but not a multiprocessing one
        return Queue()


class TestProcessImageRequestsLevel0(TestProcessImageRequests):
    """
    Test the process_image_requests function for IIIF Image API v3 level 0 compliance.
    See: https://iiif.io/api/image/3.0/compliance/.

    Note that we implicitly don't support rotations other than 0, quality other than and formats
    other than jpg and therefore we don't need to test for them.
    """

    def test(self, image, task_queue, result_queue):
        # this is all that is expected at level 0
        task = Task(image, 'full', 'max')
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, default_image_width, default_image_height)
        check_result(task, lambda img: img)


class TestProcessImageRequestsLevel1(TestProcessImageRequests):
    """
    Test the process_image_requests function for IIIF Image API v3 level 1 compliance.
    See: https://iiif.io/api/image/3.0/compliance/.
    """

    def test_regionByPx_jpegtran(self, image, task_queue, result_queue):
        x, y, w, h = 0, 0, 1024, 1024
        task = Task(image, f'{x},{y},{w},{h}', 'max')
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, 1024, 1024)
        check_result(task, lambda img: img.crop((x, y, x + w, y + h)))

    def test_regionByPx_any(self, image, task_queue, result_queue):
        x, y, w, h = 6, 191, 1002, 1053
        task = Task(image, f'{x},{y},{w},{h}', 'max')
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, 1002, 1053)
        check_result(task, lambda img: img.crop((x, y, x + w, y + h)))

    def test_regionSquare_a_square(self, tmp_path, task_queue, result_queue):
        width = 500
        height = 500
        task = Task(create_image(tmp_path, width, height), 'square', 'max')
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, height)
        check_result(task, lambda img: img)

    def test_regionSquare_a_portrait_jpegtran(self, tmp_path, task_queue, result_queue):
        width = 512
        height = 768
        task = Task(create_image(tmp_path, width, height), 'square', 'max')
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, width)
        check_result(task, lambda img: img.crop((0, 128, 512, 640)))

    def test_regionSquare_a_landscape_jpegtran(self, tmp_path, task_queue, result_queue):
        width = 768
        height = 512
        task = Task(create_image(tmp_path, width, height), 'square', 'max')
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, height, height)
        check_result(task, lambda img: img.crop((128, 0, 640, 512)))

    def test_regionSquare_a_portrait_any(self, tmp_path, task_queue, result_queue):
        width = 500
        height = 700
        task = Task(create_image(tmp_path, width, height), 'square', 'max')
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, width)
        check_result(task, lambda img: img.crop((0, 100, 500, 600)))

    def test_regionSquare_a_landscape_any(self, tmp_path, task_queue, result_queue):
        width = 700
        height = 500
        task = Task(create_image(tmp_path, width, height), 'square', 'max')
        task_queue.put(task)
        task_queue.put(None)
        process_image_requests(0, task_queue, result_queue, 1)
        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, height, height)
        check_result(task, lambda img: img.crop((100, 0, 600, 500)))

    def test_sizeByW(self, image, task_queue, result_queue):
        width = 512
        expected_height = int(default_image_height * width / default_image_width)
        task = Task(image, 'full', f'{width},')
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, expected_height)
        check_result(task, lambda img: img.resize((width, expected_height)))

    def test_sizeByH(self, image, task_queue, result_queue):
        height = 512
        expected_width = int(default_image_width * height / default_image_height)
        task = Task(image, 'full', f',{height}')
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, expected_width, height)
        check_result(task, lambda img: img.resize((expected_width, height)))

    def test_sizeByWh(self, image, task_queue, result_queue):
        width = 400
        height = 600
        task = Task(image, 'full', f'{width},{height}')
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, width, height)
        check_result(task, lambda img: img.resize((width, height)))


class TestProcessImageRequestsMisc(TestProcessImageRequests):

    def test_region_and_size_precise(self, image, task_queue, result_queue):
        # simple check to make sure region and size play nicely together
        task = Task(image, '100,200,600,891', '256,401')
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, 256, 401)
        check_result(task, lambda img: img.crop((100, 200, 600, 891)).resize((256, 401)))

    def test_region_and_size_inferred(self, tmp_path, task_queue, result_queue):
        # check to make sure region and size play nicely when they're both relying on image
        # dimension ratios
        width = 500
        height = 700
        task = Task(create_image(tmp_path, width, height), 'square', ',400')
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)
        assert os.path.exists(task.output_path)
        check_size(task, 400, 400)
        check_result(task, lambda img: img.crop((0, 100, 500, 600)).resize((400, 400)))

    def test_upscale_errors_when_not_specified(self, image, task_queue, result_queue):
        task = Task(image, '0,0,400,400', '500,500')
        task_queue.put(task)
        task_queue.put(None)

        process_image_requests(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        worker_id, result_task, exception = result_queue.get()
        assert worker_id == 0
        assert result_task == task
        assert isinstance(exception, HTTPError)
        assert exception.status_code == 400
        assert exception.reason == 'Size greater than extracted region without specifying^'


class TestWorker:

    @pytest.fixture
    def worker(self, result_queue):
        worker = Worker(0, result_queue, 1)
        worker.stop()
        return worker

    @pytest.fixture
    def task(self, image):
        return Task(image, 'full', 'max')

    @pytest.fixture
    def result_queue(self):
        return mp.Queue()

    def test_add(self, worker, task):
        assert worker.queue_size == 0

        worker.add(task)

        assert worker.queue_size == 1
        assert task.image.source_path in worker.predicted_cache
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

    def test_is_warm_for(self, tmp_path, result_queue):
        worker = Worker(0, result_queue, 2)

        image1 = create_image(tmp_path, 100, 200, identifier='vfactor:image1')
        image2 = create_image(tmp_path, 100, 200, identifier='vfactor:image2')
        image3 = create_image(tmp_path, 100, 200, identifier='vfactor:image3')
        image4 = create_image(tmp_path, 100, 200, identifier='vfactor:image4')

        task1 = Task(image1, 'full', 'max')
        task2 = Task(image2, 'full', 'max')
        task3 = Task(image3, 'full', 'max')
        task4 = Task(image4, 'full', 'max')

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

    def test_result_queue_is_used(self, tmp_path, result_queue):
        worker = Worker(0, result_queue, 2)
        image = create_image(tmp_path, 100, 200)
        task = Task(image, 'full', 'max')

        worker.add(task)
        worker.stop()

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task, None)


class TestTask:

    def test_equality(self, tmp_path):
        image1 = create_image(tmp_path, 100, 200, identifier='vfactor:image1')
        image2 = create_image(tmp_path, 100, 200, identifier='vfactor:image2')

        assert Task(image1, 'full', 'max') == Task(image1, 'full', 'max')
        assert Task(image1, 'full', 'max') != Task(image2, 'full', 'max')
        assert Task(image2, 'full', 'max') != Task(image2, 'full', '256,')
        assert Task(image2, 'square', 'max') != Task(image2, 'full', 'max')
        # semantically these are the same but we don't recognise it
        assert Task(image1, 'full', 'max') != Task(image1, '0,0,100,200', 'max')
        assert Task(image1, 'full', 'max') != Task(image1, 'full', '100,200')


class TestImageProcessingDispatcher:

    @pytest.fixture
    def dispatcher(self):
        dispatcher = ImageProcessingDispatcher()
        yield dispatcher
        dispatcher.stop()

    def test_result_listener(self):
        ioloop_mock = MagicMock()
        with patch('iiif.processing.IOLoop', ioloop_mock):
            dispatcher = ImageProcessingDispatcher()
            mock_result = ('some', 'mock', 'data')
            dispatcher.result_queue.put(mock_result)
            dispatcher.result_queue.put(None)
            dispatcher.result_thread.join()

        ioloop_mock.current().add_callback.assert_called_once_with(dispatcher.finish_task,
                                                                   *mock_result)

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

    def test_finish_task_success(self, dispatcher, tmp_path):
        worker_id = 0
        task = Task(create_image(tmp_path, 100, 200), 'full', 'max')
        mock_future = MagicMock()
        mock_worker = MagicMock()
        dispatcher.workers[worker_id] = mock_worker
        dispatcher.output_paths[task.output_path] = mock_future

        dispatcher.finish_task(worker_id, task, None)

        mock_worker.done.assert_called_once_with(task)
        mock_future.set_result.called_once_with(None)
        mock_future.set_exception.assert_not_called()

    def test_finish_task_failure(self, dispatcher, tmp_path):
        worker_id = 0
        task = Task(create_image(tmp_path, 100, 200), 'full', 'max')
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
