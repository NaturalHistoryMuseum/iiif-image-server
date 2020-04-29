import hashlib
import io
import os
import pytest
from PIL import Image
from queue import Queue

from iiif.image import IIIFImage
from iiif.processing import Task, process_image_requests

default_image_width = 4000
default_image_height = 5000


def create_image(tmp_path, width, height):
    image = IIIFImage('vfactor:image', tmp_path / 'source', tmp_path / 'cache')
    os.makedirs(os.path.dirname(image.source_path), exist_ok=True)
    img = Image.new('RGB', (width, height), color='red')
    img.save(image.source_path, format='jpeg')
    return image


@pytest.fixture
def image(tmp_path):
    return create_image(tmp_path, default_image_width, default_image_height)


@pytest.fixture
def task_queue():
    return Queue()


@pytest.fixture
def result_queue():
    return Queue()


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


class TestProcessImageRequestsLevel0:
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
        assert result_queue.get() == (0, task)
        assert os.path.exists(task.output_path)
        check_size(task, default_image_width, default_image_height)
        check_result(task, lambda img: img)


class TestProcessImageRequestsLevel1:
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
        assert result_queue.get() == (0, task)
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
        assert result_queue.get() == (0, task)
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
        assert result_queue.get() == (0, task)
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
        assert result_queue.get() == (0, task)
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
        assert result_queue.get() == (0, task)
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
        assert result_queue.get() == (0, task)
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
        assert result_queue.get() == (0, task)
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
        assert result_queue.get() == (0, task)
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
        assert result_queue.get() == (0, task)
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
        assert result_queue.get() == (0, task)
        assert os.path.exists(task.output_path)
        check_size(task, width, height)
        check_result(task, lambda img: img.resize((width, height)))
