import hashlib
import io
import os
import pytest
from PIL import Image
from queue import Queue

from iiif.image import IIIFImage
from iiif.processing import Task, process_image_request

default_image_width = 4000
default_image_height = 5000


@pytest.fixture
def image(tmp_path):
    image = IIIFImage('vfactor:image', tmp_path / 'source', tmp_path / 'cache')
    os.makedirs(os.path.dirname(image.source_path), exist_ok=True)
    img = Image.new('RGB', (default_image_width, default_image_height), color='red')
    img.save(image.source_path, format='jpeg')
    return image


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


class TestProcessImageRequestLevel0:
    """
    Test the process_image_request function for IIIF Image API v3 level 0 compliance.
    See: https://iiif.io/api/image/3.0/compliance/.

    Note that we implicitly don't support rotations other than 0, quality other than and formats
    other than jpg and therefore we don't need to test for them.
    """

    def test(self, image, task_queue, result_queue):
        # this is all that is expected at level 0
        task = Task(image, 'full', 'max')
        task_queue.put(task)
        task_queue.put(None)

        process_image_request(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task)
        assert os.path.exists(task.output_path)
        check_size(task, default_image_width, default_image_height)
        check_result(task, lambda img: img)


class TestProcessImageRequestLevel1:
    """
    Test the process_image_request function for IIIF Image API v3 level 1 compliance.
    See: https://iiif.io/api/image/3.0/compliance/.
    """

    def test_regionByPx(self, image, task_queue, result_queue):
        x, y, w, h = 0, 0, 1024, 1024
        task = Task(image, f'{x},{y},{w},{h}', 'max')
        task_queue.put(task)
        task_queue.put(None)

        process_image_request(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task)
        assert os.path.exists(task.output_path)
        check_size(task, 1024, 1024)
        check_result(task, lambda img: img.crop((x, y, x + w, y + h)))

    @pytest.mark.skip
    def test_regionSquare(self):
        # needs to be implemented for full level 1 compliance
        pass

    def test_sizeByW(self, image, task_queue, result_queue):
        width = 512
        expected_height = int(default_image_height * width / default_image_width)
        task = Task(image, 'full', f'{width},')
        task_queue.put(task)
        task_queue.put(None)

        process_image_request(0, task_queue, result_queue, 1)

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

        process_image_request(0, task_queue, result_queue, 1)

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

        process_image_request(0, task_queue, result_queue, 1)

        assert result_queue.qsize() == 1
        assert result_queue.get() == (0, task)
        assert os.path.exists(task.output_path)
        check_size(task, width, height)
        check_result(task, lambda img: img.resize((width, height)))
