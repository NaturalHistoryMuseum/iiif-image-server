#!/usr/bin/env python3
# encoding: utf-8

from collections import Counter, defaultdict

import asyncio
import humanize
import pytest
from PIL import Image

from iiif.utils import convert_image, generate_sizes, get_size, get_mss_base_url_path, \
    get_path_stats, parse_size, OnceRunner
from tests.utils import create_image


class TestConvertImage:

    def test_jpeg(self, tmp_path):
        image_path = tmp_path / 'image'
        img = Image.new('RGB', (400, 400), color='red')
        img.save(image_path, format='jpeg')

        target = tmp_path / 'converted'
        convert_image(image_path, target)

        assert target.exists()
        with Image.open(target) as converted_image:
            assert converted_image.format.lower() == 'jpeg'

    def test_jpeg_with_exif_orientation(self, tmp_path):
        image_path = tmp_path / 'image'
        img = Image.new('RGB', (700, 400), color='red')
        exif = img.getexif()
        exif[0x0112] = 6
        img.info['exif'] = exif.tobytes()
        img.save(image_path, format='jpeg')

        target = tmp_path / 'converted'
        convert_image(image_path, target)

        assert target.exists()
        with Image.open(target) as converted_image:
            assert converted_image.format.lower() == 'jpeg'
            assert 0x0112 not in converted_image.getexif()

    def test_tiff(self, tmp_path):
        image_path = tmp_path / 'image'
        img = Image.new('RGB', (400, 400), color='red')
        img.save(image_path, format='tiff')

        target = tmp_path / 'converted'
        convert_image(image_path, target)

        assert target.exists()
        with Image.open(target) as converted_image:
            assert converted_image.format.lower() == 'jpeg'

    def test_jpeg_options(self, tmp_path):
        image_path = tmp_path / 'image'
        img = Image.new('RGB', (400, 400), color='red')
        img.save(image_path, format='jpeg')

        target = tmp_path / 'converted'
        convert_image(image_path, target, quality=40, subsampling=1)

        assert target.exists()
        with Image.open(target) as converted_image:
            assert converted_image.format.lower() == 'jpeg'


def test_generate_sizes():
    sizes = generate_sizes(1000, 1001, 200)

    # test that the call result is cached
    assert generate_sizes(1000, 1001, 200) is sizes

    assert len(sizes) == 3
    assert sizes[0] == {'width': 1000, 'height': 1001}
    assert sizes[1] == {'width': 500, 'height': 500}
    assert sizes[2] == {'width': 250, 'height': 250}


def test_get_size(config):
    image_path = create_image(config, 289, 4390)
    assert get_size(image_path) == (289, 4390)


mss_base_url_scenarios = [
    ('0', '0/000'),
    ('1', '0/001'),
    ('14', '0/014'),
    ('305', '0/305'),
    ('9217', '9/217'),
    ('2389749823', '2389749/823'),
]


@pytest.mark.parametrize('name,expected', mss_base_url_scenarios)
def test_get_mss_base_url_path(name, expected):
    assert expected == get_mss_base_url_path(name)


def test_get_path_stats(tmp_path):
    total = 0
    count = 0
    for i in range(10):
        with (tmp_path / f'{i}.text').open('w') as f:
            total += f.write('beans!')
            count += 1
        (tmp_path / f'{i}').mkdir()
        with (tmp_path / f'{i}' / f'{i}.text').open('w') as f:
            total += f.write('beans again!')
            count += 1

    stats = get_path_stats(tmp_path)
    assert stats['count'] == count
    assert stats['size_bytes'] == total
    assert stats['size_pretty'] == humanize.naturalsize(total, binary=True)


def test_get_path_stats_empty(tmp_path):
    stats = get_path_stats(tmp_path / 'empty')
    assert stats['count'] == 0
    assert stats['size_bytes'] == 0
    assert stats['size_pretty'] == humanize.naturalsize(0, binary=True)


class TestParseSize:

    def test_max(self):
        assert parse_size('max', 1000, 500) == (1000, 500)

    def test_just_width(self):
        assert parse_size('500,', 1000, 500) == (500, 250)

    def test_just_height(self):
        assert parse_size(',250', 1000, 500) == (500, 250)

    def test_width_and_height(self):
        assert parse_size('500,250', 1000, 500) == (500, 250)


class TestOnceRunner:

    @pytest.mark.asyncio
    async def test_sequential_usage(self):
        result = object()
        counter = Counter()

        async def task(name):
            counter[name] += 1
            await asyncio.sleep(1)
            return result

        runner = OnceRunner('test', 100)
        assert await runner.run('task1', task, 'task1') is result
        assert await runner.run('task1', task, 'task1') is result
        assert await runner.run('task2', task, 'task2') is result
        assert counter['task1'] == 1
        assert counter['task2'] == 1

    @pytest.mark.asyncio
    async def test_parallel_usage(self):
        results = defaultdict(object)
        counter = Counter()

        async def task(name):
            counter[name] += 1
            await asyncio.sleep(1)
            return name, results[name]

        runner = OnceRunner('test', 100)
        task_a = asyncio.create_task(runner.run('task1', task, 'task1'))
        task_b = asyncio.create_task(runner.run('task2', task, 'task2'))
        task_c = asyncio.create_task(runner.run('task1', task, 'task1'))

        run_results = await asyncio.gather(task_a, task_b, task_c)
        assert set(run_results) == set(results.items())
        assert counter['task1'] == 1
        assert counter['task2'] == 1

    @pytest.mark.asyncio
    async def test_sequential_usage_with_error(self):
        counter = Counter()

        async def task(name):
            counter[name] += 1
            await asyncio.sleep(1)
            raise Exception(f'heyo from {name}!')

        runner = OnceRunner('test', 100)
        with pytest.raises(Exception, match=f'heyo from task1!') as exc_info:
            await runner.run('task1', task, 'task1')
        exception = exc_info.value

        with pytest.raises(Exception, match=f'heyo from task1!') as exc_info:
            await runner.run('task1', task, 'task1')
        assert exception is exc_info.value

        with pytest.raises(Exception, match=f'heyo from task2!') as exc_info:
            await runner.run('task2', task, 'task2')

        assert counter['task1'] == 1
        assert counter['task2'] == 1

    @pytest.mark.asyncio
    async def test_parallel_usage_with_errors(self):
        counter = Counter()

        async def task(name):
            counter[name] += 1
            await asyncio.sleep(1)
            raise Exception(f'heyo from {name}!')

        runner = OnceRunner('test', 100)
        task_a = asyncio.create_task(runner.run('task1', task, 'task1'))
        task_b = asyncio.create_task(runner.run('task2', task, 'task2'))
        task_c = asyncio.create_task(runner.run('task1', task, 'task1'))

        run_results = await asyncio.gather(task_a, task_b, task_c, return_exceptions=True)
        assert len(run_results) == 3
        exceptions = set(run_results)
        assert len(exceptions) == 2
        assert 'heyo from task1!' in set(map(str, exceptions))
        assert 'heyo from task2!' in set(map(str, exceptions))
        assert counter['task1'] == 1
        assert counter['task2'] == 1

    @pytest.mark.asyncio
    async def test_parallel_mixed(self):
        results = defaultdict(object)
        counter = Counter()

        async def erroring_task(name):
            counter[name] += 1
            await asyncio.sleep(1)
            raise Exception(f'heyo from {name}!')

        async def working_task(name):
            counter[name] += 1
            await asyncio.sleep(1.5)
            return name, results[name]

        runner = OnceRunner('test', 100)
        task_a = asyncio.create_task(runner.run('task1', erroring_task, 'task1'))
        task_b = asyncio.create_task(runner.run('task2', working_task, 'task2'))
        task_c = asyncio.create_task(runner.run('task1', erroring_task, 'task1'))
        task_d = asyncio.create_task(runner.run('task2', working_task, 'task2'))
        task_e = asyncio.create_task(runner.run('task3', working_task, 'task3'))

        all_results = await asyncio.gather(task_a, task_b, task_c, task_d, task_e,
                                           return_exceptions=True)
        assert len(all_results) == 5

        exceptions = [result for result in all_results if isinstance(result, Exception)]
        working = [result for result in all_results if not isinstance(result, Exception)]

        assert len(exceptions) == 2
        assert len(set(exceptions)) == 1
        assert len(working) == 3
        assert len(set(working)) == 2

        assert 'heyo from task1!' in set(map(str, exceptions))
        assert set(working) == set(results.items())
        assert counter['task1'] == 1
        assert counter['task2'] == 1
        assert counter['task3'] == 1

    @pytest.mark.asyncio
    async def test_expire(self):
        runner = OnceRunner('test', 100)
        assert not runner.expire('task1')
        await runner.run('task1', asyncio.sleep, 1)
        assert runner.expire('task1')

    @pytest.mark.asyncio
    async def test_exception_timeout(self):
        counter = Counter()

        async def erroring_task(name):
            counter[name] += 1
            await asyncio.sleep(0.5)
            raise Exception(f'heyo!')

        runner = OnceRunner('test', 100, exception_timeout=5)
        # run the erroring task, this should raise an exception
        with pytest.raises(Exception, match='heyo!'):
            await runner.run('task1', erroring_task, 'task1')
        # run the erroring task again, this should also raise an exception but from the result cache
        with pytest.raises(Exception, match='heyo!'):
            await runner.run('task1', erroring_task, 'task1')
        # sleep for 5 seconds to make sure the exception timeout has run out
        await asyncio.sleep(5)
        # run the erroring task, this should raise an exception again, not from the cache
        with pytest.raises(Exception, match='heyo!'):
            await runner.run('task1', erroring_task, 'task1')

        # the counter is 2 because the task was run twice due to the exception timeout
        assert counter['task1'] == 2

    def test_wait(self):
        runner = OnceRunner('test', 100)
        assert runner.waiting == 0
        with runner.wait():
            assert runner.waiting == 1
        assert runner.waiting == 0

    def test_work(self):
        runner = OnceRunner('test', 100)
        assert runner.working == 0
        with runner.work():
            assert runner.working == 1
        assert runner.working == 0

    @pytest.mark.asyncio
    async def test_get_status(self):
        runner = OnceRunner('test', 100)
        stats = await runner.get_status()
        assert stats['size'] == 0
        assert stats['waiting'] == 0
        assert stats['working'] == 0

        with runner.wait():
            with runner.work():
                stats = await runner.get_status()
                assert stats['waiting'] == 1
                assert stats['working'] == 1

        await runner.run('task1', asyncio.sleep, 1)
        stats = await runner.get_status()
        assert stats['size'] == 1
