import base64
import os
import pytest
import re
from tornado.concurrent import Future
from tornado.web import HTTPError
from unittest.mock import MagicMock, patch, AsyncMock

from iiif.image import IIIFImage, ImageSourceFetcher, ImageSourceSizer
from tests.utils import create_image_data, create_image


class TestIIIFImage:

    def test_colon_must_exist(self):
        with pytest.raises(HTTPError) as e:
            IIIFImage('nocolon', MagicMock(), MagicMock())

        assert e.value.status_code == 404
        assert e.value.reason == 'Identifier type not specified'

    def test_it_does_work(self, config):
        image = IIIFImage('test:animage', config['source_path'], config['cache_path'])

        assert image.type == 'test'
        assert image.name == 'animage'
        assert image.source_path == config['source_path'] / image.type / image.name
        assert image.cache_path == config['cache_path'] / image.type / image.name


class TestImageSourceFetcher:

    @pytest.fixture
    def config(self, config):
        # override the config to add a couple of web types
        config['types']['test_web'] = dict(source='web')
        config['types']['test_trusted_web'] = dict(source='trusted_web')
        return config

    @pytest.fixture
    def fetcher(self, config):
        return ImageSourceFetcher(config)

    @pytest.fixture
    def web_image(self, tmp_path):
        return IIIFImage('test_web:image', tmp_path / 'source', tmp_path / 'cache')

    @pytest.fixture
    def trusted_web_image(self, tmp_path):
        name = base64.urlsafe_b64encode(
            f'https://does.not.matter/image.jpg'.encode('utf-8')).decode('utf-8')
        return IIIFImage(f'test_trusted_web:{name}', tmp_path / 'source', tmp_path / 'cache')

    @pytest.fixture
    def disk_image(self, tmp_path):
        # test is defined in the global config as a disk type
        return IIIFImage('test:image', tmp_path / 'source', tmp_path / 'cache')

    @pytest.mark.asyncio
    async def test_fetch_web_image_failure(self, fetcher, web_image):
        mock_fetch = AsyncMock(return_value=MagicMock(code=503))
        async_http_client_mock = MagicMock(return_value=MagicMock(fetch=mock_fetch))
        fake_url = 'https://does.not.matter/{name}/image.jpg'

        fetcher.images[web_image.identifier] = Future()

        with patch('iiif.image.AsyncHTTPClient', async_http_client_mock):
            with pytest.raises(HTTPError) as e:
                await fetcher._fetch_web_image(web_image, fake_url)

        assert e.value.status_code == 404
        assert e.value.reason == f'Source image not found (503)'
        # make sure the fetch function doesn't take it upon itself to update the image's future
        assert not fetcher.images[web_image.identifier].done()
        mock_fetch.assert_awaited_once_with(fake_url.format(name=web_image.name), raise_error=False)

    @pytest.mark.asyncio
    async def test_fetch_web_image_success(self, fetcher, web_image):
        mock_fetch = AsyncMock(return_value=MagicMock(code=200, body=b'test!'))
        async_http_client_mock = MagicMock(return_value=MagicMock(fetch=mock_fetch))
        fake_url = 'https://does.not.matter/{name}/image.jpg'

        fetcher.images[web_image.identifier] = Future()

        with patch('iiif.image.AsyncHTTPClient', async_http_client_mock):
            await fetcher._fetch_web_image(web_image, fake_url)

        # make sure the fetch function doesn't take it upon itself to update the image's future
        assert not fetcher.images[web_image.identifier].done()
        mock_fetch.assert_awaited_once_with(fake_url.format(name=web_image.name), raise_error=False)

        assert os.path.exists(web_image.source_path)
        with open(web_image.source_path, 'rb') as f:
            assert f.read() == b'test!'

    @pytest.mark.asyncio
    async def test_fetch_trusted_web_image_regex_failure(self, fetcher, trusted_web_image):
        mock_fetch = AsyncMock(return_value=MagicMock(code=503))
        async_http_client_mock = MagicMock(return_value=MagicMock(fetch=mock_fetch))

        fetcher.images[trusted_web_image.identifier] = Future()

        with patch('iiif.image.AsyncHTTPClient', async_http_client_mock):
            with pytest.raises(HTTPError) as e:
                await fetcher._fetch_trusted_web_image(trusted_web_image, re.compile('a'))

        assert e.value.status_code == 400
        assert e.value.reason == 'Type not matched'
        # make sure the fetch function doesn't take it upon itself to update the image's future
        assert not fetcher.images[trusted_web_image.identifier].done()
        mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_trusted_web_image_failure(self, fetcher, trusted_web_image):
        mock_fetch = AsyncMock(return_value=MagicMock(code=503))
        async_http_client_mock = MagicMock(return_value=MagicMock(fetch=mock_fetch))

        fetcher.images[trusted_web_image.identifier] = Future()

        with patch('iiif.image.AsyncHTTPClient', async_http_client_mock):
            with pytest.raises(HTTPError) as e:
                await fetcher._fetch_trusted_web_image(trusted_web_image, re.compile('.*'))

        assert e.value.status_code == 404
        assert e.value.reason == f'Source image not found (503)'
        # make sure the fetch function doesn't take it upon itself to update the image's future
        assert not fetcher.images[trusted_web_image.identifier].done()
        fake_url = base64.urlsafe_b64decode(trusted_web_image.name).decode('utf-8')
        mock_fetch.assert_awaited_once_with(fake_url, raise_error=False)

    @pytest.mark.asyncio
    async def test_fetch_trusted_web_image_success(self, fetcher, trusted_web_image):
        mock_fetch = AsyncMock(return_value=MagicMock(code=200, body=b'test!'))
        async_http_client_mock = MagicMock(return_value=MagicMock(fetch=mock_fetch))

        fetcher.images[trusted_web_image.identifier] = Future()

        with patch('iiif.image.AsyncHTTPClient', async_http_client_mock):
            await fetcher._fetch_trusted_web_image(trusted_web_image, re.compile('.*'))

        # make sure the fetch function doesn't take it upon itself to update the image's future
        assert not fetcher.images[trusted_web_image.identifier].done()
        fake_url = base64.urlsafe_b64decode(trusted_web_image.name).decode('utf-8')
        mock_fetch.assert_awaited_once_with(fake_url, raise_error=False)

        assert os.path.exists(trusted_web_image.source_path)
        with open(trusted_web_image.source_path, 'rb') as f:
            assert f.read() == b'test!'

    @pytest.mark.asyncio
    async def test_fetch_disk_image_should_always_error(self, fetcher, disk_image):
        fetcher.images[disk_image.identifier] = Future()

        # create an on disk image to check that the function still fails even if the source exists
        create_image_data(disk_image, 100, 200)

        with pytest.raises(HTTPError) as e:
            await fetcher._fetch_disk_image(disk_image)

        assert e.value.status_code == 404
        assert e.value.reason == f'Source image not found'
        # make sure the fetch function doesn't take it upon itself to update the image's future
        assert not fetcher.images[disk_image.identifier].done()

    @pytest.mark.asyncio
    async def test_ensure_source_exists_already_in_progress(self, fetcher, disk_image, event_loop):
        # create a future to mimic what would happen if a request for this image's source had
        # already been made
        previous_request_future = Future()
        fetcher.images[disk_image.identifier] = previous_request_future

        # launch a task to get the source image, this should await the result of the previous
        # request future created above
        task = event_loop.create_task(fetcher.ensure_source_exists(disk_image))
        # indicate the the future is complete
        previous_request_future.set_result(None)
        # make sure our task is done
        await task
        # assert it
        assert task.done()

    @pytest.mark.asyncio
    async def test_ensure_source_exists_already_errored(self, fetcher, disk_image, event_loop):
        # create a future to mimic what would happen if a request for this image's source had
        # already been made
        previous_request_future = Future()
        fetcher.images[disk_image.identifier] = previous_request_future

        # launch a task to get the source image, this should await the result of the previous
        # request future created above
        task = event_loop.create_task(fetcher.ensure_source_exists(disk_image))
        # indicate the the future is complete but it raised an error
        error = Exception('oh no!')
        previous_request_future.set_exception(error)

        with pytest.raises(Exception) as e:
            await task
        assert e.value == error

    @pytest.mark.asyncio
    async def test_ensure_source_exists_already_complete(self, fetcher, disk_image):
        # create a future to mimic what would happen if a request for this image's source had
        # already been made
        previous_request_future = Future()
        # indicate the the future is complete
        previous_request_future.set_result(None)
        fetcher.images[disk_image.identifier] = previous_request_future

        # this should complete basically instantly
        await fetcher.ensure_source_exists(disk_image)

    @pytest.mark.asyncio
    async def test_ensure_source_exists_type_not_supported(self, fetcher, disk_image):
        image = IIIFImage('beans:id', MagicMock(), MagicMock())
        with pytest.raises(HTTPError) as e:
            await fetcher.ensure_source_exists(image)

        assert e.value.status_code == 400
        assert e.value.reason == 'Identifier type not supported'

    @pytest.mark.asyncio
    async def test_ensure_source_exists_web(self, fetcher, web_image):
        fetch_mock = AsyncMock(return_value=None)
        fetcher._fetch_web_image = fetch_mock
        await fetcher.ensure_source_exists(web_image)
        fetch_mock.assert_awaited_once_with(web_image)

    @pytest.mark.asyncio
    async def test_ensure_source_exists_disk(self, fetcher, disk_image):
        fetch_mock = AsyncMock(return_value=None)
        fetcher._fetch_disk_image = fetch_mock
        await fetcher.ensure_source_exists(disk_image)
        fetch_mock.assert_awaited_once_with(disk_image)

    @pytest.mark.asyncio
    async def test_ensure_source_exists_trusted_web(self, fetcher, trusted_web_image):
        fetch_mock = AsyncMock(return_value=None)
        fetcher._fetch_trusted_web_image = fetch_mock
        await fetcher.ensure_source_exists(trusted_web_image)
        fetch_mock.assert_awaited_once_with(trusted_web_image)

    @pytest.mark.asyncio
    async def test_ensure_source_exists_source_failure(self, fetcher, config):
        existing_type = 'some_type'
        missing_source = 'some_source'
        image = IIIFImage(f'{existing_type}:nope', config['source_path'], config['cache_path'])
        fetcher.types[existing_type] = (missing_source, {})

        with pytest.raises(HTTPError) as e:
            await fetcher.ensure_source_exists(image)

        assert e.value.status_code == 500
        assert e.value.reason == f'Identifier type {missing_source} not supported'
        assert fetcher.images[image.identifier].exception() == e.value


class TestImageSourceSizer:

    @pytest.fixture
    def sizer(self):
        sizer = ImageSourceSizer(dict(size_pool_size=1))
        yield sizer
        sizer.stop()

    @pytest.mark.asyncio
    async def test_get_image_size(self, config, sizer):
        size = (60, 30)
        image = create_image(config, *size)
        assert await sizer.get_image_size(image) == size

    @pytest.mark.asyncio
    async def test_get_image_size_error(self, config, sizer):
        image = IIIFImage('test:image', config['source_path'], config['cache_path'])

        with pytest.raises(FileNotFoundError):
            await sizer.get_image_size(image)
