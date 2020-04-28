import os
import pytest
from PIL import Image
from tornado.concurrent import Future
from tornado.web import HTTPError
from unittest.mock import MagicMock, patch, AsyncMock

from iiif.image import IIIFImage, ImageSourceFetcher, ImageSourceSizer


class TestIIIFImage:

    def test_colon_must_exist(self):
        with pytest.raises(HTTPError) as e:
            IIIFImage('nocolon', MagicMock(), MagicMock())

        assert e.value.status_code == 404
        assert e.value.reason == 'Identifier type not specified'

    def test_type_must_be_supported(self):
        with pytest.raises(HTTPError) as e:
            IIIFImage('beans:id', MagicMock(), MagicMock())

        assert e.value.status_code == 404
        assert e.value.reason == 'Identifier type not supported'

    def test_it_does_work(self):
        image = IIIFImage('vfactor:animage', 'source', 'cache')

        assert image.type == 'vfactor'
        assert image.name == 'animage'
        assert image.source_path == os.path.join('source', image.type, image.name)
        assert image.cache_path == os.path.join('cache', image.type, image.name)


class TestImageSourceFetcher:

    @pytest.mark.asyncio
    async def test_fetch_mam_image_failure(self, tmp_path):
        mock_fetch = AsyncMock(return_value=MagicMock(code=503))
        async_http_client_mock = MagicMock(return_value=MagicMock(fetch=mock_fetch))

        fetcher = ImageSourceFetcher(MagicMock())
        image = IIIFImage('mam:image', tmp_path / 'source', tmp_path / 'cache')
        fetcher.images[image.identifier] = Future()

        with patch('iiif.image.AsyncHTTPClient', async_http_client_mock):
            exception = await fetcher._fetch_mam_image(image)

        assert exception is not None
        assert exception.status_code == 404
        assert exception.reason == f'MAM image not found (503)'
        assert not fetcher.images[image.identifier].done()
        mock_fetch.assert_awaited_once_with(fetcher.mam_url.format(image.name), raise_error=False)

    @pytest.mark.asyncio
    async def test_fetch_mam_image_success(self, tmp_path):
        mock_fetch = AsyncMock(return_value=MagicMock(code=200, body=b'test!'))
        async_http_client_mock = MagicMock(return_value=MagicMock(fetch=mock_fetch))

        fetcher = ImageSourceFetcher(MagicMock())
        image = IIIFImage('mam:image', tmp_path / 'source', tmp_path / 'cache')
        fetcher.images[image.identifier] = Future()

        with patch('iiif.image.AsyncHTTPClient', async_http_client_mock):
            await fetcher._fetch_mam_image(image)

        # the future should not be completed by this function, this is done by the calling
        # ensure_source_exists function
        assert not fetcher.images[image.identifier].done()

        assert os.path.exists(image.source_path)
        with open(image.source_path, 'rb') as f:
            assert f.read() == b'test!'

    @pytest.mark.asyncio
    async def test_ensure_source_exists_already_in_progress(self, tmp_path, event_loop):
        image = IIIFImage('mam:image', tmp_path / 'source', tmp_path / 'cache')
        fetcher = ImageSourceFetcher(MagicMock())

        # create a future to mimic what would happen if a request for this image's source had
        # already been made
        previous_request_future = Future()
        fetcher.images[image.identifier] = previous_request_future

        # launch a task to get the source image, this should await the result of the previous
        # request future created above
        task = event_loop.create_task(fetcher.ensure_source_exists(image))
        # indicate the the future is complete
        previous_request_future.set_result(None)
        # make sure our task is done
        await task
        # assert it
        assert task.done()

    @pytest.mark.asyncio
    async def test_ensure_source_exists_already_errored(self, tmp_path, event_loop):
        image = IIIFImage('mam:image', tmp_path / 'source', tmp_path / 'cache')
        fetcher = ImageSourceFetcher(MagicMock())

        # create a future to mimic what would happen if a request for this image's source had
        # already been made
        previous_request_future = Future()
        fetcher.images[image.identifier] = previous_request_future

        # launch a task to get the source image, this should await the result of the previous
        # request future created above
        task = event_loop.create_task(fetcher.ensure_source_exists(image))
        # indicate the the future is complete but it raised an error
        error = Exception('oh no!')
        previous_request_future.set_exception(error)

        with pytest.raises(Exception) as e:
            await task
        assert e.value == error

    @pytest.mark.asyncio
    async def test_ensure_source_exists_already_complete(self, tmp_path):
        image = IIIFImage('mam:image', tmp_path / 'source', tmp_path / 'cache')
        fetcher = ImageSourceFetcher(MagicMock())

        # create a future to mimic what would happen if a request for this image's source had
        # already been made
        previous_request_future = Future()
        # indicate the the future is complete
        previous_request_future.set_result(None)
        fetcher.images[image.identifier] = previous_request_future

        # this should complete basically instantly
        await fetcher.ensure_source_exists(image)

    @pytest.mark.asyncio
    async def test_ensure_source_exists_mam(self, tmp_path):
        image = IIIFImage('mam:image', tmp_path / 'source', tmp_path / 'cache')
        fetcher = ImageSourceFetcher(MagicMock())
        fetch_mock = AsyncMock(return_value=None)
        fetcher._fetch_mam_image = fetch_mock
        await fetcher.ensure_source_exists(image)
        fetch_mock.assert_awaited_once_with(image)

    @pytest.mark.asyncio
    async def test_ensure_source_exists_vfactor_success(self, tmp_path):
        image = IIIFImage('vfactor:there', tmp_path / 'source', tmp_path / 'cache')

        os.makedirs(os.path.dirname(image.source_path), exist_ok=True)
        with open(image.source_path, 'wb') as f:
            f.write(b'test!')

        fetcher = ImageSourceFetcher(MagicMock())
        await fetcher.ensure_source_exists(image)

    @pytest.mark.asyncio
    async def test_ensure_source_exists_vfactor_failure(self, tmp_path):
        image = IIIFImage('vfactor:there', tmp_path / 'source', tmp_path / 'cache')
        fetcher = ImageSourceFetcher(MagicMock())
        with pytest.raises(HTTPError) as e:
            await fetcher.ensure_source_exists(image)

        assert e.value.status_code == 404
        assert e.value.reason == 'VFactor image not found'
        assert fetcher.images[image.identifier].exception() == e.value

    @pytest.mark.asyncio
    async def test_ensure_source_exists_type_failure(self, tmp_path):
        # we have to mock the IIIFImage object because it doesn't allow unsupported types
        image = MagicMock(identifier='notatype:image', source_path=tmp_path / 'notthere')

        fetcher = ImageSourceFetcher(MagicMock())
        with pytest.raises(HTTPError) as e:
            await fetcher.ensure_source_exists(image)

        assert e.value.status_code == 500
        assert e.value.reason == 'Identifier type not supported'
        assert fetcher.images[image.identifier].exception() == e.value


class TestImageSourceSizer:

    @pytest.fixture
    def sizer(self):
        sizer = ImageSourceSizer(dict(size_pool_size=1))
        yield sizer
        sizer.stop()

    @pytest.mark.asyncio
    async def test_get_image_size(self, tmp_path, sizer):
        image = IIIFImage('vfactor:image', tmp_path / 'source', tmp_path / 'cache')

        os.makedirs(os.path.dirname(image.source_path), exist_ok=True)
        # create a test image
        size = (60, 30)
        Image.new('RGB', size, color='red').save(image.source_path, format='jpeg')

        assert await sizer.get_image_size(image) == size

    @pytest.mark.asyncio
    async def test_get_image_size_error(self, tmp_path, sizer):
        image = IIIFImage('vfactor:image', tmp_path / 'source', tmp_path / 'cache')

        with pytest.raises(FileNotFoundError):
            await sizer.get_image_size(image)
