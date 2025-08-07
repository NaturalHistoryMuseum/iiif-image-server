import json
import logging
from unittest.mock import MagicMock, patch

from iiif.exceptions import (
    IIIFServerException,
    ImageNotFound,
    InvalidIIIFParameter,
    ProfileNotFound,
    TooManyImages,
    handler,
)


def test_profile_not_found():
    assert ProfileNotFound('test').status_code == 404


def test_image_not_found():
    assert ImageNotFound('test', 'image').status_code == 404


def test_too_many_images():
    error = TooManyImages(1029)
    assert error.status_code == 400
    assert '1029' in error.public


def test_invalid_iiif_parameter():
    error = InvalidIIIFParameter('Goats', 'Beans')
    assert error.status_code == 400
    assert 'Goats' in error.public
    assert 'Beans' in error.public


class TestExceptionHandler:
    @patch('iiif.exceptions.logger')
    async def test_no_log_use_public(self, mock_logger):
        exception = IIIFServerException('public message', use_public_as_log=True)
        response = await handler(MagicMock(), exception)
        assert response.status_code == exception.status_code
        assert json.loads(response.body)['error'] == exception.public
        mock_logger.log.assert_called_with(exception.level, exception.log)

    @patch('iiif.exceptions.logger')
    async def test_custom_log(self, mock_logger):
        exception = IIIFServerException('public message', log='log message')
        response = await handler(MagicMock(), exception)
        assert response.status_code == exception.status_code
        assert json.loads(response.body)['error'] == exception.public
        mock_logger.log.assert_called_with(exception.level, exception.log)

    @patch('iiif.exceptions.logger')
    async def test_no_log(self, mock_logger):
        exception = IIIFServerException('public message', use_public_as_log=False)
        response = await handler(MagicMock(), exception)
        assert response.status_code == exception.status_code
        assert json.loads(response.body)['error'] == exception.public
        assert not mock_logger.log.called

    async def test_status_code(self):
        exception = IIIFServerException('public message', status_code=418)
        response = await handler(MagicMock(), exception)
        assert response.status_code == exception.status_code

    @patch('iiif.exceptions.logger')
    async def test_cause(self, mock_logger):
        cause = Exception('oh no, something went wrong')
        exception = IIIFServerException('public message', cause=cause)
        response = await handler(MagicMock(), exception)
        assert json.loads(response.body)['error'] == exception.public
        mock_logger.log.assert_called_with(
            exception.level, f'An error occurred: {cause}'
        )

    @patch('iiif.exceptions.logger')
    async def test_level(self, mock_logger):
        exception = IIIFServerException('public message', level=logging.CRITICAL)
        response = await handler(MagicMock(), exception)
        assert json.loads(response.body)['error'] == exception.public
        mock_logger.log.assert_called_with(exception.level, 'public message')
