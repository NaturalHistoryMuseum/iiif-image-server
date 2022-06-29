#!/usr/bin/env python3
# encoding: utf-8

import logging
from fastapi import Request
from starlette.responses import JSONResponse
from typing import Optional
from iiif.utils import logger


class IIIFServerException(Exception):

    def __init__(self, public: str, status_code: int = 500, log: Optional[str] = None,
                 level: int = logging.WARNING, cause: Optional[Exception] = None,
                 use_public_as_log: bool = True):
        super().__init__(public)
        self.status_code = status_code
        self.public = public
        self.log = log
        self.level = level
        self.cause = cause
        if log is not None:
            self.log = log
        else:
            if self.cause is not None:
                self.log = f'An error occurred: {self.cause}'
            elif use_public_as_log:
                self.log = self.public


class Timeout(IIIFServerException):

    def __init__(self, *args, **kwargs):
        super().__init__(f'A timeout occurred, please try again', 500, *args, **kwargs)


class ProfileNotFound(IIIFServerException):

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(f'Profile {name} not recognised', 404, *args, **kwargs)
        self.name = name


class ImageNotFound(IIIFServerException):

    def __init__(self, profile: str, name: str, *args, **kwargs):
        super().__init__(f'Image {name} not found in profile {profile}', 404, *args,
                         **kwargs)
        self.profile = profile
        self.name = name


class TooManyImages(IIIFServerException):

    def __init__(self, max_files: int, *args, **kwargs):
        super().__init__(f'Too many images requested (max: {max_files})', 400, *args, **kwargs)
        self.max_files = max_files


class InvalidIIIFParameter(IIIFServerException):

    def __init__(self, name: str, value: str, *args, **kwargs):
        super().__init__(f'Invalid IIIF option: {name} value "{value}" is invalid', 400,
                         use_public_as_log=False, *args, **kwargs)
        self.name = name
        self.value = value


async def handler(request: Request, exception: IIIFServerException) -> JSONResponse:
    log_error(exception)
    return JSONResponse(
        status_code=exception.status_code,
        content={
            'error': exception.public
        },
    )


def log_error(exception: IIIFServerException):
    if exception.log:
        logger.log(exception.level, exception.log)
