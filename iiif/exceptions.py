#!/usr/bin/env python3
# encoding: utf-8

from fastapi import HTTPException


def profile_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail='Profile not recognised')


def image_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail='Image not found')


def too_many_images(max_files) -> HTTPException:
    return HTTPException(status_code=400, detail=f'Too many images requested (max: {max_files})')


def invalid_iiif_parameter(name: str, value: str) -> HTTPException:
    return HTTPException(status_code=400, detail=f'{name} value "{value}" is invalid')
