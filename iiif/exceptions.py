from fastapi import HTTPException


def profile_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail='Profile not recognised')


def image_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail='Image not found')


def too_many_images(max_files):
    return HTTPException(status_code=400, detail=f'Too many images requested (max: {max_files})')
