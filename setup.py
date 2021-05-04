#!/usr/bin/env python3
# encoding: utf-8

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

# at the time of writing, the latest version of this lib on pypi (0.5.2) doesn't install correctly,
# whereas the latest commit on master does, hence we'll just install directly from github
jpegtran_url = 'git+https://github.com/jbaiter/jpegtran-cffi.git@70928eb#egg=jpegtran-cffi'

install_dependencies = [
    'aiocron~=1.4',
    'aiofiles~=0.6.0',
    'aiohttp[speedups]~=3.7.4',
    'aiozipstream~=0.4',
    'cffi~=1.14.5',
    'elasticsearch-dsl>=6.0.0,<7.0.0',
    'fastapi~=0.63.0',
    'humanize~=3.4.1',
    f'jpegtran-cffi @ {jpegtran_url}',
    'lru-dict~=1.1.7',
    'orjson~=3.5.2',
    'pillow~=8.2.0',
    'pyyaml~=5.4.1',
    'uvicorn[standard]~=0.13.4',
]
test_dependencies = [
    'pytest~=5.4.1',
    'pytest-asyncio~=0.11.0',
    'pytest-cov~=2.11.1',
    'requests~=2.25.1',
]

setup(
    name="iiif-image-server",
    version='1.0.0',
    author='Natural History Museum',
    author_email='data@nhm.ac.uk',
    description="A simple IIIF server that simply works for the V-Factor beetle drawer pilot",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/NaturalHistoryMuseum/iiif-image-server/",
    packages=find_packages(),
    install_requires=install_dependencies,
    dependency_links=[jpegtran_url],
    extras_require={
        'test': test_dependencies,
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
