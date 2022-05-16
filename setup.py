#!/usr/bin/env python3
# encoding: utf-8

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

# at the time of writing, the latest version of this lib on pypi (0.5.2) doesn't install correctly,
# whereas the latest commit on master does, hence we'll just install directly from github
jpegtran_url = 'git+https://github.com/jbaiter/jpegtran-cffi.git@70928eb#egg=jpegtran-cffi'

install_dependencies = [
    'aiofiles==0.8.0',
    'aiohttp[speedups]==3.8.1',
    'aiozipstream==0.4',
    'cachetools==5.0.0',
    'cffi==1.15.0',
    'elasticsearch-dsl==6.4.0',
    'fastapi==0.77.1',
    'humanize==4.1.0',
    f'jpegtran-cffi @ {jpegtran_url}',
    'pillow==9.1.0',
    'pyyaml==6.0',
    'uvicorn[standard]==0.17.6',
    'wand==0.6.7',
]
test_dependencies = [
    'pytest',
    'pytest-asyncio',
    'pytest-cov',
    'aioresponses',
    # needed by starlette's test client
    'requests',
]

setup(
    name="iiif-image-server",
    version='0.10.0',
    author='Natural History Museum',
    author_email='data@nhm.ac.uk',
    description="A media server primarily used by the NHM Data Portal",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
