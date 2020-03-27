#!/usr/bin/env python3
# encoding: utf-8

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()


# at the time of writing, the latest version of this lib on pypi (0.5.2) doesn't install correctly,
# whereas the latest commit on master does, hence we'll just install directly from github
jpegtran_url = 'git+https://github.com/jbaiter/jpegtran-cffi.git@70928eb#egg=jpegtran-cffi'

setup(
    name="iiif-image-server",
    version='0.0.1',
    author='Josh Humphries',
    author_email='data@nhm.ac.uk',
    description="A simple IIIF server that simply works for the V-Factor beetle drawer pilot",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/NaturalHistoryMuseum/iiif-image-server/",
    packages=find_packages(),
    install_requires=[
        'flask~=1.1.1',
        'Flask_Cors~=3.0.8',
        'pyyaml~=5.3.1',
        'uWSGI~=2.0.18',
        f'jpegtran-cffi @ {jpegtran_url}',
    ],
    dependency_links=[
        jpegtran_url,
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)