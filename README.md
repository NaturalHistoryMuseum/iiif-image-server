# IIIF Image Server

This is a IIIF image server designed to work under specific conditions in a pilot being run by the
Informatics team at the Natural History Museum.

Probably best not to use it for anything else at the moment as it has some hard coded assumptions
based on locations of NHM assets and cannot be used generically yet.

## OS Dependencies
    - python3.8 (probably works on 3.6+)
    - cffi
    - libjpeg8

## Architecture
This server is written using Python's asyncio framework and the Tornado web server.
Image data requests are handled through Tornado but if any image processing is required to fulfill a
request then this occurs in a separate process to avoid holding up the whole server.
Python's multiprocessing libraries are used to facilitate this with image operations queued and
handled by a fixed size pool of workers.
This approach ensures we can keep control of resource usage (i.e. RAM and CPU).

## jpegtran-cffi
[https://github.com/jbaiter/jpegtran-cffi](https://github.com/jbaiter/jpegtran-cffi) is used to work
with JPEG images efficiently.
It provides rapid operations on JPEG images by using a C libraries to do the heavy lifting and not
re-compressing the images it scales.

## Identifiers
Identifiers are expected in two parts: `<type>:<identifier>`.
This format is intended support two pieces of functionality:
    - it allows the server to handle requests for images from a variety of sources. For example,
      `abc:1` maybe be an on disk image whereas `xyz:1` maybe an image on the web that must be
      downloaded before it can be processed)
    - it discourages users just specifying a URL as the identifier which would be open to abuse

## Example
See [https://data.nhm.ac.uk/vfactor_iiif](https://data.nhm.ac.uk/vfactor_iiif).
