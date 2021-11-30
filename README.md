# IIIF Image Server
[![Actions](https://img.shields.io/github/workflow/status/NaturalHistoryMuseum/iiif-image-server/Tests?style=flat-square)](https://github.com/NaturalHistoryMuseum/iiif-image-server/actions)
[![Coveralls](https://img.shields.io/coveralls/github/NaturalHistoryMuseum/iiif-image-server/main.svg?style=flat-square)](https://coveralls.io/github/NaturalHistoryMuseum/iiif-image-server)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)

This is a IIIF image server primarily used by the [NHM Data Portal](https://data.nhm.ac.uk).
The service uses FastAPI and asynchronous code throughout.

## OS Dependencies
See the `Dockerfile` for dependency install example on Ubuntu 18.04, generally though we need:

    - python3.8 (probably works on 3.6+)
    - cffi
    - libjpeg8
    - libcurl

## Architecture
This server is written using Python's asyncio framework and FastAPI.
Image data requests start by being handled by FastAPI but if any image processing is required to
fulfill a request then this occurs in a separate process to avoid holding up the whole server.
Python's multiprocessing libraries are used to facilitate this with image operations queued and
handled by a fixed size pool of workers.
This approach ensures we can keep control of resource usage (i.e. RAM and CPU).

## jpegtran-cffi
[https://github.com/jbaiter/jpegtran-cffi](https://github.com/jbaiter/jpegtran-cffi) is used to work
with JPEG images efficiently.
It provides rapid operations on JPEG images by using the JPEG turbo C libs to do the heavy lifting
which can scale images without re-compressing them and uses SIMD as well.

## Identifiers
Identifiers are expected in two parts: `<profile>:<name>`.
This format is intended support two pieces of functionality:
    - it allows the server to handle requests for images from a variety of sources. For example,
      `abc:1` maybe be an on disk image whereas `xyz:1` maybe an image on the web that must be
      downloaded before it can be processed
    - it namespaces the names so that they only have to be unique within their profile

See the config section for more information about configuring profiles.

## Config

All configuration options are required, there are no defaults.

| Name | Description | Default |
|----|-----------|-------|
| `base_url` | The base URL to use in `info.json` responses. This should be the base URL for this server. | `http://10.11.20.12/iiif_images` |
| `cache_path` | The full path where derivative images created as part of data requests should be cached | `/var/lib/iiif_image_server/cache` |
| `source_path` | The full path where source images should be found/stored (depending on the profile) | `/var/lib/iiif_image_server/source` |
| `min_sizes_size` | The minimum size that should be returned as part of the `sizes` array in `info.json` responses | 200 |
| `image_pool_size` | The number of processes to use in the process pool that performs image manipulations to meet the needs of data requests | 2 |
| `image_cache_size_per_process` | The size of the LRU cache that stores source image data, per image pool process. Source images can be very large so this value should be considered with the size of images in memory and the value of the `image_pool_size` config option | 5 |
| `thumbnail_width` | Maximum width of thumbnail images served by the /thumbnail endpoint (for maximum efficiency, make sure this is a multiple of 16) | 512 |
| `preview_width` | Maximum width of preview images served by the /preview endpoint (for maximum efficiency, make sure this is a multiple of 16) | 2048 |
| `download_chunk_size` | Number of bytes to stream of original images at a time. Don't make this too large or the server will lock up as while the chunk is read the asyncio loop is blocked | 4096 |
| `download_max_files` | Maximum number of files to allow in a download zip | 20 |
| `default_profile` | The name of the profile which should be used as the default if no profile name is included in an identifier, for example `/abc:1` uses the profile named `abc`, if `default_profile: abc` then `/1` will also use `abc`. | 'mss' |
| `profiles` | Defines the supported profiles on this server, see the section below for details | {} |

### Profiles
The profiles system allows the server to serve up images from multiple different sources whilst
maintaining control.
Only profiles defined in this section of the config will be supported by the server.

Each profile has a unique name which is used in the identifier (see `Identifiers` above), a type
value which describes indicates the base type of the profile (e.g. `disk`), and then some extra
source-specific options.

#### Profile Types
There are currently 2 types of profile which are described next.

All profile types have some common configuration options:

| Name | Description | Default |
|----|-----------|-------|
| `rights` | The rights string to include in `info.json` responses | n/a, must be defined by the user |
| `info_json_cache_size` | The number of `info.json` responses to cache for this profile | 1000 |
| `log_level` | The log level for any logging produced from this profile | 'WARNING' |
| `cache_for` | Value to set in the `cache-control` header passed back with any image data responses from this profile. The value is appended to the string `"max-age="` and therefore the easiest way to set this value is as the number of seconds to cache the image for in seconds | 0 |

##### disk
Use this source when the images are on disk already, in the `source_path`.

Example types config:

```yaml
...
profiles:
  project1:
    type: disk
    ...
```

Under this example config, all images should be stored under `{source_path}/project1/`.
The names of the files must match the name part of the identifier, i.e. a request for
`project1:banana` will look for a source file at `{source_path}/project1/banana` and
fail if it is not found.

###### Options
There are no additional `disk` specific options.

##### mss
This profile type is only usable by internal NHM systems on the Data Portal (and hence
needs to be moved into like a plugin or something out of this repo...).
Use this source when the images are stored in the NHM's MSS.
The server will retrieve the source image from the MSS when it is requested and store
it in the source path for all subsequent requests to use.

Example types config:

```yaml
...
profiles:
  project2:
    type: mss
    url: http://example.com/storage/images/{name}.jpg
```

###### Options
See https://github.com/NaturalHistoryMuseum/iiif-image-server/blob/main/iiif/profiles/mss.py#L64 for the MSS profile options.


## Example
See [https://data.nhm.ac.uk/vfactor_iiif](https://data.nhm.ac.uk/vfactor_iiif).
