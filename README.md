# IIIF Image Server
[![Travis](https://img.shields.io/travis/NaturalHistoryMuseum/iiif-image-server/master.svg?style=flat-square)](https://travis-ci.org/NaturalHistoryMuseum/iiif-image-server)

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

See the config section for more information about configuring identifier types and sources.

## Config

| Name | Description | Example |
|----|-----------|-------|
| `base_url` | The base URL to use in `info.json` responses. This should be the base URL for this server. | `http://10.11.20.12/iiif_images` |
| `http_port` | The HTTP port to listen on | 4040 |
| `cache_path` | The full path where derivative images created as part of data requests should be cached | `/var/lib/iiif_image_server/cache` |
| `source_path` | The full path where source images should be found/stored | `/var/lib/iiif_image_server/source` |
| `min_sizes_size` | The minimum size that should be returned as part of the `sizes` array in `info.json` responses | 200 |
| `size_pool_size` | The number of processes to use in the process pool that extracts size information about images for use in `info.json` responses | 2 |
| `image_pool_size` | The number of processes to use in the process pool that performs image manipulations to meet the needs of data requests | 4 |
| `info_cache_size` | The size of the LRU cache that stores `info.json` responses | 1024 |
| `image_cache_size_per_process` | The size of the LRU cache that stores source image data, per image pool process. Source images can be very large so this value should be considered with the size of images in memory and the value of the `image_pool_size` config option | 5 |
| `max_http_fetches` | The maximum number of image source HTTP(S) requests that can be active at a time. This is not a limit on the number of clients to the server, it is a limit for images that need to be fetched from a web source before they can used by the server (i.e. `web` or `trusted_web` source types) | 10 |
| `types` | Defines the supported types on this server, see the section below for details | {} |

### Types
The types system allows the server to serve up images from multiple different sources whilst
maintaining control.
Only types defined in this section of the config will be supported by the server.

Each type has a unique name which is used in the identifier (see `Identifiers` above), a source
value which describes how to get hold of the source image and then some extra source-specific
options.

#### Source types
There are currently 3 source types which are described next.

##### disk
Use this source when the images are on disk already, in the `source_path`.

Example types config:

```yaml
types:
  project1:
    source: disk
```

Under this example config, all images should be stored under `{source_path}/project1/`.
The names of the files must match the name part of the identifier, i.e. a request for
`project1:banana` will look for a source file at `{source_path}/project1/banana` and fail if it is
not found.

##### web
Use this source when the images are stored somewhere online.
The server will retrieve the source image when it is requested and store it in the source path for
all subsequent requests to use.

Example types config:

```yaml
types:
  project2:
    source: web
    url: http://example.com/storage/images/{name}.jpg
```

Under this example config, when a request is made the name part of the identifier is used to
complete the URL specified in the config and then fetch the image.
The image is stored under `{source_path}/project2/`.
The names of the files must match the name part of the identifier, i.e. a request for
`project2:banana` will result in the image being downloaded and stored at
`{source_path}/project2/banana`.

The URL may contain `{name}` which will be replaced by the identifier's name before being fetched.

##### trusted_web
Use this source when the images are stored somewhere online but the URL they are to be fetched from
cannot be generated simply from the identifier.
This essentially allows requests that include the URL as the name part of the identifier, but
prevents arbitrary URLs being passed and fetched by ensuring they match a regex first.
This provides some level of control over which URLs we fetch on user requests.

The name part of the identifier must be a web-safe base 64 encoded utf-8 string, i.e. one that you'd
get from the `base64.urlsafe_b64encode` Python function:

The server will retrieve the source image when it is requested and store it in the source path for
all subsequent requests to use.

Example types config:

```yaml
types:
  project3:
    source: web
    regex: https://raw.githubusercontent.com/NaturalHistoryMuseum/.+
```

Under this example config, when a request is made the name part of the identifier is decoded and
matched against the regex.
If it matches then the data from the URL is downloaded and stored under the encoded name part of the
identifier.
For example:
```
    Source URL: https://raw.githubusercontent.com/NaturalHistoryMuseum/tests/test.jpg
    identifier: project3:aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL05hdHVyYWxIaXN0b3J5TXVzZXVtL3Rlc3RzL3Rlc3QuanBn
    Source storage path: `{source_path}/project3/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL05hdHVyYWxIaXN0b3J5TXVzZXVtL3Rlc3RzL3Rlc3QuanBn`
```


## Example
See [https://data.nhm.ac.uk/vfactor_iiif](https://data.nhm.ac.uk/vfactor_iiif).
