## v1.0.1 (2025-08-11)

### Fix

- use a version of setuptools that supports PEP639
- catch specific timeout error

### Docs

- add docs configs

### Style

- auto reformat python files

### Tests

- move test utils into helpers folder

### Build System(s)

- exclude docs folder
- convert to pyproject

### CI System(s)

- update github workflows

### Chores/Misc

- fix license specifier
- use basic string license specifier
- fix file endings
- add PR templates, contributing guidelines, etc
- add pre-commit config

## v1.0.0 (2025-04-19)

### Breaking Changes

- unpack splitgill encoded data
- MSS compaibility with vds and splitgill updates

### Feature

- unpack splitgill encoded data
- MSS compaibility with vds and splitgill updates

### Fix

- alter search to work with new splitgill version

### Tests

- fix tests
- fix tests with newer versions of pytest / pytest-asyncio

### Chores/Misc

- add .venv path to gitignore

## v0.16.3 (2024-11-25)

### Fix

- decrease total response timeout to 5s for mss status check

## v0.16.2 (2024-07-22)

### Fix

- make sure the image is loaded before rotating it based on exif
- rotate images on source acquisition

## v0.16.1 (2024-07-01)

### Fix

- define cause variable before using it in exception

## v0.16.0 (2023-12-11)

### Feature

- add store status to profile status
- convert non-jpeg images to jpeg before serving

### Fix

- make disk store path relative to root
- replace imghdr with filetype
- convert images to rgb before saving as jpg

### Refactor

- change 'source' to 'converted' for clarification

### Docs

- fix workflow badge
- update config option docs

### Tests

- check the disk status is returning the correct info
- exclude abstract methods from coverage report
- check that the cause is returned correctly
- add a test to check size is returned correctly
- add tests for the disk store specifically
- add tests to confirm that disk profile always returns jpg paths

### Chores/Misc

- add bump and sync workflows
- add commitizen config to pyproject.toml

## v0.15.0 (2023-11-16)

### Feature

- remove cors headers from responses

### Fix

- remove image magick from jpg conversion code

### Tests

- use an up to date runner to allow test action to run

### Build System(s)

- update version to v0.15.0

## v0.14.1 (2022-07-04)

## v0.14 (2022-07-04)

## v0.13 (2022-05-23)

## v0.12.1 (2022-05-16)

## v0.12.0 (2022-05-16)

## v0.11.0 (2022-01-11)

## v0.10.4 (2021-12-13)

## v0.10.3 (2021-12-08)

## v0.10.2 (2021-11-30)

## v0.10.1 (2021-11-18)

## v0.10.0 (2021-11-04)

## v0.9.1 (2021-10-13)

## v0.9.0 (2021-10-12)
