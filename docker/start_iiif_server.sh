#!/bin/bash
set -e

export IIIF_CONFIG=/base/example_config.yml
uvicorn iiif.web:app --port 4040 --host 0.0.0.0 --workers 1 --root-path /media
