#!/usr/bin/env python3
# encoding: utf-8
import hashlib
import io
import os

import yaml
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from jpegtran import JPEGImage

root_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(root_dir, 'config.yml'), 'r') as cf:
    config = yaml.safe_load(cf)
cache_dir = os.path.join(root_dir, config['cache_dir'])
source_dir = os.path.join(root_dir, config['source_dir'])
app = Flask(__name__)
CORS(app)


def get_image(image_name):
    return JPEGImage(os.path.join(source_dir, image_name))


@app.route('/<image_name>/<region>/<size>/0/default.jpg')
def image_data(image_name, region, size):
    cache_name = hashlib.sha1(f'{image_name}_{region}_{size}'.encode('utf-8')).hexdigest()
    path = os.path.join(cache_dir, f'{cache_name}.jpg')
    add_to_cache = False
    if os.path.exists(path):
        image = JPEGImage(path)
    else:
        image = get_image(image_name)
        add_to_cache = True

        if region != 'full':
            x, y, w, h = map(int, region.split(','))
            image = image.crop(x, y, w, h)

        if size != 'max':
            image_width = image.width
            image_height = image.height
            w, h = (float(v) if v != '' else v for v in size.split(','))
            if h == '':
                h = image_height * w / image_width
            elif w == '':
                w = image_width * h / image_height
            image = image.downscale(int(w), int(h))

    data = image.as_blob()
    if add_to_cache:
        with open(path, 'wb') as f:
            f.write(data)

    return send_file(io.BytesIO(data), mimetype='image/jpeg')


@app.route('/<image_name>/info.json')
def image_info(image_name):
    image = get_image(image_name)
    width, height = image.width, image.height

    return jsonify({
        '@context': 'http://iiif.io/api/image/3/context.json',
        # mirador/openseadragon seems to need this to work even though I don't think it's correct
        # under the IIIF image API v3
        '@id': '{}/{}'.format(config['base_url'], image_name),
        'id': '{}/{}'.format(config['base_url'], image_name),
        'protocol': 'http://iiif.io/api/image',
        'width': width,
        'height': height,
        'rights': 'http://creativecommons.org/licenses/by/4.0/',
        'profile': 'level0',
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4040)
