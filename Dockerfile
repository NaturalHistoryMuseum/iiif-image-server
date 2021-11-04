FROM ubuntu:bionic

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get -q -y update && \
    apt-get -q -y upgrade && \
    apt-get -q -y install software-properties-common && \
    add-apt-repository universe && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -q -y update && \
    apt-get -q -y install \
        # install system packages required by ckan
        python3.8 \
        python3.8-venv \
        python3.8-dev \
        libffi-dev \
        libturbojpeg0-dev \
        libjpeg-dev \
        libpcre3 \
        libpcre3-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        build-essential \
        libmagickwand-dev \
        # install packages for additional bits and bobs (like the entrypoint script)
        wget \
        git \
        curl \
        netcat \
    && apt-get -q clean \
    && rm -rf /var/lib/apt/lists/*

# we're gonna use a venv for a couple of reasons. Firstly, it avoids any OS level python package
# clashes/dependencies. Secondly, it matches how we run stuff on the live servers. And finally, it
# means we can use a different version of python3 to the one shipped with ubuntu 18.04.
# For info on the next 3 lines: https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -U pip && pip install wheel

RUN mkdir -p /base/cache && mkdir -p /base/source && mkdir -p /base/source/example
COPY docker/start_iiif_server.sh /base/start_iiif_server.sh
COPY docker/example_config.yml /base/example_config.yml
COPY docker/example_sources /base/source/example/
COPY . /base/src/

WORKDIR /base/src/

# setup the IIIF image server
RUN pip install -e .
RUN pip install -e .[test]

WORKDIR /base

# start the server
CMD ["/base/start_iiif_server.sh"]
