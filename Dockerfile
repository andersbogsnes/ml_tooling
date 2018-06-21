FROM python:3.6
LABEL maintainer="abanbn@almbrand.dk"

ARG http_proxy
ARG https_proxy

ENV no_proxy="localhost,127.0.0.1,.alm.brand.dk,.almbrand.dk"

COPY httpproxy.crt /usr/local/share/ca-certificates/

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --cert /usr/local/share/ca-certificates/httpproxy.crt --no-cache-dir -r requirements.txt

RUN mkdir ./app
COPY . ./app

WORKDIR ./app

RUN pip install --cert /usr/local/share/ca-certificates/httpproxy.crt --no-cache-dir .

CMD ["bin/bash"]
