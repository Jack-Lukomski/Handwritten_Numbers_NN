FROM ubuntu:18.04
LABEL Description="NeuralC++ Build Env"

ENV HOME /root

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    clang \
    cmake \
    gdb \
    libopenblas-dev \
    liblapack-dev \
    libarmadillo-dev

WORKDIR /root/neuralcpp

VOLUME /root/neuralcpp