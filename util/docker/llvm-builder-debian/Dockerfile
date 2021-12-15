FROM debian:latest

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        clang \
        cmake \
        curl \
        git \
        lld \
        llvm \
        ninja-build \
        pybind11-dev \
        python3-dev \
        python3-numpy-dev \
        python3-setuptools \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# install sccache (it's not in any debian repo yet)
ADD https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz \
    /var/tmp/

RUN tar -xvzf /var/tmp/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz \
        sccache-v0.2.15-x86_64-unknown-linux-musl/sccache \
    && mv /sccache-v0.2.15-x86_64-unknown-linux-musl/sccache /usr/bin/ \
    && chmod +x /usr/bin/sccache \
    && rmdir /sccache-v0.2.15-x86_64-unknown-linux-musl
