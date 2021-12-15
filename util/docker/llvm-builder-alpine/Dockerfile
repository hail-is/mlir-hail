FROM alpine:latest

RUN apk add --no-cache \
    bash \
    binutils-dev \
    build-base \
    clang \
    cmake \
    curl \
    git \
    libc-dev \
    libffi-dev \
    libxml2-dev \
    lld \
    llvm \
    musl-dev \
    ninja \
    py3-numpy-dev \
    py3-pybind11-dev \
    py3-setuptools \
    python3 \
    python3-dev \
    xz \
    zlib-dev

# I want to be able to use packaged sccache but it's not in the stable repos yet
ADD https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz \
    /var/tmp/

RUN tar -xvzf /var/tmp/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz \
        sccache-v0.2.15-x86_64-unknown-linux-musl/sccache \
    && mv /sccache-v0.2.15-x86_64-unknown-linux-musl/sccache /usr/bin/ \
    && chmod +x /usr/bin/sccache \
    && rmdir /sccache-v0.2.15-x86_64-unknown-linux-musl
