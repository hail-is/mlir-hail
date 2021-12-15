#!/usr/bin/env python3
import argparse
import os

from collections import namedtuple

from hailtop.batch import Batch, ServiceBackend

LLVM_VERSION = '13.0.0'
SOURCE_URL = f'https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/llvm-project-{LLVM_VERSION}.src.tar.xz'
SOURCE_TAR_FILENAME = f'llvm-project-{LLVM_VERSION}.src.tar.xz'
SOURCE_SHA512_SUMS = f'75dc3df8d1229c9dce05206e9a2cab36be334e0332fe5b2a7fc01657a012a630e19f2ade84139c0124865cff5d04952f8c6ef72550d74350e8c6bca0af537ad3  {SOURCE_TAR_FILENAME}'
SOURCE_ROOT_DIRNAME = f'llvm-project-{LLVM_VERSION}.src'
CMAKE_VARIABLES = {
    'CMAKE_BUILD_TYPE': 'Release',
    'CMAKE_INSTALL_PREFIX': '/opt/llvm/',
    'CMAKE_C_COMPILER': 'clang',
    'CMAKE_CXX_COMPILER': 'clang++',
    'CMAKE_C_COMPILER_LAUNCHER': 'sccache',
    'CMAKE_CXX_COMPILER_LAUNCHER': 'sccache',
    'LLVM_ENABLE_LLD': 'ON',
    'LLVM_BUILD_EXAMPLES': 'OFF',
    'LLVM_BUILD_TESTS': 'OFF',
    'LLVM_ENABLE_PROJECTS': "'mlir;clang;lld'",
    'LLVM_ENABLE_ASSERTIONS': 'OFF',
    'LLVM_ENABLE_FFI': 'ON',
    'LLVM_ENABLE_LIBCXX': 'OFF',
    'LLVM_ENABLE_PIC': 'ON',
    'LLVM_ENABLE_RTTI': 'ON',
    'LLVM_ENABLE_SPHINX': 'OFF',
    'LLVM_ENABLE_TERMINFO': 'ON',
    'LLVM_BUILD_LLVM_DYLIB': 'ON',
    'LLVM_LINK_LLVM_DYLIB': 'ON',
    'LLVM_INSTALL_UTILS': 'ON',
    'LLVM_ENABLE_ZLIB': 'ON',
    'MLIR_ENABLE_BINDINGS_PYTHON': 'ON',
}

BuildInfo = namedtuple('BuildInfo', 'name image triple')

BUILDERS = (
    BuildInfo(name='alpine',
              image='gcr.io/hail-vdc/mlir-hail-llvmbuilder:alpine',
              triple='x86_64-unknown-linux-musl'),
    BuildInfo(name='debian',
              image='gcr.io/hail-vdc/mlir-hail-llvmbuilder:debian',
              triple='x86_64-unknown-linux-gnu'),
)


def compile_llvm(batch: Batch, info: BuildInfo, output_dir: str):
    job = batch.new_bash_job(name=f'{info.name} {info.triple}')
    job.image(info.image)
    # this build has thousands of components, we want it to finish today please
    job.cpu(16)
    job.memory('standard')
    job.storage('100Gi')
    job.env('SRCDIR', f'/io/src/{SOURCE_ROOT_DIRNAME}')
    job.env('SCCACHE_GCS_BUCKET', 'hail-sccache')
    job.env('SCCACHE_GCS_KEY_PATH', '/gsa-key/key.json')
    job.env('SCCACHE_GCS_RW_MODE', 'READ_WRITE')

    # download & check
    job.command('mkdir -p $SRCDIR')
    job.command('cd ${BATCH_TEMPDIR}')
    job.command(f'curl -LfO {SOURCE_URL}')
    job.command(f"echo '{SOURCE_SHA512_SUMS}' > {job.checksum_file}")
    job.command(f'sha512sum -c {job.checksum_file}')
    job.command(f'unxz -c {SOURCE_TAR_FILENAME} | tar -C /io/src -x')

    # setup
    job.command('mkdir -p $SRCDIR/build')
    job.command('cd $SRCDIR/build')
    cmake_command = 'cmake ../llvm -Wno-dev -GNinja ' \
        + f'-DLLVM_DEFAULT_TARGET_TRIPLE={info.triple} ' \
        + ' '.join(f'-D{key}={value}' for key, value in CMAKE_VARIABLES.items())
    job.command(cmake_command)

    # build
    job.command('ninja')
    job.command('python3 ../llvm/utils/lit/setup.py build')

    # install
    job.command('DESTDIR="$PWD/pkg" ninja install')
    # NOTE this prefix may need to change depending on platform, but hardcoding it as /usr for now
    job.command(
        'python3 ../llvm/utils/lit/setup.py install --prefix=/usr --root="$PWD/pkg"'
    )
    job.command('ln -s ../../../usr/bin/lit "$PWD/pkg/opt/llvm/bin/llvm-lit"')
    job.command('cd pkg')
    job.command(f'tar cf - opt usr | xz -T0 -9 -cv > {job.ofile}')
    output_file = os.path.join(
        output_dir,
        f'llvm_all-backends_{info.name}_{info.triple}_{LLVM_VERSION}.tar.xz')
    batch.write_output(job.ofile, output_file)


def main():
    parser = argparse.ArgumentParser('batch-build-llvm')
    parser.add_argument('--billing-project',
                        help='batch billing project to use',
                        type=str,
                        default='hail')
    parser.add_argument('--remote-tmpdir',
                        help='batch remote_tmpdir to use',
                        type=str,
                        default='gs://7-day')
    parser.add_argument('--output-dir',
                        help='where to put the packaged artifact',
                        type=str,
                        default='gs://cdv-hail/llvm-pkg')
    args = parser.parse_args()

    with ServiceBackend(billing_project=args.billing_project,
                        remote_tmpdir=args.remote_tmpdir) as backend:
        batch = Batch(backend=backend, name=f'build llvm-{LLVM_VERSION}')
        for info in BUILDERS:
            compile_llvm(batch, info, args.output_dir)
        batch.run()


if __name__ == '__main__':
    main()
