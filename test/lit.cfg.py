import lit.formats

config.name = "Hail MLIR Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.hail_bin_root, 'test')

config.substitutions.append(
    ('hail-opt', os.path.join(config.hail_bin_root, 'bin', 'hail-opt'))
)
