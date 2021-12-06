configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_custom_target(check COMMAND llvm-lit "${CMAKE_CURRENT_BINARY_DIR}" -v
                  DEPENDS hail-opt)
