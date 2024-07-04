# OpenMP + Atlas Example

The example demostrates using Atlas' GPU memory management functionality in conjunction with OpenMP offloading (via the GCC compiler).

## Development Environment
- Install the CUDA Toolkit.
- Install offload capable GCC compiler.
- Check that the Atlas and ECKIT paths in the makefile make sense depending on how you have installed these dependencies.
- Ensure that if Atlas is locally installed the path to the Atlas library is added to LD_LIBRARY_PATH.
