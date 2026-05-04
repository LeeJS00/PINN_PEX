# Empty marker file. See pex_v3/docs/CROSS_BOUNDARY_legacy_src_init.md.
#
# Why this exists: Strategy v3 reuses legacy DeepPEX_Model via
# pex_v3/src/baselines/pinn_baseline.py. When pinn_baseline imports
# legacy `run_active_learning`, torch's compile/library introspection
# walks the call stack and tries to resolve the source file of every
# module referenced from the inspected frames. With legacy `/src/` as
# a NAMESPACE package (no __init__.py), torch raises
# "TypeError: <module 'src' ...> is a built-in module" because
# namespace packages have no single source file.
#
# This empty __init__.py makes /src/ a regular package, fixing the
# introspection. No behavioral impact for any existing legacy caller.
