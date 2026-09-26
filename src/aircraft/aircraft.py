"""
Aircraft selection: ONE place to choose which parameter file the 6-DOF tools use.

    from aircraft import p          # use the parameters (p.m, p.CL_alpha, ...)
    import aircraft
    aircraft.load("c182")           # switch aircraft (all 6-DOF modules follow)

Every module works with the same object `p`; load() only replaces its content.
Available: c172, c182  (files <name>_params.py)
"""
import importlib
import types

DEFAULT = "c172"
p = types.SimpleNamespace()


def load(name):
    """Load <name>_params.py into the shared parameter object p."""
    module = importlib.import_module(f"{name}_params")
    p.__dict__.clear()
    p.__dict__.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    p.AIRCRAFT = name
    return p


load(DEFAULT)