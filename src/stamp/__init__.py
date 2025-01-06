from importlib.metadata import version

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

# Warn about all incorrect type annotations
beartype_this_package()  # conf=BeartypeConf(violation_type=UserWarning))

__version__: str = version("stamp")
