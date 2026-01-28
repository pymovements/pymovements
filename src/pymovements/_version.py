"""Module for inferring pymovements version."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pymovements")
except PackageNotFoundError as e:
    e.add_note('Please install pymovements with your package manager')
    raise
