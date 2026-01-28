"""Module for inferring pymovements version."""
import warning
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pymovements")
except PackageNotFoundError as exception:
    exception.add_note(
        'Inferring pymovements version failed. '
        'Please install pymovements with your package manager.',
    )
    warning = RuntimeWarning(*exception.args)
    warning.with_traceback(exception.__traceback__)
    __version__ = 'unknown'
