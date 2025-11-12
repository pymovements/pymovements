Gaze
====

:py:class:`~pymovements.Gaze` class is a self-contained data structure that contains eye tracking
data represented as samples or events. It also includes metadata on the experiment and recording
setup.

.. currentmodule:: pymovements

.. rubric:: Classes

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: class.rst

    Gaze
    GazeDataFrame

.. currentmodule:: pymovements.gaze.transforms

.. rubric:: Transformations

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    center_origin
    downsample
    norm
    pix2deg
    deg2pix
    pos2acc
    pos2vel
    savitzky_golay

.. currentmodule:: pymovements.gaze

.. rubric:: Input / Output
    :name: gaze-io

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    from_asc
    from_csv
    from_ipc

.. rubric:: Integration

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    from_numpy
    from_pandas

.. currentmodule:: pymovements.gaze.transforms_numpy

.. rubric:: Numpy Transformations

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    pix2deg
    pos2acc
    pos2vel
    norm
    split
    downsample
    consecutive
