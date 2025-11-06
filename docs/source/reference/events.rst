.. _events_sec:

Events
======

All events have a starting time (onset) and an ending time (offset, inclusive).


.. currentmodule:: pymovements

.. rubric:: Classes

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: class.rst

    Events
    EventDataFrame

.. rubric:: Processing

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: class.rst

    EventProcessor
    EventGazeProcessor

.. currentmodule:: pymovements.events.detection

.. rubric:: Detection Methods

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    idt
    ivt
    microsaccades
    fill

.. currentmodule:: pymovements.events.properties

.. rubric:: Event Properties

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    amplitude
    duration
    dispersion
    disposition
    peak_velocity
    location
