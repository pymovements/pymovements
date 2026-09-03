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

.. currentmodule:: pymovements.events.detection

.. rubric:: Detection Methods

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    idt
    ivt
    ihmm
    microsaccades
    microsaccades.compute_threshold
    blink
    fill

.. currentmodule:: pymovements.events.correction

.. rubric:: Fixation Correction Methods
    :name: fixation-correction-methods

These functions can be used to apply a line-alignment correction algorithm to a set of fixations.
The algorithms will adjust the y-coordinates of the fixations to correct for drift and systematic
error in the eye-tracking data. This is particularly useful for paragraph reading data, where
y-alignment issues can lead to a fixation being assigned to the wrong line of text. Available
algorithms are listed under :ref:`Drift Correction Algorithms <drift-correction-algorithms>`.
The most convenient way to correct fixations is via the
:py:meth:`~pymovements.Events.correct_fixations` and
:py:meth:`~pymovements.Dataset.correct_fixations` methods.

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    correct_fixations
    correct_fixation_locations

.. currentmodule:: pymovements.events.correction.drift_algorithms

.. rubric:: Drift Correction Algorithms
    :name: drift-correction-algorithms

The following algorithms can be used to apply a line-alignment correction algorithm to a set of
fixations. These algorithms, which are described in detail by Carr et al. :cite:p:`Carr2022`,
can be applied to the fixations using the
:func:`~pymovements.events.correction.correct_fixations` function. By default, the
:func:`~pymovements.events.correction.correct_fixations` function will use the
:func:`~pymovements.events.correction.drift_algorithms.wisdom_of_the_crowd` algorithm which is an
ensemble method that combines the results of the other algorithms to produce a more robust
correction. Further drift correction approaches from the literature that are not yet available
in pymovements include the Dual Input Stream Transformer :cite:p:`Mercier2024a` and adaptive
algorithm selection based on eye movement patterns :cite:p:`AlMadi2025`.

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    attach
    chain
    cluster
    compare
    merge
    regress
    segment
    slice
    split
    stretch
    warp
    wisdom_of_the_crowd
