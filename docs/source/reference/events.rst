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

.. currentmodule:: pymovements

.. rubric:: Fixation Drift Correction
    :name: fixation-drift-correction

These functions can be used to apply a line-alignment correction algorithm to a set of fixations.
The algorithms will adjust the y-coordinates of the fixations to correct for drift and systematic
error in the eye-tracking data. This is particularly useful for paragraph reading data, where
y-alignment issues can lead to a fixation being assigned to the wrong line of text. Fixation
drift correction assumes reading data recorded on a
:py:class:`~pymovements.stimulus.TextStimulus`, whose areas of interest provide the text line
positions. Available
algorithms are listed under :ref:`Drift Correction Algorithms <drift-correction-algorithms>`.
The most convenient way to correct fixations is via the
:py:meth:`Events.correct_fixations` and
:py:meth:`Dataset.correct_fixations` methods.

.. currentmodule:: pymovements.events.correction

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    correct_fixations
    correct_fixation_locations

.. currentmodule:: pymovements

.. rubric:: Trial Correctors
    :name: trial-correctors

The algorithms listed below are functions returning a polars expression over a column. That
shape carries no state, so a corrector that has to hold something -- geometry it precomputed,
a parameter it estimated per reader -- cannot be written as one: the expression is built inside
the per-trial path, so whatever it prepares is prepared again for every trial, and it receives
the location column rather than the trial's areas of interest.

For those, ``algorithm=`` also takes a **trial corrector**: anything callable as
``corrector(fixations, aois, location_column=...)``, returning one ``[x, y]`` list per fixation
or ``None`` to decline the trial. A plain function is one, and so is an object holding state in
``__call__``; the object is built once per
:py:func:`~pymovements.events.correction.correct_fixations` call rather than once per trial.
Pass it directly, or register it under a name with :py:func:`register_corrector` and select it
the way you select ``'warp'``.

Declining a trial behaves as a drift algorithm skipping one: a warning naming the trial, and
its fixations stay uncorrected. A trial corrector cannot take part in a Wisdom of the Crowd
ensemble, because its vote has no obvious weight against the algorithmic ones.

.. currentmodule:: pymovements

.. rubric:: Drift Correction Algorithms
    :name: drift-correction-algorithms

The following algorithms can be used to apply a line-alignment correction algorithm to a set of
fixations. These algorithms, which are described in detail by Carr et al. :cite:p:`Carr2022`,
can be applied to the fixations using the
:func:`~pymovements.events.correction.correct_fixations` function. By default, the
:func:`~pymovements.events.correction.correct_fixations` function will use the
:func:`~pymovements.events.correction.wisdom_of_the_crowd` algorithm which is an
ensemble method that combines the results of the other algorithms to produce a more robust
correction. Each algorithm operates on the fixation sequence of a single trial and must be
applied per trial. The :func:`~pymovements.events.correction.correct_fixations` function and
the :py:meth:`Events.correct_fixations` method take care of this per-trial application.

.. currentmodule:: pymovements.events.correction

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
