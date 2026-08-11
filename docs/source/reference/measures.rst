Measures
========

.. currentmodule:: pymovements.measure.samples

.. rubric:: Sample Measures
    :name: sample-measures

.. autosummary::
    :toctree: api
    :template: function.rst

    amplitude
    bcea
    data_loss
    dispersion
    disposition
    location
    null_ratio
    peak_velocity
    rms_s2s
    std_rms

.. rubric:: Classes

.. autosummary::
    :toctree: api
    :template: class.rst

    SampleMeasureLibrary

.. currentmodule:: pymovements.measure.events

.. rubric:: Event Measures
    :name: event-measures

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: function.rst

    duration

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: class.rst

    EventProcessor
    EventSamplesProcessor

.. currentmodule:: pymovements.measure.reading

.. rubric:: Reading Measures

.. autosummary::
    :toctree: api
    :template: class.rst

    ReadingMeasures

.. autosummary::
    :toctree: api
    :template: function.rst

    compute_reading_measures
    first_duration
    first_fixation_duration
    first_pass_fixation_count
    first_pass_reading_time
    first_reading_time
    landing_position
    rereading_time
    regression_count_in
    regression_count_out
    regression_path_duration_exclusive
    regression_path_duration_inclusive
    right_bounded_reading_time
    saccade_length_in
    saccade_length_out
    total_fixation_count

.. rubric:: Annotations

.. autosummary::
    :toctree: api
    :template: function.rst

    annotate_fixations
    run_id
    prev_word_idx
    next_word_idx
    delta_in
    delta_out
    is_reg_in
    is_reg_out
    is_first_fixation
    is_first_pass
    regression_path_word
