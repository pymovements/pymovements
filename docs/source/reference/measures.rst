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
    build_word_level_table
    all_tokens_from_aois
    mark_skipped_tokens
    repair_word_labels
    first_duration
    first_fixation_duration
    first_pass_fixation_count
    first_pass_reading_time
    first_reading_time
    landing_position
    rereading_time
    regression_count_in
    regression_count_out
    regression_path_duration
    saccade_length_in
    saccade_length_out
    total_fixation_count

.. rubric:: Annotations

.. autosummary::
    :toctree: api
    :template: function.rst

    annotate_fixations
    annotate_run_id
    annotate_prev_word_idx
    annotate_next_word_idx
    annotate_delta_in
    annotate_delta_out
    annotate_is_reg_in
    annotate_is_reg_out
    annotate_is_first_fixation
    annotate_is_first_pass
