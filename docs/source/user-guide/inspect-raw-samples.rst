=============================
Working with Raw Gaze Samples
=============================

The term "raw gaze" or "raw eye-tracking data" is used inconsistently in the literature and can
refer to different levels of data, depending on context. Common usages include:

- **Raw eye-tracker files**: May contain samples, events, messages, and metadata mixed together.
- **Raw samples**: Gaze coordinates over time without filtering or event classification.
- **Vendor-labeled events**: Fixations or saccades produced by proprietary software.

In pymovements, raw samples refer to the lowest-level gaze data available after import, before any
additional preprocessing steps, such as smoothing, velocity computation, or event detection. These
raw samples form the foundation for all subsequent analyses. Later transformations, e.g. converting
pixel coordinates to degrees of visual angle, computing velocities, or segmenting fixations and
saccades, operate on these samples and depend on the assumptions and metadata established during
loading.

The table below shows a simplified example of raw gaze samples after import. Each row corresponds
to one time-ordered gaze sample.

+------------+--------+-----------------+
| time (ms)  | pupil  | pixel (x, y)    |
+============+========+=================+
| 2762704    | 783.0  | [512.1, 384.8]  |
+------------+--------+-----------------+
| 2762705    | 783.0  | [512.3, 385.2]  |
+------------+--------+-----------------+
| 2762708    | 783.0  | [512.5, 386.1]  |
+------------+--------+-----------------+
| 2762712    | 783.0  | [512.8, 387.4]  |
+------------+--------+-----------------+
| 2762716    | 783.0  | [513.1, 389.2]  |
+------------+--------+-----------------+

Column dtypes after import:

- ``time`` (``i64``) is the timestamp of the sample, typically in milliseconds.
- ``pupil`` (``f64``) is an estimate of pupil size (arbitrary units, device-dependent)
- ``pixel`` (``list[f64]``) contains the horizontal and vertical gaze coordinates on the display.

Inspecting Raw Samples with Time-Series Plots
---------------------------------------------

Time-series plots are often the first step when working with newly loaded gaze data. They provide a
direct view of the temporal structure of the signal and help assess data quality before any
preprocessing or event detection is applied.

The :func:`~pymovements.plotting.traceplot` function visualizes raw gaze samples from a
:class:`~pymovements.gaze.gaze.Gaze` object as signals over time, allowing inspection of
gaze position, pupil size, or derived quantities such as velocity.

Time-series inspection can reveal common issues such as signal loss, noise, blinks, sampling
irregularities, or calibration problems, which may strongly influence subsequent analysis steps.

See the :doc:`Plotting Gaze Data tutorial <../tutorials/plotting>` for an example of time-series
visualization using ``traceplot``.

Converting Units to Standardized Representations
------------------------------------------------

Raw gaze samples form the basis of all subsequent analysis, but meaningful interpretation often
requires transforming these samples into alternative representations. These transformations operate
directly on raw samples and typically precede any event detection or higher-level segmentation.

From Pixels to Degrees of Visual Angle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pixel coordinates depend on screen resolution, viewing distance, and physical screen size. To
compare gaze behaviour across setups or participants, it is often useful to convert pixels to
degrees of visual angle (dva). This conversion requires knowledge of the experimental geometry and
is handled explicitly in pymovements by the :func:`~pymovements.gaze.transforms.pix2deg` function.

From Position to Velocity
^^^^^^^^^^^^^^^^^^^^^^^^^

Many eye-movement measures are derived not from position directly but from its temporal
derivatives. Velocity is computed from changes in gaze position over time and is central to event
detection algorithms for saccades and fixations. In pymovements, velocity is computed explicitly
from position data with the :func:`~pymovements.gaze.transforms.pos2vel` function, using the
sampling rate stored in the eye tracker definition.
