===============================
Understanding Eye-Tracking Data
===============================

Before preprocessing, event detection, or statistical analysis, it is important to understand
what eye-tracking data look like at their most basic level and how they are structured. This
section introduces the core components of eye-tracking recordings and the representations
commonly used in analysis.

Eye trackers and sampling frequency (rate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Eye trackers differ substantially in their technical characteristics, including spatial accuracy,
precision, robustness to head movement, and sampling rate. The sampling rate, typically reported in
hertz (Hz), specifies how many gaze samples are recorded per second. Values range from around 30 Hz
for low-cost or webcam-based systems to 500 Hz or 2000 Hz for high-end research-grade eye trackers.

The sampling rate determines the temporal resolution of the gaze signal and constrains which
eye-movement phenomena can be reliably analysed. In general, higher sampling rates allow finer
temporal detail to be captured. From a signal-processing perspective, the Nyquist–Shannon sampling
theorem states that a signal must be sampled at least twice as fast as its highest frequency
component to avoid aliasing. In practice, this means that high sampling rates are required to
capture rapid eye movements such as saccades, while lower sampling rates may be sufficient for
analyses focused on fixations or overall viewing patterns.

In pymovements, these device-level properties are represented explicitly via an EyeTracker object
that is part of the experiment definition (class: `Experiment`). This separates how the data were
recorded from what the recorded samples contain. The sampling rate is used implicitly when
computing derived measures such as velocity, acceleration, or event durations.

From Eye-Tracker Export Files to Gaze Samples in pymovements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Eye trackers export data in a variety of proprietary and semi-standard formats, such as binary EDF
files, ASCII or ASC exports, CSV or TSV tables, or vendor-specific text formats. These formats
differ in structure, time units, coordinate conventions, and how samples, events, and metadata are
represented, making parsing into a consistent internal representation a necessary first step for
analysis.

pymovements facilitates the transformation of these heterogeneous eye-tracker exports into a
consistent internal format by creating a Gaze object that contains gaze samples together with their
experimental context. See the Parsing SR Research EyeLink Data tutorial to walk through loading
`*.asc` files, extracting gaze samples and metadata, and inspecting the resulting Gaze object.

At this stage, the user specifies or confirms which columns represent time, gaze position, or other
quantities. Monocular and binocular recordings are detected and harmonised. Timestamps are
converted into a common unit. Device and recording metadata, such as sampling rate, tracked eye,
screen resolution, and calibration information, are attached to the data through the experiment and
eye tracker definitions.

On the Notion of Raw Data in Eye Tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The term "raw gaze" or "raw eye-tracking data" is used inconsistently in the literature and can
refer to different levels of data, depending on context. Common usages include:

**Raw eye-tracker files**
  May contain samples, events, messages, and metadata mixed together.

**Raw samples**
  Gaze coordinates over time without filtering or event classification.

**Vendor-labeled events**
  Fixations or saccades produced by proprietary software.

In pymovements, raw samples refer to the lowest-level gaze data available after import, before any
additional preprocessing steps, such as smoothing, velocity computation, or event detection. These
raw samples form the foundation for all subsequent analyses. Later transformations, e.g. converting
pixel coordinates to degrees of visual angle, computing velocities, or segmenting fixations and
saccades, operate on these samples and depend on the assumptions and metadata established during
loading.

Inspecting Raw Gaze Samples with Time-Series Plots (tsplot)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Time-series plots are often the first step when working with newly loaded gaze data. The `tsplot()`
function visualizes raw gaze samples from a Gaze object as signals over time, allowing inspection
of gaze position, velocity, or pupil size before any preprocessing or event detection is applied.
See an example in the Plotting Gaze Data tutorial.

Coordinate Systems: Screen vs. Eye-Centred Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gaze data can be expressed in different coordinate systems, depending on the experimental setup
and research question.

Screen coordinates (allocentric coordinates) describe where gaze falls on the stimulus or display
surface, typically in pixels or degrees of visual angle.

Eye-in-head coordinates (egocentric coordinates) describe eye orientation relative to the
participant’s head and are more common in head-mounted or mobile eye tracking.

pymovements primarily works with stimulus-referenced coordinates but allows explicit
transformations when the necessary experimental information is available.

From Pixels to Degrees of Visual Angle (pix2deg)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pixel coordinates depend on screen resolution, viewing distance, and physical screen size. To
compare gaze behaviour across setups or participants, it is often useful to convert pixels to
degrees of visual angle (dva). This conversion requires knowledge of the experimental geometry and
is handled explicitly in pymovements by the `pix2deg()` function.

From Position to Velocity (pos2vel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many eye-movement measures are derived not from position directly but from its temporal
derivatives. Velocity is computed from changes in gaze position over time and is central to event
detection algorithms for saccades and fixations. In pymovements, velocity is computed explicitly
from position data with the `pos2vel()` function, using the sampling rate stored in the eye tracker
definition.
