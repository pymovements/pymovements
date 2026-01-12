===============================
Understanding Eye-Tracking Data
===============================

Before preprocessing, event detection, or statistical analysis, it is important to understand
what eye-tracking data look like at their most basic level and how they are structured. This
section introduces the core components of eye-tracking recordings and the representations
commonly used in analysis.

What is Eye-Tracking Data?
--------------------------

Eye-tracking data consists of measurements of eye position over time, typically recorded at a fixed
sampling frequency.

Depending on the experimental setup, these measurements can be collected while participants are:

- reading texts,
- viewing static images,
- watching videos,
- or interacting with dynamic or real-world **stimuli**, i.e., the content presented to
  participants during the experiment.

A device called an **eye tracker** estimates the point of **gaze**, that is, where on a stimulus or
display a participant is inferred to be looking, by measuring the relative position of the pupil
and corneal reflections. During calibration, participants fixate known reference points, allowing
the system to learn a mapping from eye position signals to **gaze coordinates** on the stimulus. In
screen-based experiments, gaze coordinates are commonly expressed in pixel units, corresponding to
positions on the display surface.

Depending on the experimental setup and research question, gaze data can be expressed in different
coordinate systems. For instance, **allocentric coordinates** describe where gaze falls on the
stimulus or display surface, typically in pixels or degrees of visual angle. **Egocentric
coordinates** describe eye orientation relative to the head, often in degrees of rotation.
These coordinates are more common in head-mounted or mobile eye tracking.

pymovements primarily works with stimulus-referenced coordinates but allows explicit
transformations when the necessary experimental information is available.

At the most basic level, eye-tracking data consist of time-ordered **gaze samples** which typically
include:

- a timestamp,
- horizontal and vertical gaze coordinates,
- optional pupil size estimates,
- and device- or vendor-specific fields.

Crucially, eye-tracking data are signals rather than direct measurements of perception or
cognition. Constructs such as attention, comprehension, or cognitive processes are inferred through
preprocessing, event detection, and analysis choices.

Eye Trackers and Sampling Frequency (Rate)
------------------------------------------

Eye trackers differ substantially in their technical characteristics, including spatial accuracy,
precision, robustness to head movement, and **sampling rate**. The sampling rate, typically
reported in hertz (Hz), specifies how many gaze samples are recorded per second. Values range
from around 30 Hz for low-cost or webcam-based systems to 500 Hz or 2000 Hz for high-end
research-grade eye trackers.

The sampling rate determines the temporal resolution of the gaze signal and constrains which
eye-movement phenomena can be reliably analysed. In general, higher sampling rates allow finer
temporal detail to be captured. From a signal-processing perspective, the Nyquist–Shannon sampling
theorem states that a signal must be sampled at least twice as fast as its highest frequency
component to avoid aliasing. In practice, this means that high sampling rates are required to
capture rapid eye movements such as saccades, while lower sampling rates may be sufficient for
analyses focused on fixations or overall viewing patterns.

In pymovements, these device-level properties are represented explicitly via an
:class:`~pymovements.gaze.eyetracker.EyeTracker` object that is part of the experiment definition
(class: :class:`~pymovements.gaze.experiment.Experiment`). This separates how the data were
recorded from what the recorded samples contain. The sampling rate is used implicitly when
computing derived measures such as velocity, acceleration, or event durations.

From Eye-Tracker Export Files to Gaze Samples in pymovements
------------------------------------------------------------

Eye trackers export data in a variety of proprietary and semi-standard formats, such as binary
``EDF`` files, ``ASCII`` or ``ASC`` exports, ``CSV`` or ``TSV`` tables, or vendor-specific text
formats. These formats differ in structure, time units, coordinate conventions, and how samples,
events, and metadata are represented, making parsing into a consistent internal representation a
necessary first step for analysis.

pymovements facilitates the transformation of these heterogeneous eye-tracker exports into a
consistent internal format by creating a :class:`~pymovements.gaze.gaze.Gaze` object that
contains gaze samples together with their experimental context. See the
:doc:`Parsing SR Research EyeLink Data tutorial <tutorials/parsing-dataset>` to walk through
loading ``*.asc`` files, extracting gaze samples and metadata, and inspecting the resulting Gaze
object.

At this stage, the user specifies or confirms which columns represent time, gaze position, or other
quantities. Monocular and binocular recordings are detected and harmonised. Timestamps are
converted into a common unit. Device and recording metadata, such as sampling rate, tracked eye,
screen resolution, and calibration information, are attached to the data through the experiment and
eye tracker definitions.

Eye-tracking Analysis as a Sequence of Transformations
------------------------------------------------------

Thus, the eye-tracking analysis involves a sequence of transformations that convert raw gaze
samples into higher-level representations. Each step builds on the previous one and introduces
assumptions and analytical choices that shape the final results.

Common stages in this process include:

- **Recorded data**: Vendor-specific files produced by the eye tracker and experiment software.
- **Raw samples**: Time-ordered gaze measurements extracted from the recordings.
- **Preprocessed samples**: Samples that have been cleaned, filtered, transformed into meaningful
  units, or restricted to relevant time windows (e.g. trials).
- **Events**: Segments of the signal classified as fixations, saccades, blinks, or other
  eye-movement events using detection algorithms.
- **Analysis measures**: Summary statistics, models, or visualizations derived from samples or
  events.

However, there is no single preprocessing pipeline or set of eye-tracking measures that is optimal
for all research questions. Instead, appropriate choices depend on the experimental design, the
properties of the recording device, and the quality of the data
(see :doc:`Inspecting Data Quality <data-quality>`). Making these transformations explicit and
transparent is therefore essential for valid, interpretable, and reproducible analysis.
