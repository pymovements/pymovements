===========================
Inspecting Raw Gaze Samples
===========================

On the Notion of Raw Data
-------------------------

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

Time-Series Plots
-----------------

Time-series plots are often the first step when working with newly loaded gaze data.
The `tsplot()` function visualizes raw gaze samples from a Gaze object as signals over time,
allowing inspection of gaze position, velocity, or pupil size before any preprocessing or event
detection is applied.

See an example in the Plotting Gaze Data tutorial.

From Pixels to Degrees of Visual Angle
--------------------------------------

Pixel coordinates depend on screen resolution, viewing distance, and physical screen size. To
compare gaze behaviour across setups or participants, it is often useful to convert pixels to
degrees of visual angle (dva). This conversion requires knowledge of the experimental geometry and
is handled explicitly in pymovements by the `pix2deg()` function.

From Position to Velocity
-------------------------

Many eye-movement measures are derived not from position directly but from its temporal
derivatives. Velocity is computed from changes in gaze position over time and is central to event
detection algorithms for saccades and fixations. In pymovements, velocity is computed explicitly
from position data with the `pos2vel()` function, using the sampling rate stored in the eye tracker
definition.
