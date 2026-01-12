===========================
Inspecting Raw Gaze Samples
===========================


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
