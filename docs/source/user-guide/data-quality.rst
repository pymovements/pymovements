=======================
Inspecting Data Quality
=======================

.. attention::

   Work in progress ...

After loading and initial inspection of raw gaze data, the next step is to
assess data quality. High-quality data is essential for reliable analysis
and interpretation of eye-tracking results. Common data quality issues
include missing data, noise, drift, and artifacts caused by blinks or
head movements.

Inspecting Data Loss
--------------------

Data loss can occur during tracking due to blinks, tracking errors, or
participants looking away from the screen. The length and frequency of
consecutive data-loss segments are useful indicators of overall dataset
usability. The function `pymovements.plotting.data_loss_histogram` plots
the distribution of consecutive missing-data chunk lengths (expressed in
samples or time), which makes it easy to see whether loss is mostly
brief (e.g. short blinks) or prolonged (e.g. long tracking failures).

Example
~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import pymovements as pm
   from pymovements.gaze.experiment import Experiment

   # Create an Experiment with the appropriate display geometry
   experiment = Experiment(
      screen_width_px=1280,
      screen_height_px=1024,
      screen_width_cm=38,
      screen_height_cm=30.2,
      distance_cm=68,
      origin='upper left',
      sampling_rate=250.0,
   )

   # Load a dataset that contains simulated blinks / tracking loss
   gaze_with_loss = pm.gaze.from_csv(
      '../examples/gaze-with-loss.csv',
      experiment=experiment,
      time_column='time',
      pixel_columns=['x', 'y']
   )

   # Plot the data-loss histogram in units of time
   pm.plotting.data_loss_histogram(
      gaze_with_loss,
      column='pixel',
      unit='time',
      sampling_rate=experiment.sampling_rate,
   )
   plt.show()

Interpretation
~~~~~~~~~~~~~~

Short, infrequent gaps often reflect normal blinks or brief signal dropouts
and are commonly tolerated or interpolated in preprocessing. A large number
of long gaps suggests severe tracking problems and may warrant re-collection
or exclusion of affected trials or participants.
