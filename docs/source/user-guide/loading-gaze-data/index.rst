==================
Loading Gaze Data
==================

Eye trackers export data in a variety of proprietary and semi-standard
formats, such as binary `EDF` files, `ASCII` (`ASC`) exports, `CSV` or `TSV`
tables, or vendor-specific text formats. These files differ in structure,
time units, coordinate conventions, and in how samples, events, and metadata
are represented. Converting them into a consistent internal representation is
therefore a necessary first step before analysis.

Loading data into `pymovements` performs this conversion. The loading
functions transform heterogeneous eye-tracker exports into a unified data
structure by creating a :py:class:`~pymovements.Gaze` object (read more about it
in :doc:`Understanding Eye-Tracking Data <../getting-started/understanding-data>`).
This object stores time-ordered gaze samples together with the experimental metadata
required for further interpretation and analysis.

.. toctree::
   :maxdepth: 3

   csv
   asc
   begaze
   ipc
