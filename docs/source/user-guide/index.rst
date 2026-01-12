==========
User Guide
==========

This user guide introduces the concepts, data structures, and workflows
used in pymovements. It is intended for students, researchers, and
practitioners who work with eye-tracking data and want to understand how
to move from raw recordings to analysis-ready and reusable datasets.

.. signals → structure → interpretation → reuse

The guide focuses on *conceptual understanding* and *transparent data
processing*, rather than step-by-step recipes.

Throughout this guide, we emphasize that there is no single “correct”
eye-tracking pipeline. Processing choices depend on research goals and data
quality, and each transformation introduces assumptions that should be made
explicit for reproducible analysis.

To follow the examples in this guide, you will need a working Python
environment and the pymovements package installed.

pymovements can be installed via pip: `pip install pymovements`

For more information, see the next section on :doc:`Installation Options <installation>`.

.. In the following sections, you will learn how to:

.. - understand the structure and meaning of eye-tracking data,
.. - inspect and evaluate data quality,
.. - preprocess raw samples,
.. - detect and evaluate oculomotoric events,
.. - work with standardized datasets and metadata,
.. - visualize eye movements and summarize behavior,
.. - prepare, validate, and publish reusable eye-tracking datasets.

.. toctree::
   :maxdepth: 2

   installation
   understanding-data
   inspect-raw-samples
   data-quality
   event-detection
   work-with-datasets
