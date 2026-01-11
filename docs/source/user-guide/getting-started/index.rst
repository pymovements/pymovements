================
Getting Started
================

Whether you are working with eye-tracking data for the first time in a student assignment,
routinely collecting and analysing reading data in the lab, or reusing and/or publishing
datasets as part of a larger research effort, working with eye-tracking data often begins in the
same way. You are faced with files exported from an eye tracker, often accompanied by stimuli,
logs, and metadata, and the task of turning these heterogeneous files into data that are
meaningful, reliable, and reusable.

This guide introduces the concepts and pymovements workflows, data structures, and functions needed
to move from raw eye-tracker recordings to analysis-ready and reusable data.

In the following sections, you will learn how to:

- understand the structure and meaning of eye-tracking data,
- inspect and evaluate data quality,
- preprocess raw samples,
- detect and evaluate oculomotoric events,
- visualize eye movements and summarize behavior,
- work with standardized datasets and metadata,
- prepare, validate, and publish reusable eye-tracking datasets.

You can think of this guide as moving from signals → structure → interpretation → reuse.

To follow the examples in this guide, you will need a working Python environment and the
pymovements package installed.

In short, pymovements can be installed via pip: `pip install pymovements`. For more information,
see the section on :doc:`Installation Options <installation>`.
