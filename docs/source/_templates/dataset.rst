.. -*- mode: rst -*-

.. _{{ data['name'] }}:

{{ data['name'] }}
-------------------------------------------------

{{ data['long_name'] }}

{{ data['description'] }}


How to Download
^^^^^^^^^^^^^^^

.. code-block:: python

    import pymovements as pm

    # Initialize the dataset object with its name
    # Specify your local directory for saving and loading data
    dataset = pm.Dataset(name='{{ data['name'] }}', path='path/to/your/data/directory')

    # Download the dataset and extract all archives.
    dataset.download()

    # Load the dataset into memory for processing
    dataset.load()
