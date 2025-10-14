==========
 Datasets
==========

**Please cite the respective paper for every dataset that you use in your research.**

Download your dataset and load it into memory with these few lines of code:

.. code-block:: python

    import pymovements as pm

    # Initialize the dataset object with its name
    # Specify your local directory for saving and loading data
    dataset = pm.Dataset(name='EMTeC', path='path/to/your/data/directory')

    # Download the dataset and extract all archives.
    dataset.download()

    # Load the dataset into memory for processing
    dataset.load()

-----------------
 Public Datasets
-----------------

.. csv-table::
   :file: public_datasets.csv
   :widths: 30, 35, 5, 5, 5, 5, 5, 5, 5
   :header-rows: 1

------------------
 Example Datasets
------------------

.. csv-table::
   :file: example_datasets.csv
   :widths: 65, 5, 5, 5, 5, 5, 5, 5
   :header-rows: 1

.. datatemplate:yaml:: datasets.yaml
   :template: datasets.rst
