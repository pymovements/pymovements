============================
 Contributing Documentation
============================

To generate documentation pages, you can install the necessary dependencies using:

.. code:: bash

    pip install -e '.[docs]'

`Sphinx <https://www.sphinx-doc.org>`_ generates the API documentation from the
numpydoc-style docstring of the respective modules/classes/functions.
You can build the documentation locally using the respective tox environment:

.. code:: bash

    tox -e docs

It will appear in the ``build/docs`` directory`.
Please note that in order to reproduce the documentation locally, you may need to install
``pandoc``.
If necessary, please refer to the `installation guide <https://pandoc.org/installing.html>`_ for
detailed instructions.

To rebuild the full documentation use

.. code:: bash

    tox -e docs -- -aE
