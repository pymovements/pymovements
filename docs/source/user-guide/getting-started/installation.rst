=====================
 Installation Options
=====================

.. grid::
    :gutter: 5

    .. grid-item-card::

        .. card-header::

            **Install pymovements**

        Choose your preferred way:

        .. tab-set::

            .. tab-item:: pip

                *pymovements* can be installed via pip from `PyPI <https://pypi.org/project/pymovements>`__:

                .. code-block:: bash

                    pip install pymovements

            .. tab-item:: uv

                *pymovements* can also be installed via `uv <https://github.com/astral-sh/uv>`__ from PyPI:

                .. code-block:: bash

                    uv pip install pymovements

            .. tab-item:: conda

                *pymovements* is part of the `conda-forge <https://anaconda.org/conda-forge/pymovements>`__ repository and can be installed with Anaconda or Miniconda:

                .. code-block:: bash

                    conda install -c conda-forge pymovements

                If you haven't configured *conda-forge* yet, add it to your channels and enable strict priority:

                .. code-block:: bash

                    conda config --add channels conda-forge
                    conda config --set channel_priority strict

                Then install *pymovements* into your conda environment:

                .. code-block:: bash

                    conda install -c conda-forge pymovements

            .. tab-item:: source (dev)

                To use the latest development version or try out tutorials, clone the repository and install in editable mode:

                .. code-block:: bash

                    git clone https://github.com/pymovements/pymovements.git
                    pip install --upgrade pip
                    pip install -e '.[dev]'


Optional Dependencies
---------------------

A plain installation covers most of pymovements. A few file formats and the pandas
interoperability need an additional package. If you do not want to think about it, install
everything:

.. code-block:: bash

    pip install 'pymovements[all]'

If you want to keep your environment small, install only the extra you need:

.. list-table::
    :header-rows: 1

    * - Extra
      - Packages
      - Needed for
    * - ``rdata``
      - ``pyreadr``
      - R data files (``.rda``), as used by :ref:`MECOL1W1`, :ref:`MECOL1W2`, :ref:`MECOL2W1`
        and :ref:`MECOL2W2`.
    * - ``excel``
      - ``fastexcel``
      - Excel files (``.xlsx``), as used by :ref:`ChineseReading`.
    * - ``pandas``
      - ``pandas``
      - pandas DataFrames, as read by :py:func:`~pymovements.gaze.from_pandas`.
    * - ``all``
      - all of the above
      - Every optional feature.

If a feature needs a package that is not installed, pymovements raises an ``ImportError`` that
names the extra to install.

Conda has no extras. The conda-forge package ``pymovements-all`` is the counterpart of the
``all`` extra:

.. code-block:: bash

    conda install -c conda-forge pymovements-all

For a single extra, install its packages from the table next to pymovements, for example
``conda install -c conda-forge pymovements pyreadr`` for R data files.
