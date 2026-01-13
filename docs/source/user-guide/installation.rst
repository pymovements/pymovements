=====================
 Installation Options
=====================

.. grid::
    :gutter: 5

    .. grid-item-card::

        **Install pymovements**
        ^^^^^^^^^^^^^^^^^^^^^^^

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
                    pip install -e .[dev]


.. grid::
    :gutter: 5

    .. grid-item-card::

        **Already installed?**
        ^^^^^^^^^^^^^^^^^^^^^^

        Get familiar with *pymovements* by working through this beginner tutorial:

        .. button-ref:: pymovements_in_10_minutes
            :color: primary
            :shadow:

            pymovements in 10 minutes
