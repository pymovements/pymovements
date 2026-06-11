<p style="text-align:center;">
<img width="110%" height="110%" alt="pymovements"
 src="https://raw.githubusercontent.com/pymovements/pymovements/main/docs/source/_static/logo.svg"
 onerror="this.onerror=null;this.src='./docs/source/_static/logo.svg';"/>
</p>

---

[![PyPI Latest Release](https://img.shields.io/pypi/v/pymovements.svg)](https://pypi.python.org/pypi/pymovements/)
[![Conda Latest Release](https://img.shields.io/conda/vn/conda-forge/pymovements)](https://anaconda.org/conda-forge/pymovements)
[![PyPI status](https://img.shields.io/pypi/status/pymovements.svg)](https://pypi.python.org/pypi/pymovements/)
[![Python version](https://img.shields.io/pypi/pyversions/pymovements.svg)](https://pypi.python.org/pypi/pymovements/)
![Operating System](https://img.shields.io/badge/os-linux%20%7C%20macOS%20%7C%20windows-blue)
[![License](https://img.shields.io/pypi/l/pymovements.svg)](https://github.com/pymovements/pymovements/blob/master/LICENSE.txt)
[![Test Status](https://img.shields.io/github/actions/workflow/status/pymovements/pymovements/tests.yml?label=tests)](https://github.com/pymovements/pymovements/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/pymovements/badge/?version=latest)](https://pymovements.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/pymovements/pymovements/branch/main/graph/badge.svg?token=QY3NDHAT2C)](https://app.codecov.io/gh/pymovements/pymovements)
[![PyPI Downloads](https://static.pepy.tech/badge/pymovements)](https://pepy.tech/projects/pymovements)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pymovements/pymovements/HEAD?labpath=docs%2Fsource%2Ftutorials)

pymovements is an open-source python package for processing eye movement data. It provides a simple
interface to download publicly available datasets, preprocess gaze data, detect oculomotoric events
and render plots to visually analyze your results.

- **Website:** https://github.com/pymovements/pymovements
- **Documentation:** https://pymovements.readthedocs.io
- **Source code:** https://github.com/pymovements/pymovements
- **PyPI package:** https://pypi.org/project/pymovements
- **Conda package:** https://anaconda.org/conda-forge/pymovements
- **Bug reports:** https://github.com/pymovements/pymovements/issues
- **Contributing:** https://github.com/pymovements/pymovements/blob/main/CONTRIBUTING.md
- **Mailing list:** pymovements@python.org ([subscribe](https://mail.python.org/mailman3/lists/pymovements.python.org/))
- **Discord:** https://discord.gg/K2uS2R6PNj

## Getting Started

If you are new to pymovements or to eye-tracking data analysis, we recommend starting with the **User Guide**, which introduces the concepts, data
structures, and workflows used throughout the library: 👉 :doc:`user-guide/index`


### Quick example

For a minimal example of loading and processing eye-tracking data with pymovements:

```python
import pymovements as pm

dataset = pm.Dataset(
    'JuDo1000',  # choose a public dataset from our dataset library
    path='data/judo100',  # setup your local dataset path
)
dataset.download()  # download a public dataset from our dataset library
dataset.load()  # load the dataset
```

Transform coordinates and calculate velocities:

```python
dataset.pix2deg()  # transform pixel coordinates to degrees of visual angle
dataset.pos2vel()  # transform positional data to velocity data
```

Detect oculomotoric events:

```python
dataset.detect('ivt')  # detect fixation using the I-VT algorithm
dataset.detect('microsaccades')  # detect saccades using the microsaccades algorithm
```

<!-- With pymovements loading your eye movement [datasets](https://pymovements.readthedocs.io/en/stable/datasets/index.html) is just a few lines of code away -->

### Quick Links
- :doc:`Installation Options <user-guide/getting-started/installation>`
- [Tutorials](https://pymovements.readthedocs.io/en/stable/tutorials/index.html)
- [API Reference](https://pymovements.readthedocs.io/en/stable/reference/index.html)

## Contributing

We welcome any sort of contribution to pymovements!

For a detailed guide, please refer to our [CONTRIBUTING.md](https://github.com/pymovements/pymovements/blob/main/CONTRIBUTING.md) first.

If you have any questions, please [open an issue](
https://github.com/pymovements/pymovements/issues/new/choose) or write to us at
[pymovements@python.org](mailto:pymovements@python.org)

## Citing

If you use pymovements for a scientific publication, we would appreciate citations to the published
software and the following paper:

- [pymovements on Zenodo](https://doi.org/10.5281/zenodo.17397139). Please find us on Zenodo and
  replace with the citation for the version you are using. You can replace the full author list from
  there with "The pymovements project team" like in the example below.

  ```bibtex
    @software{pymovements,
      author    = {The pymovements project team},
      title     = {pymovements: A Python Package for Processing Eye Movement Data},
      month     = apr,
      year      = 2026,
      publisher = {Zenodo},
      version   = {v0.26.2},
      doi       = {10.5281/zenodo.19486286},
      url       = {https://doi.org/10.5281/zenodo.19486286},
    }
  ```
- [pymovements: A Python Package for Processing Eye Movement Data <pymovementsPaper](https://doi.org/10.1145/3588015.3590134)

  ```bibtex
    @inproceedings{pymovementsPaper,
      author    = {
        Krakowczyk, Daniel G. and Reich, David R. and Chwastek, Jakob and Jakobi, Deborah N.
        and Prasse, Paul and Süss, Assunta and Turuta, Oleksii and Kasprowski, Pawe\l{}
        and J\"{a}ger, Lena A.
      },
      title     = {pymovements: A Python Package for Processing Eye Movement Data},
      year      = {2023},
      isbn      = {9798400701504},
      publisher = {Association for Computing Machinery},
      address   = {New York, NY, USA},
      url       = {https://doi.org/10.1145/3588015.3590134},
      doi       = {10.1145/3588015.3590134},
      booktitle = {Proceedings of the 2023 Symposium on Eye Tracking Research and Applications},
      articleno = {53},
      numpages  = {2},
      location  = {Tubingen, Germany},
      series    = {ETRA '23}
    }
  ```

In case you want to cite specific parts of the package, like our dataset library or data quality
reports, please consider citing our other publications listed on our
[citing page](https://pymovements.readthedocs.io/en/stable/about-us/citing.html).


## Acknowledgements

The pymovements project was partially funded by the Swiss National Science Foundation (SNSF) and
the German Federal Ministry of Education and Research (BMBF). Please refer to our
[acknowledgements page](https://pymovements.readthedocs.io/en/stable/about-us/citing.html) for a
comprehensive overview.
