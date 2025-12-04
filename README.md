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

With pymovements loading your eye movement [datasets](https://pymovements.readthedocs.io/en/stable/datasets/index.html) is just a few lines of code away

```python
import pymovements as pm

dataset = pm.Dataset(
    'JuDo1000',  # choose a public dataset from our dataset library
    path='data/judo100',  # setup your local dataset path
)
dataset.download()  # download a public dataset from our dataset library
dataset.load()  # load the dataset
```

calculate velocities on the fly

```python
dataset.pix2deg()  # transform pixel coordinates to degrees of visual angle
dataset.pos2vel()  # transform positional data to velocity data
```

and extract events with different eye movements event extraction algorithms

```python
dataset.detect('ivt')  # detect fixation using the I-VT algorithm
dataset.detect('microsaccades')  # detect saccades using the microsaccades algorithm
```

Check out our guide on how to install *pymovements* and get started here:
[Installation](https://pymovements.readthedocs.io/en/stable/getting-started.html)

We provide a range of tutorial aimed at beginners:
[Tutorials](https://pymovements.readthedocs.io/en/stable/tutorials/index.html)

The complete reference of the package can be found here:
[API Reference](https://pymovements.readthedocs.io/en/stable/reference/index.html)

## Contributing

We welcome any sort of contribution to pymovements!

For a detailed guide, please refer to our [CONTRIBUTING.md](https://github.com/pymovements/pymovements/blob/main/CONTRIBUTING.md) first.

If you have any questions, please [open an issue](
https://github.com/pymovements/pymovements/issues/new/choose) or write us at
[pymovements@python.org](mailto:pymovements@python.org)

## Citing

If you are using pymovements in your research, we would be happy if you cite our work by using the following BibTex entry:

```bibtex
@inproceedings{pymovements,
    author = {Krakowczyk, Daniel G. and Reich, David R. and Chwastek, Jakob and Jakobi, Deborah N.
 and Prasse, Paul and Süss, Assunta and Turuta, Oleksii and Kasprowski, Paweł
 and Jäger, Lena A.},
    title = {pymovements: A Python Package for Processing Eye Movement Data},
    year = {2023},
    isbn = {979-8-4007-0150-4/23/05},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3588015.3590134},
    doi = {10.1145/3588015.3590134},
    booktitle = {2023 Symposium on Eye Tracking Research and Applications},
    location = {Tubingen, Germany},
    series = {ETRA '23},
}
```

There is also a preprint available on [arxiv](https://arxiv.org/abs/2304.09859).
