# How to use pymovements in R

.. warning::

   This page gives you a working route from R to pymovements objects through ``reticulate``.
   It is documented, not supported: the pymovements team does not develop or maintain R bindings.
   The code on this page is not covered by tests, so a pymovements or reticulate release may break it without notice.

This guide shows how to use `pymovements` from R via the `reticulate` package.

---

## Install and load reticulate in R

```r
 install.packages("reticulate")
```

Load the package.

```r
library(reticulate)
```

## Installing pymovements.

```r
py_install("pymovements")
```

If this fails, create a dedicated python environment and make sure to point R to it.

### Set up a dedicated environment

Skip this step if you already have an environment containing `pymovements`.

#### 1. using reticulate functionality in R

```r
reticulate::install_miniconda()
pymovements_packages <- c(
  "python==3.10",
  "pymovements"
)
reticulate::conda_create("pymovements_env", packages = pymovements_packages, pip = TRUE)
```

#### 2. using terminal

If you work with Conda:

```bash
conda create -n pymovements_env python=3.10 # supported: 3.10–3.13
conda activate pymovements_env
conda install -c conda-forge pymovements
```

If you prefer virtualenv:

```bash
python -m venv pymovements_env
# Activate the environment:
# macOS/Linux:
source pymovements_env/bin/activate
# Windows:
pymovements_env\Scripts\activate
pip install pymovements
```

### Point R to use your Python environment

If you used Conda:

```r
use_condaenv("pymovements_env", required = TRUE)
```

If you used virtualenv:

```r
use_virtualenv("pymovements_env", required = TRUE)
```

## Working with pymovements

Import pymovements as `pm`.

```r
pm <- import("pymovements")
```

Now pymovements should appear as `pm` under values in your environment

Access functions and data within python modules and classes via the `$` operator

To test, you can proceed with the "pymovements in 10 minutes" tutorial,
for example, this is how you download the ToyDataset:

```r
dataset = pm$Dataset('ToyDataset', path='data/ToyDataset')
dataset$download()
```

Now let's load in the dataset into R and display the found files:

```r
dataset$load()
dataset$fileinfo
```

## Getting gaze samples into an R data.frame

`dataset$gaze` is a list of `Gaze` objects, one per file, and each holds its samples as a polars DataFrame.
reticulate converts pandas data frames to R data.frames but has no converter for polars, so the samples take one step through pandas.
The `unnest()` call first splits the nested `pixel` column into `pixel_x` and `pixel_y`, so every column arrives in R as a plain vector:

```r
gaze <- dataset$gaze[[1]]
gaze$unnest()
samples <- gaze$samples$to_pandas()
```

`samples` is now an ordinary R data.frame of the first file's gaze samples:

```r
head(samples)
```

This shows the first rows with the columns `time`, `stimuli_x`, `stimuli_y`, `pixel_x` and `pixel_y`.

## Related handy functions:

Load a python shell in R.

```r
repl_python()
```

Information about the version of Python currently being used by reticulate as well as which Python environment R sees:

```r
py_config()
```

Show all envs R can see:

```r
conda_list()
```
