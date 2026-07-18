# Pymovements Development Guide

This document provides overview information about the `pymovements` repository and detailed instructions on its development container configuration.

## Repository Overview

`pymovements` is an open-source Python package designed for processing eye-tracking and gaze data. It facilitates downloading public eye-tracking datasets, preprocessing coordinate/gaze data, detecting eye movement events (e.g., fixations, saccades, microsaccades), and visualizing results.

### Codebase Structure

The codebase is organized as follows:
- **[src/pymovements/](file:///workspaces/pymovements/src/pymovements/)**: Main source code of the Python package.
  - **[dataset/](file:///workspaces/pymovements/src/pymovements/dataset/)**: Core `Dataset` class and interfaces.
  - **[datasets/](file:///workspaces/pymovements/src/pymovements/datasets/)**: Pre-configured definitions for public datasets (e.g., JuDo1000).
  - **[events/](file:///workspaces/pymovements/src/pymovements/events/)**: Event detection algorithms (such as I-VT, microsaccades).
  - **[gaze/](file:///workspaces/pymovements/src/pymovements/gaze/)**: Main gaze and position-velocity data representations.
  - **[measure/](file:///workspaces/pymovements/src/pymovements/measure/)**: Metrics and data quality checks.
  - **[plotting/](file:///workspaces/pymovements/src/pymovements/plotting/)**: Plotting and visualization helper functions.
  - **[transforms/](file:///workspaces/pymovements/src/pymovements/transforms/)**: Mathematical and coordinate transformation functions (e.g., pixel coordinates to degrees of visual angle).
  - **[synthetic/](file:///workspaces/pymovements/src/pymovements/synthetic/)**: Tools for generating synthetic eye-tracking data.
  - **[stimulus/](file:///workspaces/pymovements/src/pymovements/stimulus/)**: Stimulus configuration and utilities.
- **[tests/](file:///workspaces/pymovements/tests/)**: Automated unit and integration tests written in `pytest`.
- **[docs/](file:///workspaces/pymovements/docs/)**: Package documentation and tutorials built with Sphinx.

### Tech Stack & Dependencies

- **Python (>= 3.10)**: Core programming language.
- **Polars / Pandas / NumPy**: For high-performance tabular and array operations.
- **Matplotlib**: For rendering eye-tracking visualization plots.
- **Scipy / Scikit-learn**: For scientific calculations and data analysis.
- **R Programming Language**: Used in conjunction with `reticulate` for integrating specialized R libraries.
- **DataLad**: Used for version-controlling datasets and downloading eye-tracking data.

---

## Development Container Configuration

This repository is configured to be developed inside a VS Code Development Container. Below is the technical breakdown of the configuration.

### Environment Overview

The development container is built using:
- **Base Image:** `mcr.microsoft.com/devcontainers/python:3-3.12-bookworm` (Debian Bookworm with Python 3.12)
- **Remote User:** `vscode`
- **Workspace Folder:** `/workspaces/pymovements`

### Configuration Files

The container is configured using the following files:
1. **[devcontainer.json](file:///workspaces/pymovements/.devcontainer/devcontainer.json):** Main configuration file defining the build, workspace folder, VS Code settings, extensions, and lifecycle scripts.
2. **[Dockerfile](file:///workspaces/pymovements/.devcontainer/Dockerfile):** Defines the system dependencies, Node.js installation, R installation and CRAN snapshots, Python tooling, and isolated DataLad installation.
3. **[postCreate.sh](file:///workspaces/pymovements/.devcontainer/postCreate.sh):** Startup script executed after container creation to set up the Python virtual environment (`.venv`), install dependencies, configure pre-commit hooks, and run basic sanity checks.

### System-Level Dependencies

The Dockerfile installs the following system packages:
- **R Programming Language:** `r-base` and `r-base-dev`
- **Node.js:** Node.js v24 (installed via the official NodeSource APT repository)
- **Data Versioning/Git Tools:** `git-annex` (required for DataLad)
- **C/C++ Build & Link Libraries:** `libcurl4-openssl-dev`, `libssl-dev`, `libxml2-dev`, libcairo2-dev
- **Fonts:** `fonts-dejavu`
- **Other Utilities:** `curl`, `ca-certificates`, `gnupg`

### R Configuration & Packages

The CRAN package manager is pinned to a specific snapshot to ensure reproducibility:
- **CRAN Snapshot:** `https://packagemanager.posit.co/cran/2026-06-24`
- **Pre-installed R Packages:**
  - `tidyverse`
  - `data.table`
  - `jsonlite`
  - `arrow`
  - `reticulate`
  - `testthat`

### Python Environment & Packages

Python dependencies are managed using `uv` inside a virtual environment to prevent permission issues and speed up installations.

- **System-level Tools:** `uv` and `pre-commit` are pre-installed in the container.
- **Virtual Environment:** A virtual environment is created at `[pymovements/.venv](file:///workspaces/pymovements/.venv)` during the post-creation phase.
- **Project Packages:** Installed in editable mode with development and documentation dependencies using:
  ```bash
  uv pip install -e ".[dev,docs]"
  ```
- **DataLad:** DataLad and the OSF connector are installed inside an isolated virtual environment at `/opt/datalad-env`, which is prepended to the system `PATH`.

### VS Code Customizations

#### Extensions

The following extensions are automatically installed in the VS Code environment:
- **Python Support:** `ms-python.python`, `ms-python.vscode-pylance`
- **Formatting & Linting:** `charliermarsh.ruff`, `davidanson.vscode-markdownlint`, `esbenp.prettier-vscode`
- **Interactive Notebooks:** `ms-toolsai.jupyter`
- **R Support:** `REditorSupport.r`, `RDebugger.r-debugger`

#### Settings

- **Python Interpreter:** Configured to point to the workspace virtual environment: `/workspaces/pymovements/.venv/bin/python`.
- **Testing:** `pytest` is enabled, configured to discover tests in `src/` and `tests/`.
- **Terminal:** Automatic environment activation is enabled.

### Container Lifecycle Hooks

- **`postCreateCommand`**: Executes `[postCreate.sh](file:///workspaces/pymovements/.devcontainer/postCreate.sh)` to create the virtual environment, install package dependencies, configure pre-commit hooks, and run tests.
- **`postStartCommand`**: Configures Git to trust the workspace directory (`git config --global --add safe.directory /workspaces/pymovements`) to prevent permission and ownership warnings when running git commands inside the container.
