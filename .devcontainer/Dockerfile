# syntax=docker/dockerfile:1.7
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---- System deps + R + git-annex ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev libcairo2-dev \
    git-annex \
    curl ca-certificates gnupg \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# ---- Node.js 24 (multi-arch via NodeSource apt repo) ----
# Works for both amd64 and arm64 on Debian Bookworm.
RUN install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && chmod a+r /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ---- R package baseline (no Stan) ----
ARG CRAN_SNAPSHOT=https://packagemanager.posit.co/cran/2026-06-24
RUN echo "options(repos = c(CRAN = '${CRAN_SNAPSHOT}'))" >> /usr/lib/R/etc/Rprofile.site \
    && Rscript -e "install.packages(c('tidyverse','data.table','jsonlite','arrow','reticulate','testthat'))"

# ---- Python tooling ----
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir uv pre-commit

# ---- DataLad in isolated venv ----
RUN python -m venv /opt/datalad-env \
    && /opt/datalad-env/bin/pip install --no-cache-dir datalad datalad-osf
ENV PATH="/opt/datalad-env/bin:${PATH}"

# ---- Sanity check versions at build time ----
RUN node --version && npm --version && python --version && R --version

WORKDIR /workspaces/pymovements
