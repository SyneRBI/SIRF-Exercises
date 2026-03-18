# syntax=docker/dockerfile:1
FROM ghcr.io/synerbi/sirf:latest
USER root
RUN apt update && export DEBIAN_FRONTEND=noninteractive \
    && apt -y install --no-install-recommends \
    libgl1-mesa-glx xvfb
USER $NB_UID

RUN conda install -y conda-forge::pyvista \
    && pip install --no-cache-dir opengate

