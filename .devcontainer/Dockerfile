FROM synerbi/sirf:devel-service

# Copy environment.yml (if found) to a temp location so we update the environment. Also
COPY environment.yml /tmp/conda-tmp/
=RUN umask 0002 && mamba env update -n base -f /tmp/conda-tmp/environment.yml && rm -rf /tmp/conda-tmp
