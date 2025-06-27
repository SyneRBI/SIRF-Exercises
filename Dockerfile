FROM ghcr.io/synerbi/sirf:petric AS petric

# hardcoded from env_sirf.sh
ENV SIRF_PATH=/opt/SIRF-SuperBuild/sources/SIRF
ENV SIRF_INSTALL_PATH=/opt/SIRF-SuperBuild/INSTALL
ENV LD_LIBRARY_PATH=/opt/SIRF-SuperBuild/INSTALL/lib:/opt/SIRF-SuperBuild/INSTALL/lib64:$LD_LIBRARY_PATH
ENV DYLD_FALLBACK_LIBRARY_PATH=/opt/SIRF-SuperBuild/INSTALL/lib
ENV PYTHONPATH="/opt/SIRF-SuperBuild/INSTALL/python${PYTHONPATH:+:${PYTHONPATH}}"
ENV SIRF_PYTHON_EXECUTABLE=/opt/conda/bin/python3
ENV PATH=/opt/SIRF-SuperBuild/INSTALL/bin:$PATH
ENV STIR_PATH=/opt/SIRF-SuperBuild/sources/STIR
ENV SIRF_EXERCISES_WORKING_PATH=~/SIRF-Exercises/workdir
ENV SIRF_EXERCISES_DATA_PATH=/mnt/materials/SIRF/Fully3D/SIRF

RUN cat <<EOF | tee -a /opt/conda/etc/conda/activate.d/env_sirf.sh /opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh
export SIRF_EXERCISES_WORKING_PATH=$SIRF_EXERCISES_WORKING_PATH
export SIRF_EXERCISES_DATA_PATH=$SIRF_EXERCISES_DATA_PATH
EOF
