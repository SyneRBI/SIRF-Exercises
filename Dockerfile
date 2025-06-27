FROM ghcr.io/synerbi/sirf:petric AS petric
RUN cat <<EOF | tee -a /opt/conda/etc/conda/activate.d/env_sirf.sh /opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh
export SIRF_EXERCISES_WORKING_PATH=~/SIRF-Exercises/workdir
export SIRF_EXERCISES_DATA_PATH=/mnt/materials/SIRF/Fully3D/SIRF
EOF
