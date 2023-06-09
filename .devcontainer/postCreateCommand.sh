sed -i 's/\r$//' /workspaces/SIRF-Exercises/scripts/download_data.sh
bash /workspaces/SIRF-Exercises/scripts/download_data.sh -m -p
mamba env update --file environment.yml