#! /bin/bash
# A sript to download data for the PET part of the SIRF-Exercises course
#
# Usage:
#   /path/download_PET_data.sh optional_destination_directory
# if no argument is used, "${SIRF_EXERCISES_PATH}/data/PET" will be used, or finally
# the destination directory will be set to ~/.sirf-exercises/data/PET.
#
# Author: Kris Thielemans, Ashley Gillman
# Copyright (C) 2018-2021 University College London
# Copyright (C) 2021 CSIRO

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
"$SCRIPT_DIR/download_data.sh" -o "$1"