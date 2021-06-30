#! /bin/bash
# A script to download data for the PET part of the SIRF-Exercises course
#
# Usage:
#   /path/download_PET_data.sh optional_destination_directory
# if no argument is used, the destination directory will be set to <repository-root>/data
#
# Author: Ashley Gillman
# Copyright (C) 2021 CSIRO

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
"$SCRIPT_DIR/download_data.sh" -p -d "$1"
