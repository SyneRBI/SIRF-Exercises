'''Library of common utilities shared between notebooks in SIRF-Exercises.'''

# Author: Ashley Gillman
# Copyright 2021 Commonwealth Scientific and Industrial Research Organisation
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os


def exercises_data_path(*data_type):
    '''
    Returns the path to data used by SIRF-exercises.

    data_type: either 'PET', 'MR' or 'Synergistic', or use multiple arguments for
    subdirectories like exercises_data_path('PET', 'mMR', 'NEMA_IQ').
    '''
    try:
        from .data_path import data_path
    except ImportError:
        data_path = os.getenv('SIRF_EXERCISES_DATA_PATH')

    if data_path is None or not os.path.exists(data_path):
        raise RuntimeError(
            "Exercises data weren't found. Please run download_data.sh in the "
            "scripts directory (use its -h option to get help)")

    return os.path.join(data_path, *data_type)



def exercises_working_path(*subfolders):
    '''
    Creates and returns the working directory for the
    current exercise, based on the argument(s). If multiple
    strings are given, they will be treated as subdirectories.

    Implementation detail: this is defined as
    {exercises_data_path()}/working_folder/{subfolders[0]}/{subfolders[1]}/...

    subfolders: the path will include this.
    Multiple arguments can be given for nested subdirectories.
    '''
    try:
        from .working_path import working_dir
    except ImportError:
        working_dir = os.getenv('SIRF_EXERCISES_WORKING_PATH', exercises_data_path('working_folder'))
    wkdir = os.path.join(working_dir, *subfolders)
    os.makedirs(wkdir, exist_ok=True)
    return wkdir


def cd_to_working_dir(*subfolders):
    '''Same as os.chdir(exercises_working_path(*subfolders))'''
    os.chdir(exercises_working_path(*subfolders))
