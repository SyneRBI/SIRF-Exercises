import os
import sys

# Add the SIRF-Exercises Python library to the PYTHONPATH
this_directory = os.path.dirname(__file__)
repo_directory = os.path.dirname(os.path.dirname(this_directory))
sys.path.append(os.path.join(repo_directory, 'lib'))