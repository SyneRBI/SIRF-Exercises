# Note to participants of one of our courses

We provide online access to cloud resources. There is no need to follow any
installation process therefore. Instructions below are for setting it up yourself.

# Installation instructions

Author: Kris Thielemans

An easy way to run the exercises is to use the SIRF Virtual Machine or a Docker image where
SIRF has already been installed for you. To install SIRF and SIRF-Exercises with these methods, please read https://github.com/SyneRBI/SIRF/wiki/How-to-obtain-SIRF.

Alternatively you can install SIRF yourself. Below are some brief instructions.

The exercises use [Jupyter notebooks](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).
These provide a nice GUI interface to Python. You need 2 components:
The jupyter notebook server and a web browser to open it.


## Installing SIRF and the exercises yourself

You will need SIRF of course. Please check the [instructions on our wiki](https://github.com/SyneRBI/SIRF/wiki/How-to-obtain-SIRF).

The SIRF-exercises themselves can just be downloaded, preferably via

    mkdir ~/devel
    cd ~/devel
    git clone https://github.com/SyneRBI/SIRF-Exercises
    cd SIRF-Exercises


adjusting the path to where you want to install the SIRF-Exercises of course.

You will need to install the additional Python dependencies needed for the
exercises also

    $SIRF_PYTHON_EXECUTABLE -m pip install -r requirements.txt

where we used the environment variable created when you follow the `SIRF-SuperBuild` instructions to make
sure that you use a Python version which is compatible with how you compiled SIRF.

Of course, if you've used (Ana)conda to install Python etc (and are sure
SIRF was compiled with that Python version), you can use conda to install
dependencies as well (except the brainweb module, at the time of writing).
Or you could still choose to use conda's `pip` after

    conda install pip


After all this, you will need to do the steps indicated in the instructions above for the VM, after the `update_VM.sh` step.

### Updating the exercises after installation

You can repeat clone the repository again to get any updates. However, if you made any changes
to the exercises, you might want to keep those. Merging your changes and any
"upstream" ones is unfortunately a bit complicated
due to the file format used by jupyter notebooks. The following should work

    cd SIRF-Exercises
    nbstripout --install
    git config --global filter.nbstripout.extrakeys '
      metadata.celltoolbar 
      metadata.language_info.codemirror_mode.version
      metadata.language_info.pygments_lexer metadata.language_info.version'
    git pull

(You do not have to write the `nbstripout` lines on the VM, and on other systems you have to write those only once).

### Contributing to the SIRF-Exercises

Follow the guidelines given in https://github.com/SyneRBI/SIRF/blob/master/CONTRIBUTING.md. Please do follow the `nbstripout` instructions above **before** commiting any changes.
