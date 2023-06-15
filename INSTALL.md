# Note to participants of one of our courses

We provide online access to cloud resources. There is no need to follow any
installation process therefore. Instructions below are for setting it up yourself.

# Installation instructions

Author: Kris Thielemans

The exercises use [Jupyter notebooks](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).
These provide a nice GUI interface to Python. You need a few components:
The jupyter notebook server and a web browser to open it, SIRF and the SIRF-Exercises.

An easy way to run the exercises is to use a GitHub Codespace, the SIRF Virtual Machine or a Docker image where
SIRF has already been installed for you. Please read https://github.com/SyneRBI/SIRF/wiki/How-to-obtain-SIRF.

Alternatively you can install SIRF and the SIRF-Exercises yourself. Below are some brief instructions.


## Installing SIRF and the exercises yourself

You will need SIRF of course. Please check the [instructions on our wiki](https://github.com/SyneRBI/SIRF/wiki/How-to-obtain-SIRF).

The SIRF-exercises themselves can just be downloaded, preferably via

    mkdir ~/devel
    cd ~/devel
    git clone https://github.com/SyneRBI/SIRF-Exercises
    cd SIRF-Exercises


adjusting the path to where you want to install the SIRF-Exercises of course.

You will need to install the additional Python dependencies needed for the
exercises. This includes a Jupyter notebook server.
If you've installed SIRF using a `conda` python environment, we recommend

    conda env update -n <your_env_name> --file environment.yml

If you didn't use `conda`, use

    $SIRF_PYTHON_EXECUTABLE -m pip install -r requirements.txt

where we used the environment variable created when you follow the `SIRF-SuperBuild` instructions to make
sure that you use a Python version which is compatible with how you compiled SIRF. Please note that the
`conda` environment includes dependencies (in particular [`cil`](https://github.com/TomographicImaging/CIL))
which are not available with `pip`.

**Warning:** At present, SIRF is not yet available via conda or pip. The above instructions therefore do not install SIRF itself.

Next steps are in the [DocForParticipants.md](DocForParticipants.md).

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
