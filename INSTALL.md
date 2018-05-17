Installation instructions
-------------------------
Author: Kris Thielemans

An easy way to run the exercises is to use the SIRF Virtual Machine where
all of this has been done for you, but you can install all of this yourself
of course. Below are some brief instructions.



Installing SIRF and the exercises
---------------------------------

You will need SIRF of course. Please check the [instructions on our wiki](https://github.com/CCPPETMR/SIRF/wiki/How-to-obtain-SIRF).

The SIRF-exercises themselves don't need further installation. Just get them, preferably via

    git clone https://github.com/CCPPETMR/SIRF-Exercises

You will also need to get extra data for some of the exercises. Please execute

    ~/devel/SIRF-Exercises/scripts/download_PET_data.sh

adjusting the path to where you installed the SIRF-Exercises of course.

Updating the exercises
----------------------
You can repeat of course clone the repository again to get any updates. However, if you made any changes
to the exercises, you might want to keep those. Merging your changes and any
"upstream" ones is unfortunately a bit complicated
due to the file format used by jupyter notebooks. The following should work

    pip install --user nbstripout
    cd SIRF-Exercises
    nbstripout --install
    git pull

(You have to write the `nbstripout` lines only once).

Jupyter notebooks
---------------
The exercises use [Jupyter notebooks](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).
These provide a nice GUI interface to Python. You need 2 components:
the jupyter notebook server and a web browser.

Check first if your system comes with the jupyter server. The following
should then work

    cd SIRF-Exercises
    jupyter notebook

If you need to install it, we currently recommend using `pip`. The following command
should work for you

       $SIRF_PYTHON_EXECUTABLE -m pip install jupyter

Note that we used an environment variable set when you install SIRF to make
sure that you use a Python version which is compatible with how you compiled SIRF.

Of course, if you've used (Ana)conda to install Python etc (and are sure
SIRF was compiled with that Python version), you can use conda to install
jupyter as well.
