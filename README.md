# SIRF-Exercises

This material is intended for practical demonstrations using 
[SIRF](https://github.com/CCPPETMR/SIRF/wiki/Software-Documentation) on PET and MR Image Reconstruction.

This repository contains exercises to get you going
with SIRF. Please check the [INSTALL.md](INSTALL.md) first.

Authors:
- Kris Thielemans (this document and PET exercises)
- Christoph Kolbitsch (MR exercises)
- David Atkinson (MR exercises)
- Evgueni Ovtchinnikov (PET and MR exercises)
- Edoardo Pasca (overall check and clean-up)

This software is distributed under an open source license, see [LICENSE.txt](LICENSE.txt)
for details.


Introduction
============

We are using Python for the exercises. Python is an open-source interactive language, 
a bit like MATLAB. We provide Python scripts for the exercises, so you should be fine.
Nevertheless, it would be best to read a Python tutorial first, see the [Appendices](#appendices).

We use
[Jupyter notebooks](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).

The rest of this document contains some information to get started.
***Please read this before the course.***



Information
===========


Jupyter notebooks
-----------------
If you are trying this on your own and have never used Jupyter notebooks,
you could [read the official documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html).
A useful introduction to the notebook interface [can be found here](http://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html).

In a nut-shell, you need to start the server
```bash
   cd /wherever/it/is/SIRF-Exercises
   jupyter notebook
```

This will open a web-page that looks like a file browser
(the *Jupyter Notebook dashboard*).
Click on `notebooks`, and drill down until you find a file with the extension `.ipynb`
that looks of interest, and click on that. This should open a new tab in your
web browser with the notebook open, all ready to run.

You will normally work by executing each *cell*
bit by bit, and then editing it to do some more work. Useful 
shortcuts:

-   `LEFT-CTRL + <RETURN>` executes the current cell.
-   `SHIFT + <RETURN>` executes the current cell and advances the cursor to the next
     cell.
-   `TAB` tries to complete the word/command you have just typed.


File extensions
---------------

- `.dat`: Siemens raw MR data file
- '.h5': HDF5 file with MR data or images
- `.hv`: Interfile header for an image (volume)
- `.ahv`: (ignore) old-style Interfile header for an image
- `.v`: raw data of an image (in floats)
- `.hs`: Interfile header for projection data (sinograms)
- `.s`: raw data of projection data (in floats)
- `.py`: Python file
- `.ipynb`: jupyter notebook

A note on keyboard short-cuts inside a VirtualBox VM
----------------------------------------------------

On Windows and Linux, VirtualBox sets the "host-key" by default to `Right-CTRL` on Windows/Linux, so
unless you change this, you have to use `Left-CTRL` to "send" the `CTRL`-keystroke
to the Virtual Machine. So, below we will always type `Left-CTRL`.

Linux Terminal
--------------

If you have never used a Linux/Unix terminal before, have a look at 
[a tutorial](https://help.ubuntu.com/community/UsingTheTerminal).

You can use `UPARROW` to go to previous commands, and use copy-paste shortcuts 
`Left-CTRL-SHIFT-C` and `Left-CTRL-SHIFT-V`.


Python
------
Here is some suggested material on Python (ordered from easy to quite time-consuming).

-   The official Python tutorial. Just read Section 1, 3, a bit of 4 and a tiny bit of 6.
    <https://docs.python.org/2/tutorial/>

-   Examples for matplotlib, the python module that allows you to make plots almost like in MATLAB
    <https://github.com/patvarilly/dihub-python-for-data-scientists-2015/blob/master/notebooks/02_Matplotlib.ipynb>

-   You could read bits and pieces of Python the Hard Way
    <http://learnpythonthehardway.org/book/index.html>

-   Google has an online class on Python for those who know some programming.
    This goes quite in depth and covers 2 days.
    <https://developers.google.com/edu/python/?csw=1>

One thing which might surprise you that in Python *indentation is important*. You would write for instance
```python
for z in range(0,image.shape[0]):
   plt.figure()
   plt.imshow(image[z,:,:])
# now do something else
```
iPython
-------
You might be able to convince spyder to run iPython.
And here are some useful iPython "magic" commands that you can use in the iPython
console on the right (but not in the scripts). Most of these are identical
to what you would use in the terminal. (*Note*: these commands do not work in a Python console.)

- change how figures appear
    - separate figures
    ```
    %matplotlib
    ```

    - inline with other output
    ```     
    %matplotlib inline
    ```

    - inline in the notebook but with extra options for the figures (required for animations)
    ```     
     %matplotlib notebook
    ```

-   change to a new directory
```python
    cd some_dir/another_subdir
```
-   change back 2 levels up
```python
    cd ../..
```
-   print current working directory
```python
    pwd
```
-   list files in current directory
```python
    ls *.hs
```
-   Running system commands from the iPython prompt can be done via an exclamation mark
```python
    !FBP2D FBP.par
```
-   Get rid of everything in memory
```python
    %reset
```

