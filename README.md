# SIRF-Exercises

This material is intended for practical demonstrations using 
[SIRF](https://github.com/CCPPETMR/SIRF/wiki/Software-Documentation) on PET and MR Image Reconstruction.

This repository contains exercises to get you going
with SIRF. Please check [INSTALL.md](INSTALL.md) first (unless
you have a Virtual Machine with the exercises
pre-installed).

Authors (this document):
- Kris Thielemans

This software is distributed under an open source license, see [LICENSE.txt](LICENSE.txt)
for details.


Introduction
============

We are using Python for the exercises. Python is an open-source interactive language, 
a bit like MATLAB. We provide Python scripts for the exercises, so you should be fine.
Nevertheless, it would be best to read a Python tutorial first, see the [Appendices](#appendices). We will use
[Spyder](https://pythonhosted.org/spyder) as our Python environment, but are looking
into Jupyter notebooks.

See the appendices at the end of this document for some information to get started.
***Please read this before the course.***



Appendices
==========


File extensions
---------------

- `.dat`: Siemens raw MR data file
- '.h5': HDF5 file with MR data or images
- `.hv`: Interfile header for an image (volume)
- `.ahv`: (ignore) old-style Interfile header for an image
- `.v`: raw data of an image (in floats)
- `.hs`: Interfile header for projection data (sinograms)
- `.s`: raw data of projection data (in floats)
- `.par`: STIR parameter file
- `.sh`: Shell script (sequence of commands)
- `.bat`: Windows batch file
- `.log`: log file (used to record output of a command)
- `.py`: Python file

A note on keyboard short-cuts inside a VirtualBox VM
----------------------------------------------------

On Windows and Linux, VirtualBox sets the "host-key" by default to `Right-CTRL`, so
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

Spyder
------
We use Spyder as a nice Integrated Development Environment (IDE) for Python
(or iPython which is a slightly friendlier version of Python). You need only
minimal knowledge of Python for this course, but it would be good to read-up a bit (see below).

You will normally work by loading an example script in Spyder in the editor, executing it
bit by bit, and then editing it to do some more work. Useful 
[shortcuts for in the editor](http://www.southampton.ac.uk/~fangohr/blog/spyder-the-python-ide.html)
(these are in Windows-style, including the usual copy-paste shortcuts `Left-CTRL-C`
and `Left-CTRL-V`):

-   `F9` executes the currently highlighted code.
-   `LEFT-CTRL + <RETURN>` executes the current cell (menu entry `Run -> Run cell`).
     A cell is defined as the code between two lines which start with the agreed tag `#%%`.
-   `SHIFT + <RETURN>` executes the current cell and advances the cursor to the next
     cell (menu entry `Run -> Run cell and advance`).
-   `TAB` tries to complete the word/command you have just typed.

The [Spyder Integrated Development Environment](https://pythonhosted.org/spyder/) 
(IDE) has of course lots of parameters which you can tune to your liking. The main
setting that you might want to change is if the graphics are generated "inline" in
the iPython console, or as separate windows. Go to `Tools` > `Preferences` > `Console`>
`Graphics` > `Graphics backend`. Change from "inline" to "automatic" if you prefer the separate windows.

iPython
-------
You might be able to convince spyder to run iPython.
And here are some useful iPython "magic" commands that you can use in the iPython
console on the right (but not in the scripts). Most of these are identical
to what you would use in the terminal. (*Note*: these commands do not work in a Python console.)

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
-   edit a file
```python
    edit FBP.par
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


