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
- Johannes Mayer (MR exercises)
- Richard Brown (registration exercises)

This software is distributed under an open source license, see [LICENSE.txt](LICENSE.txt)
for details.

Gentle request
--------------
If you are attending the course, ***please read this before the course.*** A brief summary of the steps to get started
is in our [documentation for participants](https://github.com/CCPPETMR/SIRF-Exercises/blob/master/DocForParticipants.md).  


Introduction
============

The SIRF documentation can be found [here](https://github.com/CCPPETMR/SIRF/wiki/Software-Documentation).
***The current version of these exercises needs SIRF v2.0.***

We are using Python for the exercises. Python is an open-source interactive language, 
a bit like MATLAB. We provide Python scripts for the exercises, so you should be fine.
Nevertheless, it would be best to read a Python tutorial first, see the [Appendices](#appendices).

We use
[Jupyter notebooks](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).
If you have never used Jupyter notebooks,
you could [read the official documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html).
A useful introduction to the notebook interface [can be found here](http://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html).

The rest of this document contains some information to get started.


Getting started
================
Some exercises use data that you will need. Check the [installation instructions](INSTALL.md).

The MR exercises will need you to start a Gadgetron server. If you have installed SIRF properly,
you should be able to type in a terminal
```sh
gadgetron
```
You can kill it at the end by going back to the terminal and pressing `Ctrl-C`.

Starting the jupyter server
---------------------------
To start with the exercises, you need to start the jupyter server. Depending on how/where you run these exercises, this will
be in different ways:

### On the VM using the web browser of your host system (recommended)
In the VM terminal, type
  ```bash
    cd ~/devel/SIRF-Exercises
    jupyter notebook  --no-browser --ip 0.0.0.0
```
You will get a message that ends `Or copy and paste one of these URLs:
        http://(vagrant or 127.0.0.1):8888/?token=axxxxcf837a4ab78aa13275dc893af9b91143225c226xxxx`
        
On your laptop/desktop, open a web-browser and use the full address from the message (including the `token=...` part). You will need to edit the `(vagrant or 127.0.0.1)` to `127.0.0.1`. Alternatively, use the address http://localhost:8888 and then when requested, copy and paste the token value, which in this example would be `axxxxcf837a4ab78aa13275dc893af9b91143225c226xxxx`. 

### On the VM using a web browser in the VM
You will need to install a web browser on the VM such as Firefox. 
```bash
   sudo apt install firefox
```
Now do

```bash
   cd ~/devel/SIRF-Exercises
   jupyter notebook
```
which should start your web-browser automatically.

### You have installed SIRF and the exercises yourself

You have a jupyter server (as you followed the [installation instructions](INSTALL.md) so just use
```bash
   cd /wherever/you/installed/it/SIRF-Exercises
   jupyter notebook
```

Using the notebooks
===================
The initial web-page that you will see looks like a file browser
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

Jupyter notebooks (normally) run iPython, [see the section below](#iPython) for some useful commands.

Check the [jupyter doc on closing a notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html#close-a-notebook-kernel-shut-down).
(Note: it is *not* just closing the browser tab).

***Warning:*** Due to a limitation in SIRF (and ISMRMRD) you cannot have 2 Python sessions simultaneously accessing
the same MR HDF5 file.
You have to use “File->Close and halt”  after you’ve finished with a notebook (or just “Kernel->Shutdown”).

Appendices
==========

File extensions
---------------

- `.dat`: Siemens raw MR data file
- `.h5`: HDF5 file with MR data or images
- `.hv`: Interfile header for an image (volume)
- `.ahv`: (ignore) old-style Interfile header for an image
- `.v`: raw data of an image (in floats)
- `.hs`: Interfile header for PET acquisition data (sinograms)
- `.s`: raw data of PET acquisition data (in floats)
- `.py`: Python file
- `.ipynb`: jupyter notebook
- `.par`: STIR parameter files.
- `.md`: text file with documentation (in Markdown format)

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

We use [matplotlib](https://matplotlib.org/), a python module that allows you to make plots almost like in MATLAB.
Check [here](https://github.com/patvarilly/dsghent-python-for-data-scientists-2016/blob/master/notebooks/02_MatplotlibAndSeaborn.ipynb) for some examples.

iPython
-------
The jupyter notebooks will normally be running iPython, although this depends a bit on your configuration.
iPython is Python with a few extensions to make the experience a bit friendlier.

Here are some useful iPython "magic" commands that you can use in the iPython
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

