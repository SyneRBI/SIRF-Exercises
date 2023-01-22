# Instructions for participants

This instruction contain documentation and links to get started with the exercises of SIRF.

- [Start Here](#start-here)

- [Start a jupyter notebook server for the SIRF exercises](#Start-a-jupyter-notebook-server-for-the-SIRF-exercises)
    - [Using an Azure Client](#using-and-azure-client-(if-available))
    - [Using the VM](#using-the-vm)
    - [Using Docker](#using-docker)
    - [Using your own installed SIRF and SIRF-exercises](#using-your-own-installed-sirf-and-sirf-exercises )

- [Starting a terminal via Jupyter](#starting-a-terminal-via-jupyter)

- [Start a Gadgetron server](#start-a-gadgetron-server)

- [Getting the Data](#getting-the-data)

- [Get started with the course!](#get-started-with-the-course)

- [Appendix of useful info](#appendix)

The SIRF documentation can be found [here](https://github.com/SyneRBI/SIRF/wiki/Software-Documentation).
***The current version of these exercises needs SIRF v3.2.0 (SPECT needs v3.3.0 (pre-)release)***

Documentation is in the form of MarkDown files (`*.md`), which are simple text files which you open from the Jupyter notebook, but they look nicer when browsing to [GitHub](https://github.com/SyneRBI/SIRF-Exercises/tree/master/).

## Start Here

We are using Python for the exercises. Python is an open-source interactive language, 
a bit like MATLAB. We provide Python scripts for the exercises, so you should be fine.
Nevertheless, it would be best to read a Python tutorial first, see the [Appendices](#appendices).

We use
[Jupyter notebooks](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).
If you have never used Jupyter notebooks,
you should [read the official documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html).
A useful introduction to the notebook interface [can be found here](http://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html).

There are several ways to get SIRF and its exercises running for the training course. 

1. Accessing a remote server prepared for the training course (could be Azure, STFC cloud or others). This is specific for organized training courses. Please check with your instructors. 
2. Installing and running the SIRF Virtual Machine (VM).
3. Installing and running the SIRF Docker image.
4. Installing and building SIRF and the SIRF-Exercises on your machine from source.

We recommend that if you are installing this for a training course, you use any of the first three options, as installing SIRF from source is harder than the rest.
The VM works well in any operating system, Docker works well in Linux and MacOS, (Windows support for our Docker images is currently untested).

Instructions for all (except the training course specific server, as this will be given in the training course) can be found at https://github.com/SyneRBI/SIRF/wiki/How-to-obtain-SIRF.
Instructions to self-build the exercises (once you have SIRF built from source) can be found in this repository at [the installation instructions page](INSTALL.md).

Once you have SIRF and the exercises on your system, or access to a server with SIRF, it is time to get them running. 

The next sections contain instructions to start the Jupyter notebook server with the SIRF exercises for all the different installation options. In following steps, we will start a Gadgetron server and download data.

## Start a jupyter notebook server for the SIRF exercises

***Warning:** these instructions are when using JupyterLab as opposed to the "classic" notebook
interface. If you choose to use the classic interface, you will have to modify the notebooks
marginally by replacing `%matplotlib widget` with `%matplotlib notebook`.
See also the [iPython section](#ipython) below.


### Using an Azure client (if available)

The web-address should be something like https://sirf1....cloudapp.azure.com:9999/. See local instructions of your training sessoin.
*Do not forget the `https`*. You will need to accept the secure exception. The Jupyter notebook password is `virtual`.

If the instructors tell you, you might have to create a terminal via the jupyter notebook and type `update_VM.sh`.
Later in the course, you can use this terminal to start `gadgetron`.

### Using an STFC Cloud instance (if available)
Follow instructions given elsewhere.

### Using the VM

1. start the VM from VirtualBox (user `sirfuser`, password `virtual`)
2. Open terminal (either via `Activities` or pressing `ctrl-alt-T`) and type

  ```bash
  # optionally update to the latest release of SIRF and the SIRF-Exercises
  update_VM.sh
  jupyter lab
  ```

Then open a web-browser on your computer (i.e. the host) and point it to http://localhost:8888 (fill in the password or the token).

If this fails, you could try to use web browser in the VM instead.
```bash
   sudo apt install firefox
   jupyter lab --browser firefox
```

### Using Docker

The instructions to start Docker and SIRF are documented in the [Docker instructions at SIRF](https://github.com/SyneRBI/SIRF-SuperBuild/blob/master/docker/README.md), please follow those to start it. 
Docker is easiest in Linux, so if you are not familiar with Docker and are running on Windows, we suggest using the VM instead. 

Please note that for at present (at least up to SIRF 3.4), you need to point your (host) web-browser to http://localhost:9999 (fill in the `virtual` password).

### Using your own installed SIRF and SIRF-exercises 

You have a jupyter server (as you followed the [installation instructions](INSTALL.md)) so just use
   ```bash
   cd /wherever/you/installed/it/SIRF-Exercises
   jupyter lab
   ```

## Starting a terminal via Jupyter

It is often useful to run commands in a shell where the Python kernels run (i.e. on Azure/VM/STFC cloud/docker).
- Jupyter "classic": on the "Home" tab, click on `New` on the right, and choose `Terminal`
- JupyterLab: go to the Launcher (click on the `+` sign top-left), and choose `Terminal`.

Copy-paste in the terminal window can be tricky. Normally, you can shift+right click in the terminal and then select from th epop-up menu. See also the [JupyterLab doc](https://jupyterlab.readthedocs.io/en/stable/user/terminal.html#copy-paste).

## Start a Gadgetron server

SIRF uses Gadgetron for MR reconstruction. You will need to start a "server" such that SIRF can communicate to it. Docker already starts this automatically, but if you are using anything else you need to start Gadgetron yourself.

Open a new terminal (for the Jupyter interface, see above) and type
```bash
gadgetron
```
This starts `gadgetron`. Leave the terminal window open.

You can kill the server at the end by going back to the terminal and pressing `Ctrl-C`.


## Getting the Data

Some exercises use data that you will need. In the cloud and in Azure, we provide the data you need for the tests, but otherwise, you will need to download it.
There are download scripts available for that, available in the `SIRF-Exercises/scripts` folder. The introductory notebooks contain
cells for running the script, but you can also do this from the command line (see above on how to start a terminal from Jupyter).

- Get example data
  ```bash
  cd /wherever/you/installed/it/SIRF-Exercises
  scripts/download_data.sh -m -p
  ```
  On the VM and Azure, the exercises are installed in `~/devel`, in docker in `/devel`, and in the STFC Cloud in `~`. (Apologies for that!).
  
  This will be a ~3 GB download.

  Note that if you want to run notebooks in MR/Old_notebooks (not recommended),
  you will have to get some more data
  ```bash
  scripts/download_data.sh -m -p -o
  ```

  Note that the `download_data.sh` script has several options allowing you to put data in other places. Run
  ```bash
  scripts/download_data.sh -h
  ```
  for more information.

## Get started with the course

All notebooks are located in several subdirectories of [`notebooks`](./notebooks) . Each have a `README.md` file that you should read beforehand, as some notebooks have special requirements (e.g., the order that they're run in). Note that you can open a `README.md` from the Jupyter notebook, but they look nicer when browsing to [GitHub](https://github.com/SyneRBI/SIRF-Exercises/tree/master/notebooks).

- [Introductory](./notebooks/Introductory/) notebooks are designed to familiarise you with Python, Jupyter, SIRF, and patterns seen in the other notebooks.
- [Geometry](./notebooks/Geometry/) notebooks contain lessons on how SIRF deals with spatial geometry of images.
- [PET](./notebooks/PET/) notebooks contain lessons on using SIRF for PET reconstruction and simulation.
- [SPECT](./notebooks/SPECT/) notebooks contain lessons on using SIRF for SPECT reconstruction and simulation.
- [MR](./notebooks/MR/) notebooks contain lessons on using SIRF for MR reconstruction and simulation.
- [Reg](./notebooks/Reg/) notebooks contain lessons on using SIRF's image registration and resampling tools.
- [Synergistic](./notebooks/Synergistic/) notebooks contain lessons demonstrating more advanced features of SIRF for synergistic image reconstruction.

Start with the [introductory notebooks](notebooks/Introductory/) and the associated [README.md](notebooks/Introductory/README.md).



---------------------------

# Appendix

- [Python Basics](#python)
- [iPython](#ipython)
- [Jupyter notebook manipulations](#jupyter-notebook-manipulations)
- [File extensions](#file-extensions)
- [A note on keyboard short-cuts inside a VirtualBox VM](#a-note-on-keyboard-short-cuts-inside-a-virtualbox-vm)
- [The Linux terminal](#linux-terminal)


## Python Basics

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

## iPython

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
       - when using a Jupyter Notebook server ("classic" mode)
         ```     
         %matplotlib notebook
         ```
       - when using a JupyterLab server
         ```     
         %matplotlib widget
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

## Jupyter notebook manipulations

The initial web-page that you will see looks like a file browser
(the *Jupyter Notebook dashboard*).
Click on `notebooks`, and drill down until you find a file with the extension `.ipynb`
that looks of interest, and click on that. This should open a new tab in your
web browser (or JupyterLab window) with the notebook open, all ready to run.

You will normally work by executing each *cell*
bit by bit, and then editing it to do some more work. Useful 
shortcuts:

-   `LEFT-CTRL + <RETURN>` executes the current cell.
-   `SHIFT + <RETURN>` executes the current cell and advances the cursor to the next cell.
-   `TAB` tries to complete the word/command you have just typed.
-   `Enter` edits "edit" mode to change a cell, `Esc` exits "edit" mode to "command" mode.
-    In "command" mode, press `A` to create a new cell Above, or `B` below your current cell. You can also use `C`, `X`, `V`.
-    Other keyboard shortcuts:
     - When using Jupyter "classic" mode, pressing `H` in "command" mode gives you a useful list of shortcuts.
     - When using JupyterLab, you need to go to the Advanced Settings Editor item in the Settings menu, then select Keyboard Shortcuts in the Settings tab. You probably want to check the `notebook` category. See the [JupyterLab doc](https://jupyterlab.readthedocs.io/en/stable/user/interface.html#keyboard-shortcuts)/

Jupyter notebooks (normally) run iPython, [see the section below](#iPython) for some useful commands.

Every open notebook has its own kernel. Closing a notebook tab does **not** stop its kernel.
Use the `File` menu ("classic": `Close and halt`, JupyterLab: `Close and shutdown notebook`).

***Warning:*** Due to a limitation in SIRF (and ISMRMRD) you cannot have 2 Python sessions simultaneously accessing
the same MR HDF5 file.
You have to use “File->Close and ...”  after you’ve finished with a notebook (or just “Kernel->Shutdown”). In JupyterLab, you can also use the [Running panel in the sidebar](https://jupyterlab.readthedocs.io/en/stable/user/running.html).

***Warning:*** Clicking `Logout` will mean all sessions are closed and you will have to start again. Clicking 
`Quit` means the server will quit and you will be in trouble. (On Azure, the server should restart after a few seconds,
but not on the VM, so start it again as above).



## File extensions

- `.h5`: HDF5 file with MR data or images
- `.hv`: Interfile header for an image (volume)
- `.ahv`: (ignore) old-style Interfile header for an image
- `.v`: raw data of an image (in floats)
- `.nii` or `.nii.gz`: image files in Nifti format.
- `.hs`: Interfile header for PET or SPECT acquisition data (sinograms)
- `.s`: raw data of PET acquisition data (in floats)
- `.py`: Python file
- `.ipynb`: jupyter notebook
- `.par`: STIR parameter files.
- `.md`: text file with documentation (in Markdown format)

### Siemens data files

These are after extraction from DICOM. Please check [our Wiki](https://github.com/SyneRBI/SIRF/wiki) for information.

- `.dat`: raw MR data file
- `.l` and `.l.hdr`: list mode data and header
- `.n` and `.n.hdr`: normalisation data and header
- `.s` and `.s.hdr`: "sinogram" (i.e. acquisition data) and header

Note that sometimes the PET data files are called `.bf` ("binary file").

Always pass the header files to SIRF, not the name of the data file.

## A note on keyboard short-cuts inside a VirtualBox VM

On Windows and Linux, VirtualBox sets the "host-key" by default to `Right-CTRL` on Windows/Linux, so
unless you change this, you have to use `Left-CTRL` to "send" the `CTRL`-keystroke
to the Virtual Machine. This is why we wrote to use `Left-CTRL`.

## Linux Terminal

If you have never used a Linux/Unix terminal before, have a look at 
[a tutorial](https://help.ubuntu.com/community/UsingTheTerminal).

You can use `UPARROW` to go to previous commands, and use copy-paste shortcuts 
`Left-CTRL-SHIFT-C` and `Left-CTRL-SHIFT-V`.
