# Instructions for instructors

This is a basic list of things to do and tell people during the course.

## Start
1. Show how to connect to Azure or JupyterHub clients (if available). Should be something like https://sirf1....cloudapp.azure.com:9999/.
*Do not forget the `https`*. People will need to accept the secure exception. Jupyter notebook password is `virtual`.

2. Navigate to [.../SIRF-Exercises/DocForParticipants.md](DocForParticipants.md) to show how to open files and give them something to read.

3. Show how to start the VM. Show about resizing, and scaling (VB menu: `View`->`Scaling factor`), and how to connect: open terminal (either via `Activities` or pressing `ctrl-alt-T`) and type
  ```bash
  update_VM.sh # if required
  cd ~/devel/SIRF-Exercises/
  scripts/download_data.sh -m -p
  jupyter lab
  ```
  Then open a web-browser on your local laptop and point it to http://localhost:8888 (fill in the password or the token).

4. Start a docker instance.
Then open a web-browser on your local laptop and point it to http://localhost:9999 (fill in the password).

5. From now on, everything is the same whatever option you used (except for Gadgetron)

## Basic jupyter notebook manipulations
1. Open notebook and tell them about execution and keyboard shortcuts (via menu).
2. Show how to create a terminal ("Launcher" (or first "+"), `Terminal`). Use this to start the Gadgetron. Say this is not necessary on Docker as already started.
   ```bash
   gadgetron&
   ```
3. Warn that you cannot have 2 Python sessions simultaneously accessing
the same MR HDF5 file, and that closing a notebook does not stop its kernel. Work-arounds:
 - use “Kernel -> "Shutdown kernel" in one notebook
 - use "File" -> "Close and Shutdown Notebook"
 - go to the [Running panel in the sidebar](https://jupyterlab.readthedocs.io/en/stable/user/running.html).

4. Warn that pressing `Logout` will mean all sessions are closed and you will have to start again. Warn that pressing
`Quit` means the server will quit and they will be in trouble. (On Azure, the server should restart after a few seconds,
but not on the VM).

## Get started with the course!
Start with the [introductory notebooks](notebooks/Introductory/) and the associated [README.md](notebooks/Introductory/README.md).



