# Instructions for instructors

This is a basic list of things to do and tell people during the course.

## Start
1. Show how to connect to Azure clients (if available). Should be something like https://sirf1....cloudapp.azure.com:9999/.
*Do not forget the `https`*. People will need to accept the secure exception. Jupyter notebook password is `virtual%1`.

2. Navigate to [.../SIRF-Exercises/DocForParticipants.md](DocForParticipants.md) to show how to open files and give them something to read.

3. Show how to start the VM. Show about resizing, and scaling (VB menu: `View`->`Scaling factor`), and how to connect: open terminal (either via `Activities` or pressing `ctrl-alt-T`) and type
```bash
  update_VM.sh
  cd ~/devel/SIRF-Exercises/scripts/
  ./download_PET_data.sh
  ./download_MR_data.sh
  cd ~/devel/SIRF-Exercises/
  jupyter notebook
```
Then open a web-browser on your local laptop and point it to http://localhost:8888 (fill in the password or the token).

4. From now on, everything is the same when using the cloud or the VM.

## Basic jupyter notebook manipulations
1. Open notebook and tell them about execution and keyboard shortcuts (via menu).
2. Warn that closing the tab leaves a python session running. This can be surprising (for instance with read/write access).
Show how to close the session (either `File->Close and halt` or go "home" and click on the `Running` tab.
3. Show how to create a terminal ("home", `New` on the right, `Terminal`). Use this to start the Gadgetron.
```bash
gadgetron&
```
4. Warn that you cannot have 2 Python sessions simultaneously accessing
the same MR HDF5 file.
The only work-around is to use “File->Close and halt”  after you’ve finished with a notebook (or just “Kernel->Shutdown”).

5. Warn that pressing `Logout` will mean all sessions are closed and you will have to start again. Warn that pressing
`Quit` means the server will quit and they will be in trouble. (On Azure, the server should restart after a few seconds,
but not on the VM).

## Get started with the course!
Start with [notebooks/MR/interactive/a_fully_sampled.ipynb](notebooks/MR/a_fully_sampled.ipynb).



