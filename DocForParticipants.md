# Instructions for participants

This is a basic "getting started" for participants of one of our courses. We will run these
exercises either via the cloud or in a Virtual Machine. The start-up phase is different if you
run one of these, but the rest of the usage should be the same.

Please check our [README](README.md).


## Start

### Using an Azure client (if available). 

The web-address should be something like https://sirf1....cloudapp.azure.com:9999/. See local instructions.
*Do not forget the `https`*. You will need to accept the secure exception. The Jupyter notebook password is `virtual%1`.

If the instructors tell you, you might have to create a terminal via the jupyter notebook and type `update_VM.sh`.
Later in the course, you can use this terminal to start `gadgetron`.

### Using the VM

1. start the VM from VirtualBox (user `sirfuser`, password `virtual`)
2. Open terminal (either via `Activities` or pressing `ctrl-alt-T`) and type
```bash
  update_VM.sh
  cd ~/devel/SIRF-Exercises/scripts/
  ./download_PET_data.sh
  ./download_MR_data.sh
  cd ~/devel/SIRF-Exercises/
  jupyter notebook --no-browser --ip 0.0.0.0
```
Then open a web-browser on your laptop and point it to https://localhost:8888 (fill in the password or the token).


## jupyter notebook manipulations

[README.md](README.md) contains links on how to use the Jupyter notebook.

***Warning:*** Clicking `Logout` will mean all sessions are closed and you will have to start again. Clicking 
`Quit` means the server will quit and you will be in trouble. (On Azure, the server should restart after a few seconds,
but not on the VM, so start it again as above).

## Get started with the course!
Start with [notebooks/MR/interactive/a_fully_sampled.ipynb](notebooks/MR/interactive/a_fully_sampled.ipynb).



