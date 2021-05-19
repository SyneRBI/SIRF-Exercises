# Instructions for participants

This is a basic "getting started" for course participants. We will run these
exercises either via the Azure cloud or in a Virtual Machine (VM). The start-up phase is different for Azure and the VM, but the rest of the usage should be the same.

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
  ./download_data.sh -m -p
  cd ~/devel/SIRF-Exercises/
  jupyter notebook
```
Then open a web-browser on your laptop and point it to http://localhost:8888 (fill in the password or the token).


## jupyter notebook manipulations

[README.md](README.md) contains links on how to use the Jupyter notebook.

***Warning:*** Clicking `Logout` will mean all sessions are closed and you will have to start again. Clicking 
`Quit` means the server will quit and you will be in trouble. (On Azure, the server should restart after a few seconds,
but not on the VM, so start it again as above).

## start a Gadgetron server

SIRF uses Gadgetron for MR reconstruction. You will need to start a "server" such that SIRF can communicate to it.
You can do this from in the Jupyter interface by creating a terminal that runs on Azure/VM/docker (on the "Home" tab, click on `New` on the right, and choose `Terminal`) and typing
```bash
gadgetron&
```
This starts `gadgetron` and runs it in the "background". You can leave the terminal window open and navigate
back to "Home".

## Get started with the course!
Start with the [notebooks/Introductory/](introductory notebooks) and the associated [README.md](notebooks/Introductory/README.md).

*Warning*: Due to a limitation in the ISMRMRD library that we use to read MR HDF5 files, you cannot have
2 Python sessions simultaneously accessing the same MR HDF5 file.
The only work-around is to use “File->Close and halt”  after you’ve finished with a notebook (or just “Kernel->Shutdown”).



