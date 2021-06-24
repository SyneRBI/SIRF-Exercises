# SIRF-Exercises

This material is intended for practical demonstrations using 
[SIRF](https://github.com/CCPPETMR/SIRF/wiki/Software-Documentation) on PET and MR Image Reconstruction.

This repository contains exercises to get you going
with SIRF. Please check the [INSTALL.md](INSTALL.md) first.
There is also some basic information on [CIL](https://www.ccpi.ac.uk/CIL) functionality to show similarities with SIRF.

Authors:
- Kris Thielemans (this document and PET exercises)
- Christoph Kolbitsch (MR exercises)
- Johannes Mayer (MR exercises)
- David Atkinson (MR and geometry exercises)
- Evgueni Ovtchinnikov (PET and MR exercises)
- Edoardo Pasca (overall check and clean-up)
- Richard Brown (PET and registration exercises)
- Daniel Deidda and Palak Wadhwa (HKEM exercise)
- Ashley Gillman (overall check and clean-up)

This software is distributed under an open source license, see [LICENSE.txt](LICENSE.txt)
for details.

# Getting started

Full instructions in getting started is in our [documentation for participants](https://github.com/CCPPETMR/SIRF-Exercises/blob/master/DocForParticipants.md).  


Gentle request
--------------
If you are attending the course, ***please read this before the course.*** 



Starting the jupyter server
---------------------------
To start with the exercises, you need to start the jupyter server. Depending on how/where you run these exercises, this will
be in different ways:

### On the VM using the web browser of your host system (recommended)
In the VM terminal, type
  ```bash
    jupyter notebook
```
You will get a message that ends `Or copy and paste one of these URLs:
        http://(vagrant or 127.0.0.1):8888/?token=axxxxcf837a4ab78aa13275dc893af9b91143225c226xxxx`

On your (host) laptop/desktop, open a web-browser and either
- navigate to [http://localhost:8888](http://localhost:8888). The password should be `virtual`.
- use the full address from the message (including the `token=...` part). You will need to edit the `(vagrant or 127.0.0.1)` to `127.0.0.1`.
- use the address http://localhost:8888 and then when requested, copy and paste the token value, which in this example would be `axxxxcf837a4ab78aa13275dc893af9b91143225c226xxxx`. 

### On the VM using a web browser in the VM
You will need to install a web browser on the VM such as Firefox. 
```bash
   sudo apt install firefox
```
Now do

```bash
   jupyter notebook --browser firefox
```
which should start your web-browser automatically.

