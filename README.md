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



Jupyter Lab
-----------
If you are more familiar with the Jupyter Lab interface, rather than the Jupyter Notebook one, you can access this by navigating to `<IP>:<port>/lab` (i.e., changing the path part of the url to `/lab`). We don't yet recommend this unless you know what you are doing as we cannot support both interfaces.

Contents
========
Notebooks are located in the [`notebooks`](./notebooks) subdirectory. Each have a README file that you should read beforehand, as some notebooks have special requirements (e.g., the order that they're run in).

- [Introductory](./notebooks/Introductory/) notebooks are designed to familiarise you with Python, Jupyter, SIRF, and patterns seen in the other notebooks.
- [Geometry](./notebooks/Geometry/) notebooks contain lessons on how SIRF deals with spatial geometry of images.
- [PET](./notebooks/PET/) notebooks contain lessons on using SIRF for PET reconstruction and simulation.
- [MR](./notebooks/MR/) notebooks contain lessons on using SIRF for MR reconstruction and simulation.
- [Reg](./notebooks/Reg/) notebooks contain lessons on using SIRF's image registration and resampling tools.
- [Synergistic](./notebooks/Synergistic/) notebooks contain lessons demonstrating more advanced features of SIRF for synergistic image reconstruction.

