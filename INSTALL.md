Installation instructions
-------------------------
Author: Kris Thielemans

An easy way to run the exercises is to use the SIRF Virtual Machine where
all of this has been done for you, but you can install all of this yourself
of course. Below are some brief instructions.



Installing SIRF and the exercises
---------------------------------

You will need SIRF of course. Please check the [instructions on our wiki](https://github.com/CCPPETMR/SIRF/wiki/How-to-obtain-SIRF).

The SIRF-exercises themselves don't need further installation. Just get them, preferably via

    git clone https://github.com/CCPPETMR/SIRF-Exercises

Other utilities
--------------
### Spyder

We recommend that you run the exercises from in an interactive Python
IDE such as [Spyder](https://pythonhosted.org/spyder/), so you will need to install that.
On Ubuntu, the following should work
```
sudo apt-get update
sudo apt-get install spyder
```
We recommend to use iPython as it allows some "magic" commands making life easier. To get this
into spyder, try
```
sudo apt-get install ipython ipython-qtconsole python-zmq
```
Or when using [Anaconda](https://www.anaconda.com/what-is-anaconda/), check
[here for iPython](https://anaconda.org/anaconda/ipython) and
[here for Spyder](https://anaconda.org/anaconda/spyder).

