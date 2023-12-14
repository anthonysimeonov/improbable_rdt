# Improbable Real (Robot) Deployment Tools \[RDT\]

Python Package containing tools for interfacing with the real world in learning experiments. Avoiding the use of ROS here. 
- Interfacing with robots (primarily the Franka Panda)
    - Utilities for using the `polymetis` package for interacing with the Panda
- Interfacing with RGB(-D) cameras (such as RealSense, potentially Kinect)
- Motion planning
- Inverse kinematics
- LCM message passing
- etc. 

## Cloning (don't forget the `--recurse` flag!!)

```
git clone --recurse git@github.com:anthonysimeonov/improbable_rdt.git
```

## Common setup for `polymetis``
```
# install mamba (so much faster than conda)
conda install -n base -c conda-forge mamba

# use mamba (or conda, if you didn't install mamba)
mamba env create -f polymetis_conda_env.yml

# some other installs
pip install cython

# install rdt
pip install -e .

# install LCM
cd /path/to/lcm  # ~/repos/source/lcm on my machine
cd lcm-python
pip install .
```

## More instructions
For spacemouse
```
numpy
git+https://github.com/cheng-chi/spnav
termcolor
atomics
scipy
```
