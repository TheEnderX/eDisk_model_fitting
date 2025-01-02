# Modified Model and Fitting Code for ALMA Large Program eDisk

This repository contains modified and extended models and fitting codes based on the [pdspy package](https://github.com/psheehan/pdspy) by Patrick Sheehan. These modifications were developed during the research with the eDisk program at UIUC under instruction of Professor Leslie W. Looney.
The initial codebase, developed by Patrick Sheehan and other members of the eDisk model group, serves as the foundation for this project. Enhancements and extensions have been added to support binary disks modeling and advanced analysis.


## Overview
This project extends the original model fitting code functionality with:
- Availability for binary (or more) disks model fitting
- Modifying and creating new models, such as asymmetry models, stacked-disk models, etc.
- Creating native model for profile analysis
- Other modification of the existing codebase for better compatibility and performance

## Dependencies
- Python 3
- [pdspy](https://github.com/psheehan/pdspy)
- [dynesty](https://github.com/joshspeagle/dynesty)
- Other Python Libraries: numpy, matplotlib, mpi4py

## Directories
- [`code/`](code/): Contains all current source code.
  - [`model.py`](code/model.py): Single-disk modeling code.
  - [`model_binary.py`](code/model_binary.py): Binary-disk modeling code.
  - [`model_trinary.py`](code/model_trinary.py): Trinary-disk modeling code.
- [`original/`](original/): Original unmodified code provided at the start of the project.
