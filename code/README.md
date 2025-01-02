# Code Directory

This directory contains the main source code for the eDisk model fitting project, including single-disk and binary-disk modeling.

## Files

- **`model.py`**: Contains the current implementation of single-disk models with enhancements for performance and flexibility.
- **`model_binary.py`**: Developed for binary-disk systems, extending the capabilities of single-disk modeling.

## My Contributions

This project builds upon an initial codebase provided by the eDisk collaboration group. My primary contributions to the project include:

1. **Binary Disk Model Implementation**:
   - Designed and implemented a framework to extend the single disk models to fit binary systems.
   - The `model_binary.py` script introduces functionality for combining two independent disk models into a binary configuration in uv-space.
   - All modifications and enhancements by me are clearly marked with `Revision` comments in the code.
   - Another similar **Trinary Disk Model** has also been developed.

2. **New Model Development**:
   - Added and revisted several models to handle advanced disk structures, such as:
     - **Asymmetric Rings/Gap**
       - Solved the angular discontinuity problem in the original model
     - **Stacked 3-Disk Model**
       - Special model developed for fitting L1489IRS's wrapped structure.
     - **Spiral Arm Model**
       - Added independent d.o.f. for spiral arm's decay to the original model
     - Other useful models and combinations for specific sources fitting.
     - `model.py` and `model_binary.py` might not included all the implemented models.

3. **Code Optimization**:
   - Improved compatibility of existing scripts
   - Allowed printing out native model for profile analysis

All listed enhancements and extensions were designed and implemented by me. The original codebase is included in the [`original/`](original/) directory for reference.
