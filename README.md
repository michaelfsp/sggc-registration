SGGC-registration
=================
A Python implementation of HajiRassouliha et al. (see references) phase-based subpixel image registration algorithm using Savitzky-Golay differentiators.

This code is an adaptation of the function `register_translation`, which is the CaImAn (https://github.com/flatironinstitute/CaImAn/blob/master/caiman/motion_correction.py) implementation of the Guizar-Sicairos et al. algorithm for subpixel image registration (see references).

It uses the function `sgolay2d` from https://github.com/mpastell/SciPy-CookBook/blob/master/ipython/SavitzkyGolay.py

# References
* HajiRassouliha, A., Taberner, A. J., Nash, M. P., & Nielsen, P. M. F. (2017). Subpixel phase-based image registration using Savitzky-Golay differentiators in gradient-correlation. Computer Vision and Image Understanding, (August), 0–1. https://doi.org/10.1016/j.cviu.2017.11.003

* Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R. (2008). Efficient subpixel image registration algorithms. Optics Letters, 33(2), 156–158. https://doi.org/10.1364/OL.33.000156
