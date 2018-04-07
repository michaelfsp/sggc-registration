import scipy
import scipy.signal
import scipy.ndimage
import numpy as np
import cv2
from cv2 import dft as fftn
from cv2 import idft as ifftn

from past.utils import old_div
from caiman.motion_correction import _compute_phasediff, _upsampled_dft

import pylab as pl


def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb = None, shifts_ub = None,
                         max_shifts = (10,10), opencv=True):
    """

    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Parameters:
    ----------
    src_image : ndarray
        Reference image.

    target_image : ndarray
        Image to register.  Must be same dimensionality as ``src_image``.

    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)

    space : string, one of "real" or "fourier"
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.

    Returns:
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.

    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    Raise:
    ------
     NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

     ValueError("Error: images must really be same size for "
                         "register_translation")

     ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    References:
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must really be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        if opencv:
            src_freq_1 = fftn(src_image,flags=cv2.DFT_COMPLEX_OUTPUT+cv2.DFT_SCALE)
            src_freq  = src_freq_1[:,:,0]+1j*src_freq_1[:,:,1]
            src_freq   = np.array(src_freq, dtype=np.complex128, copy=False)
            target_freq_1 = fftn(target_image,flags=cv2.DFT_COMPLEX_OUTPUT+cv2.DFT_SCALE)
            target_freq  = target_freq_1[:,:,0]+1j*target_freq_1[:,:,1]
            target_freq = np.array(target_freq , dtype=np.complex128, copy=False)
        else:
            src_image_cpx = np.array(src_image, dtype=np.complex128, copy=False)
            target_image_cpx = np.array(target_image, dtype=np.complex128, copy=False)
            src_freq = np.fft.fftn(src_image_cpx)
            target_freq = np.fft.fftn(target_image_cpx)

    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if opencv:

        image_product_cv = np.dstack([np.real(image_product),np.imag(image_product)])
        cross_correlation = fftn(image_product_cv,flags=cv2.DFT_INVERSE+cv2.DFT_SCALE)
        cross_correlation = cross_correlation[:,:,0]+1j*cross_correlation[:,:,1]
    else:
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        cross_correlation = np.fft.ifftn(image_product)

    # Locate maximum
    new_cross_corr  = np.abs(cross_correlation)

    if (shifts_lb is not None) or (shifts_ub is not None):

        if  (shifts_lb[0]<0) and (shifts_ub[0]>=0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0],:] = 0
        else:
            new_cross_corr[:shifts_lb[0],:] = 0
            new_cross_corr[shifts_ub[0]:,:] = 0

        if  (shifts_lb[1]<0) and (shifts_ub[1]>=0):
            new_cross_corr[:,shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:,:shifts_lb[1]] = 0
            new_cross_corr[:,shifts_ub[1]:] = 0
    else:

        new_cross_corr[max_shifts[0]:-max_shifts[0],:] = 0

        new_cross_corr[:,max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(old_div(axis_size, 2)) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:

        src_amp = old_div(np.sum(np.abs(src_freq) ** 2), src_freq.size)
        target_amp = old_div(np.sum(np.abs(target_freq) ** 2), target_freq.size)
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = old_div(np.round(shifts * upsample_factor), upsample_factor)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(old_div(upsampled_region_size, 2.0))
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
                              np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape),
                          dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + old_div(maxima, upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, cross_correlation, src_freq,_compute_phasediff(CCmax)

def sgolay2d ( z, window_size, order, derivative = None ):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2 ) / 2.0

    if  window_size % 2 == 0:
        raise ValueError( 'window_size must be odd' )

    if window_size ** 2 < n_terms:
        raise ValueError( 'order is too high for the window size' )

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ ( k - n, n ) for k in range( order + 1 ) for n in range( k + 1 ) ]

    # coordinates of points
    ind = np.arange( -half_size, half_size + 1, dtype = np.float64 )
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1] ).reshape( window_size ** 2, )

    # build matrix of system of equation
    A = np.empty( ( window_size ** 2, len( exps ) ) )
    for i, exp in enumerate( exps ):
        A[:, i] = ( dx ** exp[0] ) * ( dy ** exp[1] )

    '''
    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros( ( new_shape ) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs( np.flipud( z[1:half_size + 1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs( np.flipud( z[-half_size - 1:-1, :] ) - band )
    # left band
    band = np.tile( z[:, 0].reshape( -1, 1 ), [1, half_size] )
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size + 1] ) - band )
    # right band
    band = np.tile( z[:, -1].reshape( -1, 1 ), [1, half_size] )
    Z[half_size:-half_size, -half_size:] = band + np.abs( np.fliplr( z[:, -half_size - 1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs( np.flipud( np.fliplr( z[1:half_size + 1, 1:half_size + 1] ) ) - band )
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs( np.flipud( np.fliplr( z[-half_size - 1:-1, -half_size - 1:-1] ) ) - band )

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs( np.flipud( Z[half_size + 1:2 * half_size + 1, -half_size:] ) - band )
    # bottom left corner
    band = Z[-half_size:, half_size].reshape( -1, 1 )
    Z[-half_size:, :half_size] = band - np.abs( np.fliplr( Z[-half_size:, half_size + 1:2 * half_size + 1] ) - band )
    '''

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv( A )[0].reshape( ( window_size, -1 ) )
        return scipy.signal.fftconvolve( Z, m, mode = 'valid' )
    elif derivative == 'col':
        c = np.linalg.pinv( A )[1].reshape( ( window_size, -1 ) )
        return scipy.signal.fftconvolve( Z, -c, mode = 'valid' )
    elif derivative == 'row':
        r = np.linalg.pinv( A )[2].reshape( ( window_size, -1 ) )
        return scipy.signal.fftconvolve( Z, -r, mode = 'valid' )
    elif derivative == 'both':
        c = np.linalg.pinv( A )[1].reshape( ( window_size, -1 ) )
        r = np.linalg.pinv( A )[2].reshape( ( window_size, -1 ) )
        #import ipdb; ipdb.set_trace()
        #return Z, scipy.signal.fftconvolve( Z, -r, mode = 'valid' ), scipy.signal.fftconvolve( Z, -c, mode = 'valid' )
        #return scipy.signal.filtfilt( Z, -r), scipy.signal.filtfilt( Z, -c)
        #return scipy.ndimage.filters.convolve(z, -r, mode = 'nearest'), scipy.ndimage.filters.convolve(z, -c, mode = 'nearest')
        return scipy.signal.fftconvolve( z, -r, mode = 'valid' ), scipy.signal.fftconvolve( z, -c, mode = 'valid' )


def sggc_registration(src_image, target_image, window_size=7, order=3, use_scipy=False, hamming=True, shifts_lb = None, shifts_ub = None, max_shifts = (10,10)):
    if use_scipy:
        half_size = window_size // 2

        src_image_sgd_x_y = []
        src_image_sgd_x_y.append(scipy.signal.savgol_filter(src_image, window_size, order, deriv=1, mode='nearest', axis=1)[half_size:-half_size])
        src_image_sgd_x_y.append(scipy.signal.savgol_filter(src_image, window_size, order, deriv=1, mode='nearest', axis=0)[half_size:-half_size])

        target_image_sgd_x_y = []
        target_image_sgd_x_y.append(scipy.signal.savgol_filter(target_image, window_size, order, deriv=1, mode='nearest', axis=1)[half_size:-half_size])
        target_image_sgd_x_y.append(scipy.signal.savgol_filter(target_image, window_size, order, deriv=1, mode='nearest', axis=0)[half_size:-half_size])

    else:
        src_image_sgd_x_y = sgolay2d(src_image, window_size, order, derivative='both')
        target_image_sgd_x_y = sgolay2d(target_image, window_size, order, derivative='both')

    #import ipdb; ipdb.set_trace()

    src_image_sgd = src_image_sgd_x_y[0] + src_image_sgd_x_y[1]*1j
    target_image_sgd = target_image_sgd_x_y[0] + target_image_sgd_x_y[1]*1j

    src_image_sgd -= src_image_sgd.mean()
    src_image_sgd /= src_image_sgd.max()

    target_image_sgd -= target_image_sgd.mean()
    target_image_sgd /= target_image_sgd.max()

    src_image_sgd_freq = np.fft.fft2(src_image_sgd)
    target_image_sgd_freq = np.fft.fft2(target_image_sgd)

    cross_product = src_image_sgd_freq * target_image_sgd_freq.conj()
    shape = src_image_sgd_freq.shape

    sggc = np.fft.ifft2(cross_product)

    if hamming:
        h1 = scipy.signal.hamming(sggc.shape[0], sym=True)
        h2 = scipy.signal.hamming(sggc.shape[1], sym=True)
        ham2d = np.sqrt(np.outer(h1,h2))

        sggc = np.multiply(sggc, ham2d)

    cross_correlation = sggc
    new_cross_corr = np.abs(cross_correlation)

    if (shifts_lb is not None) or (shifts_ub is not None):

        if  (shifts_lb[0]<0) and (shifts_ub[0]>=0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0],:] = 0
        else:
            new_cross_corr[:shifts_lb[0],:] = 0
            new_cross_corr[shifts_ub[0]:,:] = 0

        if  (shifts_lb[1]<0) and (shifts_ub[1]>=0):
            new_cross_corr[:,shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:,:shifts_lb[1]] = 0
            new_cross_corr[:,shifts_ub[1]:] = 0
    else:

        new_cross_corr[max_shifts[0]:-max_shifts[0],:] = 0

        new_cross_corr[:,max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(old_div(axis_size, 2)) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    #src_amp = old_div(np.sum(np.abs(src_image_sgd_freq) ** 2), src_image_sgd_freq.size)
    #target_amp = old_div(np.sum(np.abs(target_image_sgd_freq) ** 2), target_image_sgd_freq.size)
    #CCmax = cross_correlation.max()

    for dim in range(src_image_sgd_freq.ndim):
        if shape[dim] == 1:
            print('bad')
            shifts[dim] = 0


    #import ipdb; ipdb.set_trace()
    return shifts, sggc

