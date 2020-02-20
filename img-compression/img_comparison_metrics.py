import numpy as np

float_type = 'float64'  # uint8 pixel values will be converted to float_type first for float arithmetic


def mse(img1, img2):
    """

    :param img1: Numpy array holding the first RGB image batch.
    :param img2: Numpy array holding the second RGB image batch.
    :return:
    """
    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    return mse


def psnr(img1, img2, max_val=255, mse=None):
    """

    :param img1:
    :param img2:
    :param max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    :param mse: pre-computed MSE, if available
    :return:
    """
    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    if mse is None:
        mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


# from msssim import MultiScaleSSIM
# Below copied from msssim.py:
# !/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of MS-SSIM.
Adapted from https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py
for properly handling image batches.
Slightly different version here: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py

Usage:

python msssim.py --original_image=original.png --compared_image=distorted.png
"""
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).

    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.

    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)), axis=(1, 2, 3))
    cs = np.mean(v1 / v2, axis=(1, 2, 3))
    return ssim, cs


def ms_ssim(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
            k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.

    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
      weights: List of weights for each level; if none, use five levels and the
        weights from the original paper.

    Returns:
      MS-SSIM score between `img1` and `img2`.

    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    batch_size = len(img1)
    mssim = np.empty([levels, batch_size])
    mcs = np.empty([levels, batch_size])
    for i in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1, im2, max_val=max_val, filter_size=filter_size,
            filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim[i] = ssim
        mcs[i] = cs
        filtered = [convolve(im, downsample_filter, mode='reflect')
                    for im in [im1, im2]]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]  # downsampling by factor of 2
    return (np.prod(mcs[0:levels - 1] ** weights[0:levels - 1, np.newaxis], axis=0) *
            (mssim[levels - 1] ** weights[levels - 1]))


def compare_two_imgs(img1, img2, use_tf=False):
    if isinstance(img1, np.ndarray):
        assert isinstance(img2, np.ndarray)
        assert np.issubdtype(img1.dtype, np.integer)
        assert np.issubdtype(img2.dtype, np.integer)
        x = img1
        y = img2
        comparison_modes = ('RGB',)
    else:
        from PIL import Image

        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')

        x = np.asarray(img1)
        y = np.asarray(img2)
        x_yc = np.asarray(img1.convert('YCbCr'))  # YCbCr, H x W x 3, uint8
        y_yc = np.asarray(img2.convert('YCbCr'))  # YCbCr, H x W x 3, uint8

        comparison_modes = ('RGB', 'Luma', 'Chroma')

    results = {}
    for mode in comparison_modes:
        if mode == 'RGB':
            x_comp = x
            y_comp = y
        elif mode == 'Luma':
            x_comp = x_yc[..., 0:1]
            y_comp = y_yc[..., 0:1]
        elif mode == 'Chroma':
            x_comp = x_yc[..., 1:]
            y_comp = y_yc[..., 1:]
        else:
            raise NotImplementedError

        # reshape into batch of 1
        x_comp = x_comp[None, ...]
        y_comp = y_comp[None, ...]

        if not use_tf:
            results['MSE (%s)' % mode] = mse(x_comp, y_comp)
            results['PSNR (%s)' % mode] = psnr(x_comp, y_comp, max_val=255)
            results['MS-SSIM (%s)' % mode] = ms_ssim(x_comp, y_comp, max_val=255)
        else:
            import tensorflow as tf

            x_comp = tf.convert_to_tensor(x_comp)
            y_comp = tf.convert_to_tensor(y_comp)

            mses = tf.reduce_mean(tf.squared_difference(tf.cast(x_comp, 'float32'), tf.cast(y_comp, 'float32')),
                                  axis=(1, 2, 3))
            psnrs = tf.image.psnr(y_comp, x_comp, 255)
            ssims = tf.image.ssim(y_comp, x_comp, 255)
            msssims = tf.image.ssim_multiscale(y_comp, x_comp,
                                               255)  # can crash: https://github.com/tensorflow/tensorflow/issues/33840
            # Also crashes with "Invalid argument: Computed output size would be negative" when input images are too small (this
            # is expected behavior: the default setting (https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale)
            # uses 5 2x downsampling scales and a filter size of 11, so for a small image, say 64x64, 3 2x scaling reduce it to
            # 8x8, which is smaller than kernel size 11)
            tensors = [mses,
                       psnrs,
                       ssims,
                       msssims,
                       ]
            if not tf.executing_eagerly():
                # if sess is not None:
                with tf.Session() as sess:
                    # with sess as sess:
                    tensors = sess.run(tensors)
            [mses, psnrs, ssims, msssims] = tensors
            results['MSE (%s)' % mode] = mses
            results['PSNR (%s)' % mode] = psnrs
            results['SSIM (%s)' % mode] = ssims
            results['MS-SSIM (%s)' % mode] = msssims

    for key, val in results.items():
        assert len(val) == 1
        results[key] = float(val[0])

    return results


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # High-level options.
    # parser.add_argument(
    #     "--verbose", "-V", action="store_true",
    #     help="If set, use tf.logging.INFO; otherwise use tf.logging.WARN, for tf.logging")

    parser.add_argument("img1")
    parser.add_argument("img2")
    parser.add_argument('--use_tf', action="store_true", help='If set, will use tf')

    args = parser.parse_args()

    results = compare_two_imgs(args.img1, args.img2, args.use_tf)

    import json

    out = json.dumps(results, indent=4, sort_keys=True)
    print(out)
