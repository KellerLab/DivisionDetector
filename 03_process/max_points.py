from __future__ import division
import numpy as np
import time
import math
from scipy.ndimage import measurements, label, maximum_filter
from scipy.ndimage.filters import gaussian_filter


def sphere(radius):
    """ Creates a mask (np array of boolean values) that is
    true when the point is within the sphere described by radius,
    and false otherwise.

    :param radius: a tuple with the same number of entries as dimensions in the image
    :return: an np array representing a mask to be used as the footprint in maximum_filter function
    """
    grid = np.ogrid[tuple(slice(-r, r + 1) for r in radius)]
    dist = sum([
        a.astype(np.float)**2/r**2 if r > 0 else 0
        for a, r in zip(grid, radius)
    ])
    return (dist <= 1)


def smooth(predictions, resolution, sigma):
    """ Smooths the predictions using a multidimensional gaussian filter with parameter sigma"""
    print("Smoothing predictions...")
    sigma = tuple(float(s) / r for s, r in zip(sigma, resolution))
    print("voxel-sigma: %s" % (sigma,))
    start = time.time()
    predictions = gaussian_filter(predictions, sigma, mode='constant')
    print("%.3fs" % (time.time() - start))
    return predictions


def find_maxima(predictions, resolution, radius):
    """ Creates a mask (np array of boolean values) that is true
    when the point is a local maxima in the sphere described by radius,
    and false otherwise."""
    print("Finding maxima...")
    start = time.time()
    radius = tuple(int(math.ceil(float(ra) / re)) for ra, re in zip(radius, resolution))
    print("voxel-radius: %s" % (radius,))
    max_filtered = maximum_filter(predictions, footprint=sphere(radius))
    maxima = max_filtered == predictions
    print("%.3fs" % (time.time() - start))
    return maxima


def apply_nms(predictions, maxima):
    """Takes the maxima mask and returns the center frame of predictions with the
    non-maximal points set to zero """
    print("Applying NMS...")
    start = time.time()
    center = predictions.shape[0] // 2
    maxima = maxima[center]
    predictions_filtered = np.zeros_like(predictions[center])
    predictions_filtered[maxima] = predictions[center][maxima]
    print("%.3fs" % (time.time() - start))
    return predictions_filtered


def find_max_points(
        predictions,
        resolution,
        radius,
        sigma=None,
        min_score_threshold=0):
    '''Find all points that are maximal withing a sphere of ``radius`` and are
    strictly higher than min_score_threshold. Optionally smooth the prediction
    with sigma.'''

    # smooth predictions
    if sigma is not None:
        predictions = smooth(predictions, resolution, sigma)

    maxima = find_maxima(predictions, resolution, radius)
    predictions_filtered = apply_nms(predictions, maxima)

    print("Finding blobs...")
    start = time.time()
    blobs = predictions_filtered > min_score_threshold
    labels, num_blobs = label(blobs)
    print("%.3fs"%(time.time()-start))

    print("Found %d blobs after NMS"%num_blobs)

    print("Finding centers, sizes, and maximal values...")
    start = time.time()
    label_ids = np.arange(1, num_blobs + 1)
    centers = measurements.center_of_mass(blobs, labels, index=label_ids)
    sizes = measurements.sum(blobs, labels, index=label_ids)
    maxima = measurements.maximum(predictions, labels, index=label_ids)
    print("%.3fs"%(time.time()-start))

    centers = {
        str(label): { 'center': center, 'score': max_value }
        for label, center, size, max_value in zip(label_ids, centers, sizes, maxima)
    }

    return (centers, labels)
