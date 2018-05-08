import numpy as np
import time
import math
from scipy.ndimage import measurements, label, maximum_filter
from scipy.ndimage.filters import gaussian_filter

def sphere(radius):

    grid = np.ogrid[tuple(slice(-r, r + 1) for r in radius)]
    dist = sum([
        a.astype(np.float)**2/r**2
        for a, r in zip(grid, radius)
    ])
    return (dist <= 1)

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
        print("Smoothing predictions...")
        sigma = tuple(float(s)/r for s, r in zip(sigma, resolution))
        print("voxel-sigma: %s"%(sigma,))
        start = time.time()
        predictions = gaussian_filter(predictions, sigma, mode='constant')
        print("%.3fs"%(time.time()-start))

    print("Finding maxima...")
    start = time.time()
    radius = tuple(int(math.ceil(float(ra)/re)) for ra, re in zip(radius, resolution))
    print("voxel-radius: %s"%(radius,))
    max_filtered = maximum_filter(predictions, footprint=sphere(radius))

    # for the following, we only process the center frame
    center = predictions.shape[0]/2

    maxima = max_filtered[center] == predictions[center]
    print("%.3fs"%(time.time()-start))

    print("Applying NMS...")
    start = time.time()
    predictions_filtered = np.zeros_like(predictions[center])
    predictions_filtered[maxima] = predictions[center][maxima]
    print("%.3fs"%(time.time()-start))

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
        label: { 'center': center, 'score': max_value }
        for label, center, size, max_value in zip(label_ids, centers, sizes, maxima)
    }

    return (centers, labels)
