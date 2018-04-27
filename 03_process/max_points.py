import numpy as np
from scipy.ndimage import measurements, label, maximum_filter
from scipy.ndimage.filters import gaussian_filter

def sphere(radius):

    grid = np.ogrid[tuple(slice(-r, r + 1) for r in radius)]
    dist = sum([
        a.astype(np.float)**2/r**2
        for a, r in zip(grid, radius)
    ])
    return (dist <= 1)

def find_max_points(predictions, radius, sigma=None, min_score_threshold=0):
    '''Find all points that are maximal withing a sphere of ``radius`` and are
    strictly higher than min_score_threshold. Optionally smooth the prediction
    with sigma.'''

    # smooth predictions
    if sigma is not None:
        print("Smoothing predictions...")
        predictions = gaussian_filter(predictions, sigma, mode='constant')

    print("Finding maxima...")
    max_filtered = maximum_filter(predictions, footprint=sphere(radius))
    maxima = max_filtered == predictions

    print("Applying NMS...")
    predictions_filtered = np.zeros_like(predictions)
    predictions_filtered[maxima] = predictions[maxima]

    print("Finding blobs...")
    blobs = predictions_filtered > min_score_threshold
    labels, num_blobs = label(blobs)

    print("Found %d blobs after NMS"%num_blobs)

    print("Finding centers, sizes, and maximal values...")
    label_ids = np.arange(1, num_blobs + 1)
    centers = measurements.center_of_mass(blobs, labels, index=label_ids)
    sizes = measurements.sum(blobs, labels, index=label_ids)
    maxima = measurements.maximum(predictions, labels, index=label_ids)

    centers = {
        label: { 'center': center, 'score': max_value }
        for label, center, size, max_value in zip(label_ids, centers, sizes, maxima)
    }

    return (centers, labels)
