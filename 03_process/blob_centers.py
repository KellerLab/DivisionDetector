import numpy as np
from scipy.ndimage import measurements, label

def find_blob_centers(
        predictions,
        blob_prediction_threshold,
        blob_size_threshold):

    # smooth out "U-Net noise"
    # print("Median-filtering prediction...")
    # predictions = median_filter(predictions, size=3)

    print("Finding blobs...")
    blobs = predictions > blob_prediction_threshold
    labels, num_blobs = label(blobs)
    print("Found %d blobs"%num_blobs)

    print("Finding centers, sizes, and maximal values...")
    label_ids = np.arange(1, num_blobs + 1)
    centers = measurements.center_of_mass(blobs, labels, index=label_ids)
    sizes = measurements.sum(blobs, labels, index=label_ids)
    maxima = measurements.maximum(predictions, labels, index=label_ids)

    centers = {
        label: { 'center': center, 'score': max_value }
        for label, center, size, max_value in zip(label_ids, centers, sizes, maxima)
        if size >= blob_size_threshold
    }

    return (centers, labels)
