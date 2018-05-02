import numpy as np
import time
from scipy.ndimage import measurements, label

def find_blob_centers(
        predictions,
        resolution,
        blob_prediction_threshold,
        blob_size_threshold):

    # smooth out "U-Net noise"
    # print("Median-filtering prediction...")
    # predictions = median_filter(predictions, size=3)

    print("Finding blobs...")
    start = time.time()
    blobs = predictions > blob_prediction_threshold
    labels, num_blobs = label(blobs)
    print("%.3fs"%(time.time()-start))
    print("Found %d blobs"%num_blobs)

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
        if size >= blob_size_threshold
    }

    return (centers, labels)
