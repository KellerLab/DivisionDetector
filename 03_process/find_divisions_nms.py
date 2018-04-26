from __future__ import print_function
from scipy.ndimage.filters import median_filter
from scipy.signal import argrelextrema
from scipy.ndimage import measurements, label
import h5py
import numpy as np
import json
import sys
import os

# the minimal size of a blob
blob_size_threshold = 100

def find_peaks(predictions, blob_prediction_threshold):

    # smooth out "U-Net noise"
    # print("Median-filtering prediction...")
    # predictions = median_filter(predictions, size=3)
    print("Finding maxima...")
    max_loc = argrelextrema(predictions, np.greater_equal)
    size = predictions.shape
    predictions_filtered = np.zeros(shape=size)
    print("Applying NMS...")
    pl = max_loc[0].size
    for i in range(max_loc[0].size):
        predictions_filtered[max_loc[0][i]][max_loc[1][i]][max_loc[2][i]] = predictions[max_loc[0][i]][max_loc[1][i]][max_loc[2][i]]
        print(str(i)+"/"+str(pl))
    print("Finding blobs...")
    blobs = predictions_filtered > blob_prediction_threshold
    labels, num_blobs = label(blobs)
    print("Found %d blobs after NMS"%num_blobs)

    print("Finding peaks, sizes, and maximal values...")
    label_ids = np.arange(1, num_blobs + 1)
    centers = measurements.center_of_mass(blobs, labels, index=label_ids)
    sizes = measurements.sum(blobs, labels, index=label_ids)
    maxima = measurements.maximum(predictions_filtered, labels, index=label_ids)

    peaks = {
        label: (center, max_value)
        for label, center, size, max_value in zip(label_ids, centers, sizes, maxima)
        if size >= blob_size_threshold
    }

    return (peaks, labels)

def find_divisions(
        setup,
        iteration,
        sample,
        frame,
        blob_prediction_threshold,
        thresholds,
        output_basenames,
        *args,
        **kwargs):
    '''Find all divisions in the predictions of a frame.
    Args:
        setup (string):
                The name of the setup.
        iteration (int):
                The training iteration.
        sample (string):
                The video to find divisions for.
        frame (int):
                The frame in the video to find divisions for.
        blob_prediction_threshold (float):
                The prediction threshold to find blobs.
        thresholds (list of float):
                Thresholds in the range [0, 1] to apply.
        output_basenames (list of strings):
                Basenames of the files to store the results in, one for each
                threshold. The extension '.json' will be added to each basename.
    '''

    prediction_filename = os.path.join(
        'processed',
        setup,
        str(iteration),
        sample + '_' + str(frame) + '.hdf')

    print("Reading predictions...")
    with h5py.File(prediction_filename, 'r') as f:
        ds = f['volumes/divisions']
        predictions = np.array(ds[0])
        offset = ds.attrs['offset']
        resolution = ds.attrs['resolution']

    print("Finding peaks...")
    peaks, labels = find_peaks(predictions, blob_prediction_threshold)

    with h5py.File(prediction_filename, 'r+') as f:

        try:

            if 'volumes/blobs' in f:
                del f['volumes/blobs']

            ds = f.create_dataset(
                'volumes/blobs',
                data=labels[np.newaxis,:],
                dtype=np.uint64,
                compression='gzip')

            ds.attrs['offset'] = offset
            ds.attrs['resolution'] = resolution

        except:

            print("Failed to store blobs...")

    print("Storing results...")
    for threshold, outfile_basename in zip(thresholds, output_basenames):

        threshold_peaks = {
            label: {
                'center': tuple(
                    c*r+o
                    for c, o, r
                    in zip(center, offset[1:], resolution[1:])),
                'max_value': float(max_value)
            }
            for label, (center, max_value) in peaks.items()
            if max_value >= threshold
        }

        result = {
            'divisions': threshold_peaks
        }
        with open(outfile_basename + '.json', 'w') as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":

    args_file = sys.argv[1]
    with open(args_file, 'r') as f:
        args = json.load(f)
    find_divisions(**args)
