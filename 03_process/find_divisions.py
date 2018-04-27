from __future__ import print_function
import h5py
import numpy as np
import json
import sys
import os
from blob_centers import find_blob_centers
from max_points import find_max_points

# the minimal size of a blob
blob_size_threshold = 10

def find_divisions(
        setup,
        iteration,
        sample,
        frame,
        output_filename,
        method='blob_centers',
        method_args=None,
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

        output_filename (string):

                Name of the JSON file to store the results in.

        method (string):

                'blob_centers': Find blobs by thresholding, take center as
                detection.

                'max_points': Apply non-max suppression to find local maxima.

        method_args (dict):

                Arguments passed to either ``method``.
    '''

    if not method_args:
        method_args = {}

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

    print("Finding detections...")

    if method == 'blob_centers':
        detections, blobs = find_blob_centers(predictions, **method_args)
    elif method == 'max_points':
        detections, blobs = find_max_points(predictions, **method_args)
    else:
        raise RuntimeError("Unkown method %s"%method)

    with h5py.File(prediction_filename, 'r+') as f:

        try:

            if 'volumes/blobs' in f:
                del f['volumes/blobs']

            ds = f.create_dataset(
                'volumes/blobs',
                data=blobs[np.newaxis,:],
                dtype=np.uint64,
                compression='gzip')

            ds.attrs['offset'] = offset
            ds.attrs['resolution'] = resolution

        except:

            print("Failed to store blobs...")

    # correct for offset and resolution
    detections = {
        label: {
            'center': tuple(
                c*r+o
                for c, o, r
                in zip(center, offset[1:], resolution[1:])),
            'score': float(score)
        }
        for label, (center, score) in detections.items()
    }

    result = {
        'divisions': detections
    }

    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":

    args_file = sys.argv[1]
    with open(args_file, 'r') as f:
        args = json.load(f)
    find_divisions(**args)
