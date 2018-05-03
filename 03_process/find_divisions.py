from __future__ import print_function
import h5py
import numpy as np
import json
import sys
import os
import time
from blob_centers import find_blob_centers
from max_points import find_max_points
from skimage.measure import block_reduce

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
        downsample=None,
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

        downsample (tuple, optional):

                Downsample with these factors (e.g., ``(3,2,2)`` will downsample
                z by 3, x and y by 2).
    '''

    if not method_args:
        method_args = {}

    prediction_filename = os.path.join(
        'processed',
        setup,
        str(iteration),
        sample + '_' + str(frame) + '.hdf')

    print("Reading predictions...")
    start = time.time()
    with h5py.File(prediction_filename, 'r') as f:
        ds = f['volumes/divisions']
        predictions = np.array(ds[0])
        offset = tuple(ds.attrs['offset'][1:])
        resolution = tuple(ds.attrs['resolution'][1:])
    print("%.3fs"%(time.time()-start))

    print("resolution of predictions: %s"%(resolution,))

    if downsample:
        downsample = tuple(downsample)
        print("Downsampling predictions...")
        start = time.time()
        predictions = block_reduce(predictions, downsample, np.max)
        resolution = tuple(r*d for r, d in zip(resolution, downsample))
        print("%.3fs"%(time.time()-start))
        print("new resolution of predictions: %s"%(resolution,))

    print("Finding detections...")

    if method == 'blob_centers':
        detections, blobs = find_blob_centers(predictions, resolution, **method_args)
    elif method == 'max_points':
        detections, blobs = find_max_points(predictions, resolution, **method_args)
    else:
        raise RuntimeError("Unkown method %s"%method)

    # with h5py.File(prediction_filename, 'r+') as f:

        # try:

            # if 'volumes/blobs' in f:
                # del f['volumes/blobs']

            # ds = f.create_dataset(
                # 'volumes/blobs',
                # data=blobs[np.newaxis,:],
                # dtype=np.uint64,
                # compression='gzip')

            # ds.attrs['offset'] = (1,) + offset
            # ds.attrs['resolution'] = (1,) + resolution

        # except:

            # print("Failed to store blobs...")

    # print("Storing predictions...")
    # start = time.time()
    # with h5py.File(output_filename[:-4] + 'hdf', 'w') as f:

        # ds = f.create_dataset(
            # 'volumes/predictions',
            # data=predictions[np.newaxis,:],
            # compression='gzip')
        # ds.attrs['offset'] = (1,) + offset
        # ds.attrs['resolution'] = (1,) + resolution
    # print("%.3fs"%(time.time()-start))

    print("Storing detections...")
    start = time.time()
    # correct for offset and resolution
    detections = {
        label: {
            'center': tuple(
                c*r+o
                for c, o, r
                in zip(data['center'], offset, resolution)),
            'score': float(data['score'])
        }
        for label, data in detections.items()
    }

    result = {
        'divisions': detections
    }

    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=2)
    print("%.3fs"%(time.time()-start))

if __name__ == "__main__":

    args_file = sys.argv[1]
    with open(args_file, 'r') as f:
        args = json.load(f)
    find_divisions(**args)
