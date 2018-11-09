from __future__ import print_function
import h5py
import numpy as np
import json
import sys
import os
import time
from max_points import find_max_points
from skimage.measure import block_reduce

#Applies NMS to find divisions from model output (per-voxel predictions)

def find_divisions(
        setup,
        iteration,
        sample,
        frame,
        output_filename,
        radius,
        sigma=None,
        min_score_threshold=0,
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

        radius (tuple):
                The radius in (t,z,y,x) to consider for NMS

        sigma (tuple):
                Smoothing factor to apply to predictions before NMS

        min_score_threshold (float):

                Only selects local maxima with scores strictly greater than this threshold

        downsample (tuple, optional):

                Downsample with these factors (e.g., ``(3,2,2)`` will downsample
                z by 3, x and y by 2).
    '''

    context = radius[0]

    frames = list(range(frame - context, frame + context + 1))

    prediction_filenames = [
        os.path.join(
            'processed',
            setup,
            str(iteration),
            sample + '_' + str(f) + '.hdf')
        for f in frames]

    print("Reading predictions...")
    start = time.time()
    predictions = []
    for prediction_filename in prediction_filenames:
        print("\t%s"%prediction_filename)
        with h5py.File(prediction_filename, 'r') as f:
            ds = f['volumes/divisions']
            predictions.append(np.array(ds[0]))
            offset = tuple(ds.attrs['offset'][1:])
            resolution = tuple(ds.attrs['resolution'])
    predictions = np.array(predictions)
    print("%.3fs"%(time.time()-start))

    print("resolution of predictions: %s"%(resolution,))

    if downsample:
        downsample = tuple(downsample)
        print("Downsampling predictions...")
        start = time.time()
        predictions = np.array([
            block_reduce(predictions[f], downsample, np.max)
            for f in range(predictions.shape[0])])
        resolution = (resolution[0],) + tuple(r*d for r, d in zip(resolution[1:], downsample))
        print("%.3fs"%(time.time()-start))
        print("new resolution of predictions: %s"%(resolution,))

    print("Finding detections...")
    detections, blobs = find_max_points(predictions, resolution, radius, sigma, min_score_threshold)
    print("Storing detections...")
    start = time.time()

    # correct for offset and resolution
    detections = {
        label: {
            'center': tuple(
                c*r+o
                for c, o, r
                in zip(data['center'], offset, resolution[1:])),
            'score': float(data['score'])
        }
        for label, data in detections.items()
    }

    result = {
        'divisions': detections,
        'configuration': {
            'setup': setup,
            'iteration': iteration,
            'sample': sample,
            'frame': frame,
            'radius': radius,
            'sigma': sigma,
            'min_score_threshold': min_score_threshold,
            'downsample': downsample,
        }
    }

    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=2)
    print("%.3fs"%(time.time()-start))

if __name__ == "__main__":

    args_file = sys.argv[1]
    with open(args_file, 'r') as f:
        args = json.load(f)
    find_divisions(**args)
