from __future__ import print_function
import sys
from gunpowder import *
import os
import numpy as np

data_dir = '.'

def render_blobs(frame):

    # define arrays and point sets we'll need
    raw = ArrayKey('RAW')
    divisions = PointsKey('DIVISIONS')
    division_balls = ArrayKey('DIVISION_BALLS')

    voxel_size = Coordinate((1, 5, 1, 1))
    chunk_size = Coordinate((1, 200, 100, 100))

    chunk_request = BatchRequest()
    chunk_request.add(raw, chunk_size)
    chunk_request.add(division_balls, chunk_size)

    source = KlbSource(
            os.path.join(
                data_dir,
                'SPM00_TM000*_CM00_CM01_CHN00.fusedStack.corrected.shifted.klb'),
            raw,
            ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size))
    with build(source):
        raw_spec = source.spec[raw]

    pipeline = (
        (
            # provide raw
            source,

            # provide divisions
            CsvPointsSource(
                os.path.join(
                    data_dir,
                    'all_divisions.txt'),
                divisions,
                scale=voxel_size) +
            Pad({divisions: None})
        ) +
        MergeProvider() +
        Normalize(raw) +
        RasterizePoints(
            divisions,
            division_balls,
            array_spec=ArraySpec(voxel_size=voxel_size),
            settings=RasterizationSettings(
                radius=10,
                mode='ball'
            )) +
        # increase intensity for visualization
        IntensityScaleShift(raw, scale=100.0, shift=0) +
        PrintProfilingStats(every=10) +
        Scan(chunk_request) +
        Snapshot({
                raw: 'volumes/raw',
                division_balls: 'volumes/divisions',
            },
            dataset_dtypes={
                raw: np.float32,
                division_balls: np.float32,
            },
            compression_type='gzip',
            every=1,
            output_dir='.',
            output_filename='%04d.hdf'%frame)
    )

    with build(pipeline) as b:

        # get origingal raw ROI, limited to requested frame
        request_spec = raw_spec.copy()
        offset = request_spec.roi.get_offset()
        shape = request_spec.roi.get_shape()
        request_spec.roi = Roi(
            (frame,) + offset[1:],
            (1,) + shape[1:])

        whole_request = BatchRequest({
                raw: request_spec,
                division_balls: request_spec
            })

        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))

        pipeline.request_batch(whole_request)

if __name__ == "__main__":
    set_verbose(False)
    frame = int(sys.argv[1])
    render_blobs(frame)
