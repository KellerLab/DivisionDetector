from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
import malis
import os
import math
import json
import tensorflow as tf
import numpy as np

data_dir = '../../01_data/140521'

class RandomLocationExcludeTime(RandomLocation):

    def __init__(
            self,
            raw,
            min_masked=0,
            mask=None,
            ensure_nonempty=None,
            p_nonempty=1.0,
            t=0):

        super(RandomLocationExcludeTime, self).__init__(
            min_masked,
            mask,
            ensure_nonempty,
            p_nonempty)

        self.raw = raw
        self.t = t

    def accepts(self, request):

        return not (request[self.raw].roi.get_begin()[0] <= self.t and
                    request[self.raw].roi.get_end()[0] > self.t)

def train_until(max_iteration):

    # Get the latest checkpoint.
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('net_config.json', 'r') as f:
        net_config = json.load(f)

    # define arrays and point sets we'll need
    raw = ArrayKey('RAW')
    divisions = PointsKey('DIVISIONS')
    division_balls = ArrayKey('DIVISION_BALLS')
    prediction = ArrayKey('DIVISION_PREDICTION')
    loss_gradient = ArrayKey('DIVISION_PREDICTION_LOSS')
    # a point set to ensure we have at least on division in the center of the
    # output
    divisions_center = PointsKey('DIVISIONS_CENTER')

    voxel_size = Coordinate((1, 5, 1, 1))
    input_size = Coordinate(net_config['input_shape'])*voxel_size
    output_size = Coordinate(net_config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(division_balls, output_size)
    # add divisions_center with a smaller ROI than divisions
    # (this is to ensure that there is a division somewhere in the output
    # volume)
    request.add(divisions_center, Coordinate((2, 8, 30, 30)))

    snapshot_request = BatchRequest({
        prediction: request[division_balls],
        loss_gradient: request[division_balls],
    })

    pipeline = (
        (
            # provide raw
            KlbSource(
                os.path.join(
                    data_dir,
                    'SPM00_TM000*_CM00_CM01_CHN00.fusedStack.corrected.shifted.klb'),
                raw,
                ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size)),

            # provide divisions
            CsvPointsSource(
                os.path.join(
                    data_dir,
                    'all_divisions.txt'),
                divisions,
                scale=voxel_size) +
            Pad(divisions, None),

            # provide divisions_center
            CsvPointsSource(
                os.path.join(
                    data_dir,
                    'all_divisions.txt'),
                divisions_center,
                scale=voxel_size) +
            Pad(divisions_center, None)
        ) +
        MergeProvider() +
        Normalize(raw) +
        RandomLocationExcludeTime(
            raw=raw,
            ensure_nonempty=divisions_center,
            p_nonempty=0.5,
            t=360) +
        # ensure that raw is non-zero
        Reject(
            mask=raw,
            min_masked=0.00001) +
        ElasticAugment(
            control_point_spacing=[5,10,10],
            jitter_sigma=[1,1,1],
            rotation_interval=[0,math.pi/2.0],
            subsample=8) +
        SimpleAugment(mirror_only=[1, 2, 3], transpose_only=[2, 3]) +
        IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001) +
        RasterizePoints(
            divisions,
            division_balls,
            array_spec=ArraySpec(voxel_size=voxel_size),
            settings=RasterizationSettings(
                radius=10,
                mode='ball'
            )) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            'unet',
            optimizer=net_config['optimizer'],
            loss=net_config['loss'],
            inputs={
                net_config['raw']: raw,
                net_config['gt_labels']: division_balls,
            },
            outputs={
                net_config['labels']: prediction
            },
            gradients={
                net_config['logits']: loss_gradient
            },
            summary = net_config['summary'],
            log_dir ='logs',
            ) +
        # increase intensity for visualization
        IntensityScaleShift(raw, scale=100.0, shift=0) +
        Snapshot({
                raw: 'volumes/raw',
                division_balls: 'volumes/divisions',
                prediction: 'volumes/prediction',
                loss_gradient: 'volumes/gradient',
            },
            dataset_dtypes={
                division_balls: np.float32
            },
            output_filename='snapshot_{iteration}.hdf',
            every=1000,
            additional_request=snapshot_request) +
        PrintProfilingStats(every=10)
    )

    with build(pipeline) as b:

        print("Starting training...")
        for i in range(max_iteration - trained_until):
            b.request_batch(request)

if __name__ == "__main__":
    set_verbose(False)
    iteration = int(sys.argv[1])
    train_until(iteration)
