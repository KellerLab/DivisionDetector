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
samples = [
    '100', 
    '120',
    '240',
    '250', 
    '360', 
    '400',
]

def train_until(max_iteration):

    # Get the latest checkpoint.
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    # define arrays and point sets we'll need
    raw = ArrayKey('RAW')
    divisions = PointsKey('DIVISIONS')
    division_balls = ArrayKey('DIVISION_BALLS')
    loss_weights = ArrayKey('LOSS_WEIGHTS')
    prediction = ArrayKey('DIVISION_PREDICTION')
    loss_gradient = ArrayKey('DIVISION_PREDICTION_LOSS')
    # a point set to ensure we have at least on division in the center of the
    # output
    divisions_center = PointsKey('DIVISIONS_CENTER')

    voxel_size = Coordinate((5, 1, 1))
    input_size = Coordinate((74, 324, 324))*voxel_size
    output_size = Coordinate((10, 36, 36))*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(division_balls, output_size)
    # add divisions_center with a small ROI
    request.add(divisions_center, Coordinate((10, 10, 10)))

    snapshot_request = BatchRequest({
        prediction: request[division_balls],
        loss_gradient: request[division_balls],
        loss_weights: request[division_balls],
    })
    sources = tuple(
        (
            # provide raw
            KlbSource(
                os.path.join(
                    data_dir,
                    'SPM00_TM000%s_CM00_CM01_CHN00.fusedStack.corrected.shifted.klb'%sample),
                raw,
                ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size)),

            # provide divisions
            CsvPointsSource(
                os.path.join(
                    data_dir,
                    'divisions_%s.txt'%sample),
                divisions,
                scale=voxel_size) +
            Pad({divisions: None}),

            # provide divisions_center
            CsvPointsSource(
                os.path.join(
                    data_dir,
                    'divisions_%s.txt'%sample),
                divisions_center,
                scale=voxel_size) +
            Pad({divisions_center: None})
        ) +
        MergeProvider() +
        Normalize(raw) +
        RandomLocation(ensure_nonempty=divisions_center, p_nonempty = 0.9)

        for sample in samples
    )

    pipeline = (
        sources +
        RandomProvider() +
        ElasticAugment([5,10,10], [1,1,1], [0,math.pi/2.0], subsample=8) +
        SimpleAugment(transpose_only=[1,2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001) +
        #convert from vector image to pixel image
        RasterizePoints(
            divisions,
            division_balls,
            array_spec=ArraySpec(voxel_size=voxel_size),
            settings=RasterizationSettings(radius=5)) +
        #balance affinity, should add it
        BalanceLabels(
           labels=division_balls,
           scales=loss_weights) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        
        Train(
            'unet',
            optimizer=net_io_names['optimizer'],
            loss=net_io_names['loss'],
            inputs={
                net_io_names['raw']: raw,
                net_io_names['gt_labels']: division_balls,
                net_io_names['loss_weights']: loss_weights
            },
        
            outputs={
                net_io_names['labels']: prediction
            },
            gradients={
                net_io_names['labels']: loss_gradient
            
            },
            #should be a tensor, not scalar
            #we must generate a summary first
                 
            summary = net_io_names['summary'],
            log_dir ='logs',
            ) +
            
        Snapshot({
                raw: 'volumes/raw',
                division_balls: 'volumes/divisions',
                prediction: 'volumes/prediction',
                loss_gradient: 'volumes/gradient',
                loss_weights: 'volumes/loss_weights',
            },
            dataset_dtypes={
                division_balls: np.float32
            },
            output_filename='snapshot_{iteration}.hdf',
            every=100,
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
