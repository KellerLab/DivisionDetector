from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
import malis
import os
import math
import json
import tensorflow as tf

data_dir = '../../01_data'
samples = [
    '002',
    '004',
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

    # The input and output sizes of the network in nanometer.
    voxel_size = Coordinate((200, 130, 130))
    input_size = Coordinate((58, 196, 196))*voxel_size
    output_size = Coordinate((2, 92, 92))*voxel_size
    context = (input_size - output_size)/2

    # The arrays we are going to use.
    raw = ArrayKey('RAW')
    gt_labels = ArrayKey('GT_LABELS')
    loss_weights = ArrayKey('LOSS_WEIGHTS')
    pred_labels = ArrayKey('PREDICTED_LABELS')
    loss_gradient = ArrayKey('LOSS_GRADIENT')

    # The request for one training batch
    request = BatchRequest()
    request.add(raw, input_size)
    request.add(gt_labels, output_size)
    request.add(loss_weights, output_size)

    # If we make a snapshot (see blow), also include these arrays in the
    # request:
    snapshot_request = BatchRequest({
        pred_labels: request[gt_labels],
        loss_gradient: request[gt_labels]
    })

    # Create a tuple of sources to chose from.
    data_sources = tuple(

        # Read source arrays from HDF5
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                raw: 'volumes/raw',
                gt_labels: 'volumes/labels/nuclei',
            },
            array_specs = {
                gt_labels: ArraySpec(interpolatable=False)
            }
        ) +

        # Add a padding with value 0 of the size of the context around raw. This
        # way, we can use the whole raw ROI for training.
        Pad({raw: context}) +

        # Ensure raw values are in [0, 1]. Does nothing if they arleady are.
        Normalize(raw) +

        # Pick a random location for each request.
        RandomLocation() +

        # Reject batches that have no foreground label.
        Reject(gt_labels, min_masked=0.001)

        for sample in samples
    )

    train_pipeline = (

        # The tuple of sources.
        data_sources +

        # Picks a random source for each request.
        RandomProvider() +

        # Elastically deform all arrays in the batch.
        ElasticAugment([10,6,6], [1,1,1], [0,math.pi/2.0], subsample=8) +

        # Flip and rotate.
        SimpleAugment(transpose_only_xy=True) +

        # Add some intensity changes.
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +

        # Create a scale map to balance the gradient contribution between the
        # positive and negative samples.
        BalanceLabels(
            labels=gt_labels,
            scales=loss_weights) +

        # Move raw values to [-1, 1].
        IntensityScaleShift(raw, 2,-1) +

        # Precache batches from the pipeline above in several processes. This is
        # useful to keep the Train node busy, such that it never has to wait for
        # IO and augmentations.
        PreCache(
            cache_size=40,
            num_workers=10) +

        # Perform one training iteration.
        Train(
            # The network to use.
            'unet',
            # Name of the optimzer.
            optimizer=net_io_names['optimizer'],
            # Name of the operator that computes the loss.
            loss=net_io_names['loss'],
            # Map inputs of the network to arrays we provide.
            inputs={
                net_io_names['raw']: raw,
                net_io_names['gt_labels']: gt_labels,
                net_io_names['loss_weights']: loss_weights,
            },
            # Map outputs of the network to arrays that will be created by
            # Train.
            outputs={
                net_io_names['labels']: pred_labels
            },
            # Map gradients of outputs wrt to the loss to arrays that will be
            # created by Train.
            gradients={
                net_io_names['labels']: loss_gradient
            }) +

        # Get raw back into [0,1] (for snapshots)
        IntensityScaleShift(raw, 0.5, 0.5) +

        # Every 100 iterations, save the current training batch, prediction, and
        # gradient. This is very useful to check that everything works as
        # expected.
        Snapshot({
                raw: 'volumes/raw',
                gt_labels: 'volumes/labels/nuclei',
                pred_labels: 'volumes/labels/pred_nuclei',
                loss_gradient: 'volumes/loss_gradient',
            },
            every=100,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) +

        # Print some useful stats
        PrintProfilingStats(every=10)
    )

    with build(train_pipeline) as b:

        print("Starting training...")

        # Request as many batches as the number of iterations.
        for i in range(max_iteration - trained_until):
            b.request_batch(request)

    print("Training finished")

if __name__ == "__main__":
    set_verbose(False)
    iteration = int(sys.argv[1])
    train_until(iteration)
