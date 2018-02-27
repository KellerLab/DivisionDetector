import os
from gunpowder import *

data_dir = '../../01_data/140521'
samples = [
    '100',
    '120',
    '240',
    '250',
    # '350',
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

    raw = ArrayKey('RAW')
    divisions = PointsKey('DIVISIONS')
    division_balls = ArrayKey('DIVISION_BALLS')
    prediction = ArrayKey('DIVISION_PREDICTION')
    loss_gradient = ArrayKey('DIVISION_PREDICTION')

    sources = tuple(

        (
            KlbSource(
                os.path.join(
                    data_dir,
                    'SPM00_TM000%s_CM00_CM01_CHN00.fusedStack.corrected.shifted.klb'%sample),
                raw,
                ArraySpec(interpolatable=True)),
            CsvPointsSource(
                os.path.join(
                    data_dir,
                    'divisions_%s.txt'%sample),
                divisions) +
            Pad({divisions: None})
        ) +
        MergeProvider() +
        Normalize(raw, 1.0/500) +
        RandomLocation(ensure_nonempty=divisions)

        for sample in samples
    )

    pipeline = (
        sources +
        RandomProvider() +
        # TODO:
        # * augment
        # * pre-cache
        RasterizePoints(
            divisions,
            division_balls,
            raster_settings=RasterizationSettings(ball_radius=20)) +
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
                net_io_names['gt_labels']: division_balls,
            },
            # Map outputs of the network to arrays that will be created by
            # Train.
            outputs={
                net_io_names['labels']: prediction
            },
            # Map gradients of outputs wrt to the loss to arrays that will be
            # created by Train.
            gradients={
                net_io_names['labels']: loss_gradient
            }) +
        Snapshot({
            raw: 'volumes/raw',
            division_balls: 'volumes/divisions'
        },
        every=1) +
        PrintProfilingStats()
    )

    request = BatchRequest()
    request.add(raw, Coordinate((200, 200, 200)))
    request.add(division_balls, Coordinate((200, 200, 200)))

    with build(train_pipeline) as b:

        print("Starting training...")

        # Request as many batches as the number of iterations.
        for i in range(max_iteration - trained_until):
            b.request_batch(request)

if __name__ == "__main__":

    set_verbose(True)
    train_until(1)
