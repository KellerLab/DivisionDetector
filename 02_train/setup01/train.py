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

    raw = ArrayKey('RAW')
    divisions = PointsKey('DIVISIONS')
    division_balls = ArrayKey('DIVISION_BALLS')

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
        RasterizePoints(
            divisions,
            division_balls,
            raster_settings=RasterizationSettings(ball_radius=20)) +
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

    with build(pipeline):
        pipeline.request_batch(request)

if __name__ == "__main__":

    set_verbose(True)
    train_until(1)
