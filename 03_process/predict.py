import sys
from gunpowder import *
from gunpowder.tensorflow import Predict
import os
import json
import numpy as np

data_dir = '../01_data'

def predict(setup, iteration, sample, frame):

    print("Predicting divisions of setup %s, iteration %s, in sample%s "
          "frame %d"%(setup, iteration, sample, frame))

    checkpoint = os.path.join('../02_train', setup, 'unet_checkpoint_%d'%iteration)
    if not os.path.isfile(checkpoint + '.meta'):
        checkpoint = os.path.join('../02_train', setup, 'train_net_checkpoint_%d'%iteration)
    graph = os.path.join('../02_train', setup, 'test_net.meta')
    if not os.path.isfile(graph):
        graph = None
    config = os.path.join('../02_train', setup, 'test_net_config.json')
    if not os.path.isfile(config):
        config = os.path.join('../02_train', setup, 'net_config.json')
    with open(config, 'r') as f:
        net_config = json.load(f)

    voxel_size = Coordinate((1, 5, 1, 1))
    input_size = Coordinate(net_config['input_shape'])*voxel_size
    output_size = Coordinate(net_config['output_shape'])*voxel_size
    context = (input_size - output_size)//2

    raw = ArrayKey('RAW')
    divisions = ArrayKey('DIVISIONS')

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(divisions, output_size)

    source = KlbSource(
            os.path.join(
                data_dir,
                str(sample),
                'SPM00_TM000*_CM00_CM01_CHN00.fusedStack.corrected.shifted.klb'),
            raw,
            ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size))
    with build(source):
        raw_spec = source.spec[raw]

    pipeline = (
        source +
        Normalize(raw) +
        Pad(raw, context) +
        Predict(
            checkpoint,
            graph=graph,
            inputs = {
                net_config['raw']: raw,
            },
            outputs = {
                net_config['labels']: divisions,
            },
            array_specs = {
                divisions: ArraySpec(
                    roi=raw_spec.roi,
                    voxel_size=raw_spec.voxel_size,
                    dtype=np.float32
                )
            }) +
        PrintProfilingStats(every=100) +
        Scan(chunk_request) +
        Snapshot({
                raw: 'volumes/raw',
                divisions: 'volumes/divisions',
            },
            dataset_dtypes={
                raw: np.float32,
                divisions: np.float32,
            },
            every=1,
            output_dir=os.path.join('processed', setup, '%d'%iteration),
            output_filename=sample+'_'+str(frame)+'.hdf',
            compression_type='gzip')
    )

    with build(pipeline):

        # predict divisions in origingal raw ROI, limited to requested frame
        divisions_spec = raw_spec.copy()
        offset = divisions_spec.roi.get_offset()
        shape = divisions_spec.roi.get_shape()
        divisions_spec.roi = Roi(
            (frame,) + offset[1:],
            (1,) + shape[1:])

        raw_spec.roi = divisions_spec.roi.grow(context, context)

        whole_request = BatchRequest({
                # raw: raw_spec,
                divisions: divisions_spec
            })

        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))

        pipeline.request_batch(whole_request)

if __name__ == "__main__":
    setup = sys.argv[1]
    iteration = int(sys.argv[2])
    sample = sys.argv[3]
    frame = int(sys.argv[4])
    predict(setup, iteration, sample, frame)
