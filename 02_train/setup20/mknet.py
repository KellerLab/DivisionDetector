import tensorflow as tf
import json
import sys
import os
from unet import unet, conv_pass

#Creates structure of train or test network for the setup.

#Input shapes are defined as model-specific constants
train_input_shape = (7, 60, 148, 148)
test_input_shape = (7, 88, 292, 292)

def create_network(input_shape, name):
    output_path = "checkpoints/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tf.reset_default_graph()

    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)

    out = unet(raw_batched, 12, 5, [[1,2,2],[2,2,2],[2,2,2]])

    logits_batched = conv_pass(
        out,
        kernel_size=1,
        num_fmaps=1,
        num_repetitions=1,
        activation=None)

    output_shape_batched = logits_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:] # strip the batch dimension

    logits = tf.reshape(logits_batched, output_shape)
    probs = tf.sigmoid(logits)
    gt_labels = tf.placeholder(tf.float32, shape=output_shape)

    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=gt_labels,
        logits=logits,
        reduction=tf.losses.Reduction.MEAN)

    tf.summary.scalar('loss_total', loss)
    merged = tf.summary.merge_all()

    opt = tf.train.AdamOptimizer(
        learning_rate=1e-5,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename=os.path.join(output_path, name + '.meta'))

    print("input shape : %s"%(input_shape,))
    print("output shape: %s"%(output_shape,))

    names = {
        'raw': raw.name,
        'logits': logits.name,
        'labels': probs.name,
        'gt_labels': gt_labels.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summary': merged.name,
        'input_shape': input_shape,
        'output_shape': output_shape}
    with open(os.path.join(output_path, name + '_config.json'), 'w') as f:
        json.dump(names, f)

if __name__ == "__main__":
    name = sys.argv[1]

    if name == 'train_net':
        create_network(train_input_shape, name)
    elif name == 'test_net':
        create_network(test_input_shape, name)
    else:
        print("Error: name must be 'train_net' or 'test_net'")

