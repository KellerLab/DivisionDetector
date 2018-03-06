import mala
import tensorflow as tf
import json

if __name__ == "__main__":

    input_shape = (74, 324, 324)

    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)

    unet = mala.networks.unet(raw_batched, 12, 5, [[1,5,5],[3,3,3],[2,2,2]])

    labels_batched = mala.networks.conv_pass(
        unet,
        kernel_size=1,
        num_fmaps=1,
        num_repetitions=1,
        activation='sigmoid')

    output_shape_batched = labels_batched.get_shape().as_list()
    output_shape = output_shape_batched[2:] # strip the batch and channel dimension

    labels = tf.reshape(labels_batched, output_shape)
    gt_labels = tf.placeholder(tf.float32, shape=output_shape)
    loss_weights = tf.placeholder(tf.float32, shape=output_shape)

    loss = tf.losses.mean_squared_error(
        gt_labels,
        labels,
        loss_weights)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename='unet.meta')

    names = {
        'raw': raw.name,
        'labels': labels.name,
        'gt_labels': gt_labels.name,
        'loss_weights': loss_weights.name,
        'loss': loss.name,
        'optimizer': optimizer.name}
    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)
