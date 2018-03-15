import mala
import tensorflow as tf
import json

if __name__ == "__main__":

    input_shape = (7, 74, 324, 324)

    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1,) + input_shape)

    unet = mala.networks.unet(raw_batched, 12, 5, [[1,5,5],[3,3,3],[2,2,2]])

    logits_batched = mala.networks.conv_pass(
        unet,
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

    tf.train.export_meta_graph(filename='unet.meta')

    names = {
        'raw': raw.name,
        'logits': logits.name,
        'labels': probs.name,
        'gt_labels': gt_labels.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summary': merged.name}
    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)
