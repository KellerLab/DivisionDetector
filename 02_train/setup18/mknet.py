import mala
import tensorflow as tf
import json

def create_network(input_shape, name):

    tf.reset_default_graph()

    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1,) + input_shape)

    unet = mala.networks.unet(raw_batched, 12, 5, [[1,2,2],[2,2,2],[2,2,2]])

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

    tf.train.export_meta_graph(filename=name + '.meta')

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
    with open(name + '_config.json', 'w') as f:
        json.dump(names, f)

if __name__ == "__main__":

    create_network((7, 96, 188, 188), 'train_net')
    create_network((7, 88, 292, 292), 'test_net')

