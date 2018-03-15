import mala
import tensorflow as tf
import json

def focal_loss(labels, logits, gamma=2.0, alpha=0.25):
    '''Get the mean focal loss.'''

    # naive version:
    #
    # losses = -alphas*(1 - p_correct)^gamma*log(p_correct)

    probs = tf.sigmoid(logits)
    alphas =  (1.0 - alpha)*(1.0 - labels) + alpha*labels
    p_correct =  (1.0 - probs)*(1.0 - labels) + probs*labels

    losses = -alpha*(1.0 - p_correct)**gamma*tf.log(tf.maximum(p_correct, 1e-10))

    return tf.reduce_mean(losses)

if __name__ == "__main__":

    input_shape = (7, 96, 188, 188)

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

    loss = focal_loss(
        labels=gt_labels,
        logits=logits,
        alpha=0.25,
        gamma=2.0)

    tf.summary.scalar('loss_total', loss)
    merged = tf.summary.merge_all()

    opt = tf.train.AdamOptimizer(
        learning_rate=1e-5,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename='unet.meta')

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
    with open('net_config.json', 'w') as f:
        json.dump(names, f)
