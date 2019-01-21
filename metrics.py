import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def masked_sigmoid_cross_entropy(preds, labels, mask):

    """Sigmoid cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def mask_mse_loss(preds, labels, mask):
    """Sigmoid cross-entropy loss with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)

    labels *= mask 
    preds  *= mask

    # loss = tf.losses.mean_squared_error(labels=labels, predictions=preds, weights=mask)
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=preds, weights=1.0)
    loss = tf.nn.l2_loss(tf.subtract(labels, preds))

    # return tf.reduce_mean(loss)
    return loss


def mask_classification_loss(output_all, output_att, train_mask, attention_mask, labels):
    """Sigmoid cross-entropy loss with masking."""
    label_num = train_mask.shape[0]
    label_num = int(label_num)
    attend_features = tf.nn.embedding_lookup_sparse(output_att, attention_mask, None, combiner='sum')
    classifiers = tf.slice(output_all,[0,0],[label_num,-1])
    if FLAGS.use_normalization:
        # attend_features = tf.nn.l2_normalize(attend_features, dim=-1)
        classifiers = tf.nn.l2_normalize(classifiers, dim=-1)
    raw_score_per_action = tf.multiply(classifiers, attend_features)
    raw_score_per_action = tf.reduce_sum(raw_score_per_action,axis=1)
    labels = tf.one_hot(labels,label_num)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=raw_score_per_action, labels=labels)
    mask = tf.cast(train_mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask

    return tf.reduce_mean(loss), classifiers, attend_features

def mask_classification_accuracy(output_all, output_att, train_mask, attention_mask, labels):
    label_num = train_mask.shape[0]
    label_num = int(label_num)
    attend_features = tf.nn.embedding_lookup_sparse(output_att, attention_mask, None, combiner='sum')
    classifiers = tf.slice(output_all, [0, 0], [label_num, -1])
    if FLAGS.use_normalization:
        # attend_features = tf.nn.l2_normalize(attend_features, dim=-1)
        classifiers = tf.nn.l2_normalize(classifiers, dim=-1)
    raw_score_per_action = tf.multiply(classifiers, attend_features)
    # raw_score_per_action = tf.multiply(tf.slice(output_all, [0, 0], [label_num, -1]), attend_features)
    raw_score_per_action = tf.reduce_sum(raw_score_per_action, axis=1)
    raw_score_per_action = tf.nn.sigmoid(raw_score_per_action)

    mask = tf.cast(train_mask, dtype=tf.float32)
    raw_score_per_action *= mask

    top1 = tf.arg_max(raw_score_per_action, 0)
    flag = tf.equal(top1, tf.cast(labels, dtype=tf.int64))
    acc = tf.cond(flag, fn_true, fn_false)
    return (acc, top1, raw_score_per_action)

def mask_classification_softmax_loss(output_all, output_att, train_mask, attention_mask, labels):
    label_num = train_mask.shape[0]
    label_num = int(label_num)
    classifiers = tf.slice(output_all, [0, 0], [label_num, -1])
    # In the initial code, we find no normalization is better
    if FLAGS.use_normalization:
        # attend_features = tf.nn.l2_normalize(attend_features, dim=-1)
        classifiers = tf.nn.l2_normalize(classifiers, dim=-1)
    output_att = tf.reshape(output_att, [FLAGS.batch_size, -1, FLAGS.output_dim])
    losses = []
    for i in range(FLAGS.batch_size):
        output_att_one = output_att[i, :, :]
        label = labels[i]
        attend_features = tf.nn.embedding_lookup_sparse(output_att_one, attention_mask, None, combiner='sum')
        raw_score_per_action = tf.multiply(classifiers, attend_features)
        raw_score_per_action = tf.reduce_sum(raw_score_per_action,axis=1)
        label = tf.one_hot(label,label_num)
        mask = tf.cast(train_mask, dtype=tf.float32)


        loss = tf.nn.softmax_cross_entropy_with_logits(logits=raw_score_per_action, labels=label)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        losses.append(loss)
    loss = tf.add_n(losses)

    return tf.reduce_mean(loss), classifiers, attend_features

def mask_classification_softmax_accuracy(output_all, output_att, train_mask, attention_mask, labels):
    label_num = train_mask.shape[0]
    label_num = int(label_num)
    classifiers = tf.slice(output_all, [0, 0], [label_num, -1])
    if FLAGS.use_normalization:
        # attend_features = tf.nn.l2_normalize(attend_features, dim=-1)
        classifiers = tf.nn.l2_normalize(classifiers, dim=-1)
    output_att = tf.reshape(output_att, [FLAGS.batch_size, -1, FLAGS.output_dim])
    accs = []
    top1s = []
    scores = []

    for i in range(FLAGS.batch_size):
        output_att_one = output_att[i, :, :]
        label = labels[i]
        attend_features = tf.nn.embedding_lookup_sparse(output_att_one, attention_mask, None, combiner='sum')
        raw_score_per_action = tf.multiply(classifiers, attend_features)
        raw_score_per_action = tf.reduce_sum(raw_score_per_action, axis=1)

        mask = tf.cast(train_mask, dtype=tf.float32)
        # raw_score_per_action = raw_score_per_action[mask]
        raw_score_per_action = tf.nn.softmax(raw_score_per_action)
        raw_score_per_action *= mask

        top1 = tf.arg_max(raw_score_per_action, 0)
        flag = tf.equal(top1, tf.cast(label,dtype=tf.int64))
        acc = tf.cond(flag, fn_true, fn_false)
        accs.append(acc)
        top1s.append(top1)
        scores.append(raw_score_per_action)


    return (accs, top1s, scores)

        # return (acc, top1, raw_score_per_action)

def fn_true():
    return tf.constant(1)

def fn_false():
    return tf.constant(0)



