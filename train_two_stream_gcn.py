from __future__ import division
from __future__ import print_function

import time
import datetime
import os
import tensorflow as tf


from utils import *
from models import GCN_dense_mse_2s, GCN_dense_mse_2s_little


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('dataset', 'ucf101', 'Dataset string.') #ucf101, hmdb51, olympic_sports
flags.DEFINE_string('w2v_type', 'Yahoo_100m', 'Word2Vec Type.')# Google_News_w2v, Yahoo_100m
flags.DEFINE_integer('w2v_dim', 500, 'dimension of the word2vec.')
flags.DEFINE_integer('time_interval', 2, 'Number of time interval for a shot.')#64,4,2
flags.DEFINE_integer('ini_seg_num', 32, 'Number of initial number of segments.')#64,32
flags.DEFINE_integer('num_class', 1588, 'Number of chossen imageNet classes.')# 1588, 2414, 3714, 2271, 3653, 846
flags.DEFINE_integer('output_dim', 512, 'Number of units in the last layer (output the classifier).')# 300, 500
flags.DEFINE_integer('split_ind', 0, 'current zero-shot split.')
flags.DEFINE_integer('topK', 50, 'we choose topK objects for each segment.')# 40, 50, 100, 150, 200
flags.DEFINE_bool('use_normalization', 1, 'use_normalization for the classifiers.')
flags.DEFINE_bool('use_softmax', 1, 'use softmax or sigmoid for the classification.')
flags.DEFINE_bool('use_self_attention', 1, 'use self_attention or not.')
flags.DEFINE_integer('label_num', 101, 'number of actions.')
flags.DEFINE_integer('batch_size', 48, 'batch size.')
flags.DEFINE_string('use_little', 'no_use', 'whether use the little network')# no_use, use_little, use_three_layer
flags.DEFINE_string('result_save_path', './results/', 'results save dir')

 

flags.DEFINE_string('model', 'dense', 'Model string.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')#0.001, 0.0001
flags.DEFINE_string('save_path', './output_models/', 'save dir')
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')# 2048, 1024, 512, 300
flags.DEFINE_integer('hidden2', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('gpu', '0', 'gpu id')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


now_time = datetime.datetime.now().strftime('%Y-%m-%d-%T')

# Load data
data_path = 'data_yahoo_100m_v2'
all_att_inds, all_att_scores = get_imageNet_input_data(FLAGS.dataset, FLAGS.time_interval, FLAGS.ini_seg_num, FLAGS.num_class, root=data_path)#'data_yahoo_100m'
adj_all, adj_att, features, y_train, y_val, idx_train, idx_val, train_mask, test_mask, lookup_table = \
        load_data_action_zero_shot(FLAGS.dataset, FLAGS.w2v_type, FLAGS.split_ind, data_path = data_path)#data_yahoo_100m
label_num = len(train_mask)
FLAGS.label_num = label_num
if FLAGS.w2v_type == 'Yahoo_100m':
    FLAGS.w2v_dim = 500

# Some preprocessing
features, div_mat = preprocess_features_dense2(features)
features_all = features
features_att = features[label_num:,:]

if FLAGS.model == 'dense':
    support_all = [preprocess_adj(adj_all)]
    support_att = [preprocess_adj(adj_att)]
    support_att_batch = [preprocess_adj(adj_att)]
    for s in range(len(support_att_batch)):
        support_att_batch[s] = list(support_att_batch[s])
        for i in range(FLAGS.batch_size-1):
            support_att_batch[s][0] = np.concatenate((support_att_batch[s][0], support_att[s][0]+(i+1)*FLAGS.num_class))
            support_att_batch[s][1] = np.concatenate((support_att_batch[s][1], support_att[s][1]))
        support_att_batch[s][2] = tuple(np.array(support_att[s][2])*FLAGS.batch_size)
    num_supports = len(support_att)
    if FLAGS.use_little == 'use_little':
        model_func = GCN_dense_mse_2s_little
    else:
        model_func = GCN_dense_mse_2s
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

print(features.shape)

# Define placeholders
if FLAGS.use_self_attention:
    seg_number = int(FLAGS.ini_seg_num / FLAGS.time_interval)
    topK = FLAGS.topK
    print('topK = %d' %topK)
    tmp_row_index = np.arange(0, seg_number)
    tmp_row_index = np.expand_dims(tmp_row_index,1)
    tmp_row_index = np.expand_dims(tmp_row_index, 0)
    tmp_row_index = np.tile(tmp_row_index,(FLAGS.batch_size,1,topK))
    tmp_batch_index = np.arange(0, FLAGS.batch_size)
    tmp_batch_index = np.expand_dims(tmp_batch_index, 1)
    tmp_batch_index = np.expand_dims(tmp_batch_index, 1)
    tmp_batch_index = np.tile(tmp_batch_index, (1, seg_number, topK))
    placeholders = {
        'support_all': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support_att': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features_all': tf.placeholder(tf.float32, shape=(features_all.shape[0], features_all.shape[1])),
        'features_att': tf.placeholder(tf.float32, shape=(FLAGS.batch_size, seg_number, 1, features_att.shape[0])),
        'labels': tf.placeholder(tf.int32, shape=(FLAGS.batch_size)),
        'train_mask': tf.placeholder(tf.int32, shape=(train_mask.shape[0])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'learning_rate': tf.placeholder(tf.float32, shape=()),
        'label_num': tf.placeholder(tf.int32, shape=())
    }
else:
    placeholders = {
        'support_all': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support_att': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features_all': tf.placeholder(tf.float32, shape=(features_all.shape[0], features_all.shape[1])),
        'features_att': tf.placeholder(tf.float32, shape=(features_att.shape[0], features_att.shape[1])),
        'labels': tf.placeholder(tf.int32),
        'train_mask': tf.placeholder(tf.int32, shape=(train_mask.shape[0])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'learning_rate': tf.placeholder(tf.float32, shape=()),
        'label_num': tf.placeholder(tf.int32, shape=())
    }



# Create model
lookup_table_act_att = tf.SparseTensor(indices=lookup_table[0], values=lookup_table[1], dense_shape=lookup_table[2])
model = model_func(placeholders, lookup_table_act_att, input_dim=features.shape[1], logging=True)

sess = tf.Session(config=create_config_proto())

# Init variables
sess.run(tf.global_variables_initializer())

savepath = FLAGS.save_path
exp_name = os.path.basename(FLAGS.dataset)
savepath = os.path.join(savepath, exp_name)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('!!! Make directory %s' % savepath)
else:
    print('### save to: %s' % savepath)

result_save_path = FLAGS.result_save_path + FLAGS.dataset + '/'
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
    print('!!! Make directory %s' % result_save_path)
else:
    print('### save to: %s' % result_save_path)
result_file_name = result_save_path + FLAGS.dataset + '_' + FLAGS.w2v_type + '_' \
                    + str(FLAGS.time_interval) + '_' + str(FLAGS.ini_seg_num) \
                   + '_' + str(FLAGS.num_class) + '_' + FLAGS.use_little  \
                   + str(int(FLAGS.learning_rate *100000))+ '_' + str(FLAGS.hidden1) + '_' \
                   + str(FLAGS.output_dim) + '_' + str(FLAGS.use_normalization)+ '_'\
                   + str(FLAGS.use_softmax) + '_' + str(FLAGS.split_ind) + '_' \
                   + str(FLAGS.use_self_attention) + '_' + str(FLAGS.batch_size) + '_' \
                    + str(FLAGS.hidden2) + '.txt'

# Train model
now_lr = FLAGS.learning_rate
y_train = np.array(y_train)
idx_train = np.array(idx_train)
y_val = np.array(y_val)
idx_val = np.array(idx_val)
all_att_inds = np.array(all_att_inds)
all_att_scores = np.array(all_att_scores)
for epoch in range(FLAGS.epochs):
    count = 0
    rand_inds = np.random.permutation(len(y_train))
    rand_inds = rand_inds[:int(len(rand_inds)/FLAGS.batch_size)*FLAGS.batch_size]
    rand_inds = np.reshape(rand_inds,[-1, FLAGS.batch_size])
    for inds in rand_inds[:int(len(rand_inds)/5)]:
        # Construct feed dictionary
        label = y_train[inds]
        video_idx = idx_train[inds]
        if FLAGS.use_self_attention:
            features_att_this_sample = np.zeros([FLAGS.batch_size,seg_number,FLAGS.num_class])
            att_ind = all_att_inds[video_idx]
            att_score = all_att_scores[video_idx]
            att_ind = att_ind[:,:, :topK]
            att_score = att_score[:,:, :topK]
            features_att_this_sample[tmp_batch_index,tmp_row_index, att_ind] = att_score
            features_att_this_sample = np.expand_dims(features_att_this_sample, 2)
        else:
            att_ind = all_att_inds[video_idx]
            att_score = all_att_scores[video_idx]
            att_ind = att_ind[:, :topK]
            att_score = att_score[:, :topK]
            att_activation = get_att_input_activation(att_ind, att_score, FLAGS.num_class, features_att.shape[1])
            features_att_this_sample = np.multiply(features_att, att_activation) # Note here we multiply the att scores and the att features
        feed_dict = construct_feed_dict(features_all, features_att_this_sample, support_all, support_att_batch, label, train_mask, label_num, placeholders)
        feed_dict.update({placeholders['learning_rate']: now_lr})


        outs = sess.run([model.opt_op, model.loss, model.optimizer._lr, model.accuracy, model.classifier, model.attend_feat], feed_dict=feed_dict)

        if count % 1 == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "sample_batch:", '%04d' % (count + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "lr=", "{:.5f}".format(float(outs[2])))
            count += 1
    # model.save(sess=sess, save_path=savepath)
    test_accuracy = 0
    test_inds = np.arange(len(y_val))
    test_inds = test_inds[:int(len(test_inds) / FLAGS.batch_size) * FLAGS.batch_size]
    test_inds = np.reshape(test_inds, [-1, FLAGS.batch_size])
    count_test = 0
    for inds in test_inds:
        # Construct feed dictionary
        label = y_val[inds]
        video_idx = idx_val[inds]
        if FLAGS.use_self_attention:
            features_att_this_sample = np.zeros([FLAGS.batch_size, seg_number, FLAGS.num_class])
            att_ind = all_att_inds[video_idx]
            att_score = all_att_scores[video_idx]
            att_ind = att_ind[:, :, :topK]
            att_score = att_score[:, :, :topK]
            features_att_this_sample[tmp_batch_index, tmp_row_index, att_ind] = att_score
            features_att_this_sample = np.expand_dims(features_att_this_sample, 2)
        else:
            att_ind = all_att_inds[video_idx]
            att_score = all_att_scores[video_idx]
            att_ind = att_ind[:, :topK]
            att_score = att_score[:, :topK]
            att_activation = get_att_input_activation(att_ind, att_score, FLAGS.num_class, features_att.shape[1])
            features_att_this_sample = np.multiply(features_att, att_activation) # Note here we multiply the att scores and the att features
        feed_dict = construct_feed_dict(features_all, features_att_this_sample, support_all, support_att_batch, label,
                                        test_mask, label_num, placeholders)

        # Test step
        out = sess.run(model.accuracy, feed_dict=feed_dict)
        test_accuracy += np.sum(np.array(out[0]))
        count_test += 1
        if count_test % 10 == 0:
            print('%04d baches are processed for testing' % (count_test ))
    test_accuracy /= len(y_val)
    print("Epoch:", '%04d' % (epoch + 1),
          "accuracy=", "{:.5f}".format(float(test_accuracy)),
          )
    with open(result_file_name, 'a') as f:
        f.write(str(test_accuracy)+'\n')


print("Optimization Finished!")

sess.close()
