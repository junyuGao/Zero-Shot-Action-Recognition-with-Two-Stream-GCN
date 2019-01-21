import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import os

def pklLoad(fname):
    with open(fname, 'rb') as f:
        return pkl.load(f)

def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pkl.dump(obj, f)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sample_mask_sigmoid(idx, h, w):
    """Create mask."""
    mask = np.zeros((h, w))
    matrix_one = np.ones((h, w))
    mask[idx, :] = matrix_one[idx, :]
    return np.array(mask, dtype=np.bool)


def load_data_vis_multi(dataset_str, use_trainval, feat_suffix, label_suffix='ally_multi'):
    """Load data."""
    names = [feat_suffix, label_suffix, 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.NELL.{}".format(dataset_str, names[i]), 'rb') as f:
            print("{}/ind.NELL.{}".format(dataset_str, names[i]))
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    allx, ally, graph = tuple(objects)
    train_test_mask = []
    with open("{}/ind.NELL.index".format(dataset_str), 'rb') as f:
        train_test_mask = pkl.load(f)

    features = allx  # .tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.array(ally)

    idx_test = []
    idx_train = []
    idx_trainval = []

    if use_trainval == True:
        for i in range(len(train_test_mask)):

            if train_test_mask[i] == 0:
                idx_train.append(i)
            if train_test_mask[i] == 1:
                idx_test.append(i)

            if train_test_mask[i] >= 0:
                idx_trainval.append(i)
    else:
        for i in range(len(train_test_mask)):

            if train_test_mask[i] >= 0:
                idx_train.append(i)
            if train_test_mask[i] == 1:
                idx_test.append(i)

            if train_test_mask[i] >= 0:
                idx_trainval.append(i)

    idx_val = idx_test

    train_mask = sample_mask_sigmoid(idx_train, labels.shape[0], labels.shape[1])
    val_mask = sample_mask_sigmoid(idx_val, labels.shape[0], labels.shape[1])
    trainval_mask = sample_mask_sigmoid(idx_trainval, labels.shape[0], labels.shape[1])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_trainval = np.zeros(labels.shape)

    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_trainval[trainval_mask] = labels[trainval_mask]

    return adj, features, y_train, y_val, y_trainval, train_mask, val_mask, trainval_mask


def load_data_action_zero_shot(dataset_str, w2v_type, split_ind, data_path = 'data'):
    """Load data."""
    names = [w2v_type, 'labels', 'graph_all', 'graph_att', 'split_train', 'split_test', 'lookup_table']
    objects = []
    for i in range(len(names)):
        with open(data_path+"/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            print(data_path+"/ind.{}.{}".format(dataset_str, names[i]))
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    allx, ally, graph_all, graph_att, split_train, split_test, lookup_table_act_att = tuple(objects)
    zero_shot_train_classes = split_train[split_ind,:]
    zero_shot_test_classes = split_test[split_ind,:]

    features = allx  # .tolil()
    adj_all = nx.adjacency_matrix(nx.from_dict_of_lists(graph_all))
    adj_att = nx.adjacency_matrix(nx.from_dict_of_lists(graph_att))
    labels = np.array(ally) # Here the label is for each video

    # Here, idx_xxx is for indicating video samples for training, test, and validation,
    # this is a little difference between the original GCN papaer since it conduct node classification.
    # y_xxx is also for each video sample.
    idx_test = []
    idx_train = []
    y_train = []
    y_test = []

    for i in range(len(labels)):
        if labels[i] in zero_shot_train_classes:
            idx_train.append(i)
            y_train.append(labels[i])
        elif labels[i] in zero_shot_test_classes:
            idx_test.append(i)
            y_test.append(labels[i])
    idx_trainval = idx_train
    idx_val = idx_test
    y_trainval = y_train
    y_val = y_test


    # Here, we use the xxx_mask to indicate which nodes (action labels) are used in traing and tesing
    # since this is a zero-shot setting
    train_mask = zero_shot_train_classes
    test_mask = zero_shot_test_classes
    label_num = len(train_mask)+len(test_mask)
    train_mask = sample_mask(train_mask, label_num)
    test_mask = sample_mask(test_mask, label_num)

    return adj_all, adj_att, features, y_train, y_val, idx_train, idx_val, train_mask, test_mask, lookup_table_act_att

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def preprocess_features_dense(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features_dense2(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    div_mat = sp.diags(rowsum)

    return features, div_mat


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features_all, features_att, support_all, support_att, label, train_mask, label_num, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: label})
    feed_dict.update({placeholders['train_mask']: train_mask})
    feed_dict.update({placeholders['features_all']: features_all})
    feed_dict.update({placeholders['features_att']: features_att})
    feed_dict.update({placeholders['support_all'][i]: support_all[i] for i in range(len(support_all))})
    feed_dict.update({placeholders['support_att'][i]: support_att[i] for i in range(len(support_att))})
    feed_dict.update({placeholders['num_features_nonzero']: features_all[1].shape})
    feed_dict.update({placeholders['label_num']: label_num})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def create_config_proto():
    """Reset tf default config proto"""
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 0
    config.gpu_options.force_gpu_compatible = True
    # config.operation_timeout_in_ms=8000
    config.log_device_placement = False
    return config


def get_imageNet_input_data(data_set, time_interval, ini_seg_num, num_class, root='data'):
    """preprocess the imageNet scores for all the data (train and test), merge
       the scores for a fixed time interval. The final segment length of a video
       is ini_seg_length/time_interval"""
    ini_file_name = root + '/imageNet_choosen_class_scores_for_' + data_set.lower() + '.txt'
    save_file_name = root + '/input_data_imageNet_scores_' + data_set.lower() + '.pkl'

    tt = 0 # tt is used for judging whether the pre-saved file is correct for this setting by comparing with the given time_interval
    if not (ini_seg_num % time_interval) == 0:
        print('Error: The time_interval cannot be divided by ini_seg_length', time_interval, ini_seg_num)
        sys.exit()

    # We do not load the previous data for preventing errors
    if os.path.exists(save_file_name):
        saved_data = pklLoad(save_file_name)
        all_inds = saved_data[0]
        all_scores = saved_data[1]
        tt = saved_data[2]
    if not tt == time_interval:
        count_sample =0
        top_K = 0
        with open(ini_file_name) as f:
            all_inds = [] # Save the fianl index of all data
            all_scores = [] # Save the fianl scores of all data
            for line in f:# Here, one line in f denotes one sample
                count_sample += 1
                datas = line.split(',')
                if top_K == 0:
                    top_K = int(len(datas) / 2)
                    top_K = int(top_K / ini_seg_num)  # Calculate the number of topK classes per initial segment
                inds_one_sample = [] # Save the fianl index of all segments in the current data
                scores_one_sample = [] # Save the fianl scores of all segments in the current data
                ind = datas[:int(len(datas) / 2)]
                score = datas[int(len(datas) / 2):]
                for i in range(int(ini_seg_num / time_interval)):
                    final_seg_scores = np.zeros(num_class)
                    start1 = int(i*time_interval*top_K)
                    end1 = start1 + int(time_interval*top_K)
                    ind_tmp = ind[start1:end1]
                    ind_tmp = [int(nn) for nn in ind_tmp]
                    score_tmp = score[start1:end1]
                    score_tmp = [float(nn) for nn in score_tmp] #Note here
                    # score_tmp = [float(nn)/float(nn) for nn in score_tmp]
                    for j in range(time_interval):
                        start2 = int(j*top_K)
                        end2 = start2+top_K
                        ii_tmp = ind_tmp[start2:end2]
                        final_seg_scores[ii_tmp] += score_tmp[start2:end2]
                    final_seg_scores /= time_interval
                    current_seg_inds = np.argsort(-final_seg_scores)
                    current_seg_inds = current_seg_inds[:top_K]
                    current_seg_inds = np.array(current_seg_inds) # Convert list to numpy
                    current_seg_scores =  final_seg_scores[current_seg_inds]
                    current_seg_scores = np.array(current_seg_scores) # Convert list to numpy
                    # current_seg_scores[:] = 1
                    #Note here, currently we do not adopt normalization
                    # current_seg_scores /= np.sum(current_seg_scores) # Normalization
                    inds_one_sample.append(current_seg_inds)
                    scores_one_sample.append(current_seg_scores)
                all_inds.append(inds_one_sample)
                all_scores.append(scores_one_sample)
        pklSave(save_file_name, (all_inds, all_scores, time_interval))
        print(count_sample, 'samples are processed')

    return all_inds, all_scores


def get_att_input_activation(att_ind, att_score, num_att, w2v_dim):
    activation = np.zeros((num_att,1))
    att_score = np.array(att_score)
    activation[att_ind] = att_score.transpose()
    activation = np.tile(activation, w2v_dim)

    return activation
