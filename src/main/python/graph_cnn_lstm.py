import json
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn

from graph_cnn_lib import coarsening, graph, models

MIN_PER_DAY = 1440
MIN_INTERVALS = 5
NUM_BINS = MIN_PER_DAY / MIN_INTERVALS

def get_train_val_data(data,input_col_index_ranges,output_column_index,n_days,train_perc,batch_size):
    frames_train = []
    frames_val = []
    for i in range(len(input_col_index_ranges) / 2):
        start_col_index = input_col_index_ranges[2 * i]
        end_col_index = input_col_index_ranges[2 * i + 1] + 1
        cols = data[data.columns[start_col_index:end_col_index]].copy()
        frames_train.append(cols.head(int(n_days*NUM_BINS*train_perc)))
        n_eval = math.ceil(n_days * NUM_BINS * (1 - train_perc)/batch_size)*batch_size
        frames_val.append(cols.tail(int(n_eval)))
    X_train = np.delete(pd.concat(frames_train, axis=1).as_matrix(),0,0)
    y_train = np.delete(data[data.columns[output_column_index]].head(int(n_days*NUM_BINS*train_perc)).as_matrix(),0,0)
    X_val = pd.concat(frames_val, axis=1).as_matrix()
    y_val = data[data.columns[output_column_index]].tail(int(n_eval)).as_matrix()
    d = len(X_train[0])
    return X_train, y_train, X_val, y_val, d


def get_cnn_graph_input(X_train, X_val, d):
    dist, idx = graph.distance_scipy_spatial(X_train.T, k=2, metric='euclidean')
    A = graph.adjacency(dist, idx).astype(np.float32)
    assert A.shape == (d, d)
    print('d = |V| = {}, k|V| < |E| = {}'.format(d, A.nnz))
    plt.spy(A, markersize=2, color='black')
    plt.show()
    graphs, perm = coarsening.coarsen(A, levels=2, self_connections=False)
    print "X train initial size"
    print "n = " + str(len(X_train)) + ", d = " + str(len(X_train[0]))
    X_train = coarsening.perm_data(X_train, perm)
    print "X train size after graph coarsing"
    print "n = " + str(len(X_train)) + ", d = " + str(len(X_train[0]))
    X_val = coarsening.perm_data(X_val, perm)
    L = [graph.laplacian(A, normalized=True) for A in graphs]
    graph.plot_spectrum(L)
    return L, X_train, X_val

def LSTM(x, weights, biases, n_hidden, n_layers):

    # Define the LSTM cells
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cells = [lstm_cell] * n_layers
    stacked_lstm = rnn.MultiRNNCell(lstm_cells)
    outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32, time_major=False)

    h = tf.transpose(outputs, [1, 0, 2])
    #h = tf.nn.dropout(h, dropout)
    pred = tf.nn.bias_add(tf.matmul(h[-1], weights['out']), biases['out'])
    return pred

if __name__ == '__main__':

    # Input config file
    config_file_path = sys.argv[1]

    with open(config_file_path) as json_file:
        dict = json.load(json_file)

        # Data
        input_file_path = dict.get('input_file_path')
        data = pd.read_csv(input_file_path)
        input_data_column_index_ranges = dict.get('input_data_column_index_ranges')
        output_column_index = dict.get('output_column_index')
        n_days = dict.get('n_days')
        train_perc = dict.get('train_perc')

        # Overall parameters
        dir_name = 'demo'
        batch_size = dict.get('batch_size')
        eval_frequency = dict.get('eval_frequency')
        regularization = dict.get('regularization')
        dropout = dict.get('dropout')
        learning_rate = dict.get('learning_rate')
        decay_rate = dict.get('decay_rate')
        momentum = dict.get('momentum')
        decay_steps = n_days / batch_size

        # LSTM Network Parameters
        lstm_n_hidden = dict.get('lstm_n_hidden')
        lstm_n_outputs = dict.get('lstm_n_outputs')
        lstm_min_lag = dict.get('lstm_min_lag')
        lstm_n_layers = dict.get('lstm_n_layers')

        # CNN Graph Network Parameters
        cnn_filter = dict.get('cnn_filter')
        cnn_brelu = dict.get('cnn_brelu')
        cnn_pool = dict.get('cnn_pool')
        cnn_num_conv_filters = dict.get('cnn_num_conv_filters')
        cnn_poly_order = dict.get('cnn_poly_order')
        cnn_pool_size = dict.get('cnn_pool_size')
        cnn_output_dim = dict.get('cnn_output_dim')

    X_train, y_train, X_val, y_val, d = get_train_val_data(data,
                                                           input_data_column_index_ranges,
                                                           output_column_index,n_days,train_perc,
                                                           batch_size)
    L, X_train, X_val = get_cnn_graph_input(X_train, X_val, d)

    model = models.cnn_lstm(L, cnn_num_conv_filters, cnn_poly_order,
                            cnn_pool_size, cnn_output_dim, cnn_filter,
                            cnn_brelu, cnn_pool, learning_rate, decay_rate,
                            decay_steps, momentum, regularization,
                            dropout, batch_size, eval_frequency,
                            dir_name,lstm_n_hidden, lstm_n_layers, lstm_n_outputs)

    loss = model.fit(X_train, y_train, X_val, y_val,lstm_min_lag)
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax2 = ax1.twinx()
    ax2.plot(loss, 'g.-')
    ax2.set_ylabel('training loss', color='g')
    plt.show()