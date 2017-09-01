import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graph_cnn_lib import coarsening, graph, models

MIN_PER_DAY = 1440
MIN_INTERVALS = 5
NUM_BINS = MIN_PER_DAY / MIN_INTERVALS

def get_train_val_data(data,input_col_index_ranges,output_column_index,n_days):
    frames_train = []
    for i in range(len(input_col_index_ranges) / 2):
        start_col_index = input_col_index_ranges[2 * i]
        end_col_index = input_col_index_ranges[2 * i + 1] + 1
        cols = data[data.columns[start_col_index:end_col_index]].copy()
        frames_train.append(cols.head(int(n_days*NUM_BINS)))
    X_train = np.delete(pd.concat(frames_train, axis=1).as_matrix(),0,0)
    y_train = np.delete(data[data.columns[output_column_index]].head(int(n_days*NUM_BINS)).as_matrix(),0,0)
    d = len(X_train[0])
    return X_train, y_train, d


def get_cnn_graph_input(X_train, d):
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
    L = [graph.laplacian(A, normalized=True) for A in graphs]
    graph.plot_spectrum(L)
    return L, X_train

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
        attention_eval_frequency = dict.get('attention_display_step')

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

    X_train, y_train, d = get_train_val_data(data, input_data_column_index_ranges,
                                            output_column_index,n_days)
    L, X_train = get_cnn_graph_input(X_train, d)

    model = models.cnn_lstm(L, cnn_num_conv_filters, cnn_poly_order,
                            cnn_pool_size, cnn_output_dim, cnn_filter,
                            cnn_brelu, cnn_pool, learning_rate, decay_rate,
                            decay_steps, momentum, regularization,
                            dropout, batch_size, eval_frequency, attention_eval_frequency,
                            dir_name,lstm_n_hidden, lstm_n_layers, lstm_n_outputs)

    loss = model.fit(X_train, y_train,lstm_min_lag)
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax2 = ax1.twinx()
    ax2.plot(loss, 'g.-')
    ax2.set_ylabel('training loss', color='g')
    plt.show()