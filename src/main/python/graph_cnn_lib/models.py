### This code has been adopted from:
### https://github.com/mdeff/cnn_graph
### and modified as per requirements

import collections
import os
import shutil

import numpy as np
import scipy.sparse
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt

import graph


NFEATURES = 28**2
NCLASSES = 10

MIN_PER_DAY = 1440
MIN_INTERVALS = 5
NUM_BINS = MIN_PER_DAY / MIN_INTERVALS

def LSTM(x, weights, biases, n_hidden, n_layers):

    # Define the LSTM cells
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cells = [lstm_cell] * n_layers
    stacked_lstm = rnn.MultiRNNCell(lstm_cells)
    outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32, time_major=False)

    h = tf.transpose(outputs, [1, 0, 2])
    #h = tf.nn.dropout(h, dropout)
    pred = tf.nn.bias_add(tf.matmul(h[-1], weights['out']), biases['out'])
    return pred, h

def temporal_attention_pred(h, weights, biases, n_steps):
    att = []
    for i in range(1,n_steps+1):
        att.append(tf.nn.bias_add(tf.matmul(h[-i], weights['out']), biases['out']))
    return att

def temporal_attention(pred_value_2, pred_value, n_steps):
    cum_temp_att = []
    for i in range(n_steps):
        #print "Time step = " + str(i)
        rmse = np.sqrt(np.mean((pred_value_2[i] - pred_value) ** 2))
        #print "RMSE = " + str(rmse)
        cum_temp_att.append(rmse)
        #plot_comparison(pred_value_2[i], y_value, start_time, end_time)
    temp_att = []
    for i in range(n_steps - 1):
        temp_att.append(0.0 if cum_temp_att[i+1] == cum_temp_att[i] else cum_temp_att[i+1] - cum_temp_att[i])
    sum_temp_att = sum(temp_att)
    temp_att = [x / sum_temp_att if sum_temp_att > 0 else 1.0/len(temp_att) for x in temp_att]
    return temp_att

def spatial_attention(v1, lstm_training_input_batch, n_input, n_hidden, n_steps):
    input_weights = v1[:n_input, :n_hidden]
    transition_weights = v1[n_input:, :n_hidden]
    input_data = lstm_training_input_batch[0]
    spatial_att = np.zeros((n_steps, n_input))
    for i in range(n_steps):
        hidden_layer_values = input_weights.T.dot(input_data[i])
        base_rmse = np.sqrt(np.mean(hidden_layer_values ** 2))
        hidden_layer_values_pred = np.zeros((n_input, n_hidden))
        for j in range(n_input):
            hidden_layer_values_pred[j] = input_weights[j] * input_data[i, j]
            #print "Time step = " + str(i)
            #print "Input index = " + str(j)
            rmse = np.sqrt(np.mean((hidden_layer_values_pred[j] - hidden_layer_values) ** 2))
            #print "RMSE = " + str(rmse)
            spatial_att[i,j] = 0.0 if rmse == base_rmse else (base_rmse - rmse)
            '''plt.plot(hidden_layer_values)
            plt.plot(hidden_layer_values_pred[j])
            plt.show()'''
    sum_spatial_att = spatial_att.sum(axis=1)
    for i in range(n_steps):
        spatial_att[i] = spatial_att[i]/sum_spatial_att[i] if sum_spatial_att[i] > 0 else 1.0/len(spatial_att[i])
    return spatial_att

def plot_temporal_attention(temp_att, start_time, end_time):
    fig = plt.imshow(np.atleast_2d(temp_att), cmap='jet', interpolation='nearest')
    plt.xlabel('Time lag (5 min intervals)')
    plt.title('Temporal Attention, Start Time = ' + str(start_time) + ' mins, End Time = ' + str(end_time) + ' mins',
              y=12)
    fig.axes.get_yaxis().set_visible(False)
    plt.colorbar()
    plt.show()

def plot_spatial_attention(spatial_att, start_time, end_time):
    plt.imshow(spatial_att, cmap='jet', interpolation='nearest')
    plt.xlabel('Variable index')
    plt.ylabel('Time lag (5 min intervals)')
    plt.title('Spatial Attention, Start Time = ' + str(start_time) + ' mins, End Time = ' + str(end_time) + ' mins')
    plt.colorbar()
    plt.show()

# Common methods for all models


class base_model(object):
    def __init__(self):
        self.regularizers = []

    # High-level interface which runs the constructed computational graph.

    def predict(self, data, labels=None, sess=None):
        data = data[0]
        labels = labels[0]
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            batch_data = np.expand_dims(batch_data, axis=0)
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.ph_labels] = np.expand_dims(batch_labels,axis=0)
                batch_pred, batch_hidden_states_value, batch_loss = sess.run([self.op_prediction, self.hidden_states, self.op_loss], feed_dict)
                batch_pred_2 = sess.run(self.temp_att_pred, feed_dict={self.h:batch_hidden_states_value})
                loss += batch_loss
            else:
                batch_pred, batch_hidden_states_value = sess.run([self.op_prediction, self.hidden_states], feed_dict)
                batch_pred_2 = sess.run(self.temp_att_pred, feed_dict={self.h: batch_hidden_states_value})
            predictions[begin:end] = batch_pred[:end - begin]

        if labels is not None:
            return predictions, batch_pred_2, loss * self.batch_size / size
        else:
            return predictions, batch_pred_2

    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.
        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        predictions, predictions_2, loss = self.predict(data, labels, sess)
        string = 'loss: {:.2e}'.format(loss)
        return string, predictions, predictions_2, loss

    def fit(self, train_data, train_labels, min_lag):
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Training.
        losses = []
        indices_input = collections.deque()
        indices_output = collections.deque()
        num_steps = int(train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps + 1):

            print("Iteration " + str(step))

            # Be sure to have used all the samples before using one a second time.
            if len(indices_input) < self.batch_size:
                indices_input.extend(np.arange((step - 1) * self.batch_size, step * self.batch_size))
            if len(indices_output) < self.batch_size:
                indices_output.extend(np.arange((step - 1) * self.batch_size, step * self.batch_size))
            input_idx = [indices_input.popleft() for i in range(self.batch_size)]
            output_idx = [x + min_lag for x in input_idx]

            batch_data, batch_labels = np.expand_dims(train_data[input_idx, :],axis=0), np.expand_dims(train_labels[output_idx],axis=0)
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            string, predictions, predictions_2, loss = self.evaluate(batch_data,
                                                      batch_labels, sess)
            temp_att = temporal_attention(predictions_2, predictions, self.batch_size)

            for v in sess.graph.get_collection('trainable_variables'):
                if ('weights:0' in v.name):
                    trained_weights = v.eval(sess)
                if ('biases:0' in v.name):
                    v2 = v.eval(sess)
            n_steps, n_input = batch_data[0].shape
            spatial_att = spatial_attention(trained_weights, batch_data, n_input, self.lstm_n_hidden, n_steps)
            losses.append(np.sqrt(loss/self.batch_size))

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps or step > self.attention_eval_frequency:
                print "Number of iterations = " + str(step)
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss))
                print('  validation {}'.format(string))

                # Plot predictions
                plt.plot(predictions,label='Prediction')
                plt.plot(batch_labels[0],label='Actual')
                plt.legend()
                plt.show()

                # Plot losses
                plt.plot(losses)
                plt.show()

                # Print average RMSE
                print "RMSE"
                print np.sqrt(loss / self.batch_size)

                if step > self.attention_eval_frequency:
                    # Print and plot temporal attentions
                    start_time = (step * MIN_INTERVALS * self.batch_size + MIN_INTERVALS * min_lag) % MIN_PER_DAY
                    end_time = start_time + MIN_INTERVALS * len(batch_labels[0])
                    print "Temporal attentions:"
                    print temp_att
                    plot_temporal_attention(temp_att, start_time, end_time)

                    print "Spatial attentions:"
                    print spatial_att
                    plot_spatial_attention(spatial_att, start_time, end_time)

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation losses: peak = {:.2f}, mean = {:.2f}'.format(max(losses), np.mean(losses[-10:])))
        writer.close()
        sess.close()

        return losses

    # Methods to construct the computational graph.

    def build_graph(self,M_0, lstm_n_hidden, lstm_n_layers, lstm_n_output):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder("float", [None, self.batch_size, M_0],'data')
                self.ph_labels = tf.placeholder("float", [None, lstm_n_output],'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.h = tf.placeholder("float", [lstm_n_output, None, lstm_n_hidden])

                # Define weights
                self.weights = {
                    'out': tf.Variable(tf.random_normal([lstm_n_hidden, lstm_n_output]))
                }
                self.biases = {
                    'out': tf.Variable(tf.random_normal([lstm_n_output]))
                }

            # Model.
            self.lstm_n_hidden, self.lstm_n_layers, self.lstm_n_output = lstm_n_hidden, lstm_n_layers, lstm_n_output
            self.pred, self.hidden_states = LSTM(self.ph_data, self.weights, self.biases, lstm_n_hidden, lstm_n_layers)
            self.temp_att_pred = temporal_attention_pred(self.h, self.weights, self.biases, self.batch_size)
            self.op_loss = self.loss(self.pred,self.ph_labels)
            self.op_loss_average = self.op_loss
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.pred

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()

    def loss(self,pred,y):
        individual_losses = tf.reduce_sum(tf.squared_difference(pred, y), reduction_indices=1)
        loss = tf.reduce_mean(individual_losses)
        return loss

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess


class cnn_lstm(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.
    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
    L: List of Graph Laplacians. Size M x M. One per coarsening level.
    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.

    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.
    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.
    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, L, F, K, p, M, filter='chebyshev5', brelu='b1relu', pool='mpool1',
                 learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                 regularization=0, dropout=0, batch_size=100, eval_frequency=200, attention_eval_frequency = 200,
                 dir_name='', lstm_n_hidden = 100, lstm_n_layers = 1, lstm_n_output = 288):
        base_model.__init__(self)

        # Verify the consistency w.r.t. the number of layers.
        assert len(L) >= len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.

        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L

        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i + 1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                i, i + 1, L[i].shape[0], F[i], p[i], L[i].shape[0] * F[i] // p[i]))
            F_last = F[i - 1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                i, i + 1, F_last, F[i], K[i], F_last * F[i] * K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i + 1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                    i + 1, L[i].shape[0], F[i], L[i].shape[0] * F[i]))
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc - 1 else 'fc{}'.format(i + 1)
            print('  layer {}: {}'.format(Ngconv + i + 1, name))
            print('    representation: M_{} = {}'.format(Ngconv + i + 1, M[i]))
            M_last = M[i - 1] if i > 0 else M_0 if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                Ngconv + i, Ngconv + i + 1, M_last, M[i], M_last * M[i]))
            print('    biases: M_{} = {}'.format(Ngconv + i + 1, M[i]))

        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.learning_rate = learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency, self.attention_eval_frequency = batch_size, eval_frequency,\
                                                                              attention_eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)

        # Build the computational graph.
        self.build_graph(M_0, lstm_n_hidden, lstm_n_layers, lstm_n_output)

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            # tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x