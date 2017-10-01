import sys
import json
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MIN_PER_DAY = 1440
MIN_INTERVALS = 5
NUM_BINS = MIN_PER_DAY / MIN_INTERVALS
NOISE_STDEV = 0

def plot_data_first_few_days(data, input_data_column_index_ranges, output_column_index, start_day, end_day):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    #ax1.plot(data[data.columns[1]][NUM_BINS*start_day:NUM_BINS*end_day])
    #ax1.plot(data[data.columns[2]][NUM_BINS*start_day:NUM_BINS*end_day])
    #ax1.plot(data[data.columns[3]][NUM_BINS*start_day:NUM_BINS*end_day])
    ax2.plot(data[data.columns[output_column_index]][NUM_BINS*start_day:NUM_BINS*end_day],'-y')
    plt.show()

def define_loss_and_optimizer(pred, y):
    individual_losses = tf.reduce_sum(tf.squared_difference(pred, y), reduction_indices=1)
    loss = tf.reduce_mean(individual_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    return loss, optimizer

def LSTM(x, weights, biases, dropout):

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
        print "Time step = " + str(i)
        rmse = np.sqrt(np.mean((pred_value_2[i] - pred_value) ** 2))
        print "RMSE = " + str(rmse)
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
            print "Time step = " + str(i)
            print "Input index = " + str(j)
            rmse = np.sqrt(np.mean((hidden_layer_values_pred[j] - hidden_layer_values) ** 2))
            print "RMSE = " + str(rmse)
            spatial_att[i,j] = 0.0 if rmse == base_rmse else (base_rmse - rmse)
            '''plt.plot(hidden_layer_values)
            plt.plot(hidden_layer_values_pred[j])
            plt.show()'''
    sum_spatial_att = spatial_att.sum(axis=1)
    for i in range(n_steps):
        spatial_att[i] = spatial_att[i]/sum_spatial_att[i] if sum_spatial_att[i] > 0 else 1.0/len(spatial_att[i])
    return spatial_att

def get_input_features_batch(matrix, step, batch_size, n_steps, n_input):
    input_batch_data = np.zeros((batch_size, n_steps, n_input))
    for i in range(batch_size):
        input_batch_data[i] = matrix[(step - 1) * n_steps + i:step * n_steps + i,:]
    return input_batch_data

def add_input_features_first_diff(input_features_data):
    input_features_data_first_diff = input_features_data.diff().fillna(value=0).as_matrix()
    input_features_data = np.append(input_features_data, input_features_data_first_diff, axis=1)
    return input_features_data

def get_input_features_data(data,col_index_ranges,batch_size,n_steps,n_input):
    frames = []
    for i in range(len(col_index_ranges)/2):
        start_col_index = col_index_ranges[2*i]
        end_col_index = col_index_ranges[2*i+1] + 1
        cols = data[data.columns[start_col_index:end_col_index]].copy()
        frames.append(cols)
    input_features_data = pd.concat(frames,axis=1)
    input_features_data = add_input_features_first_diff(input_features_data)
    input_features_mean = input_features_data.mean(axis=0)
    input_features_noise = np.random.randn()*NOISE_STDEV*input_features_mean
    input_features_batch = get_input_features_batch(input_features_data, step, batch_size, n_steps, n_input)
    input_features_batch += input_features_noise
    return input_features_batch

def get_LSTM_training_input(data,feature_cols,batch_size,n_steps,n_input):
    input_features_data = get_input_features_data(data,feature_cols, batch_size, n_steps, n_input)
    lstm_input = input_features_data.reshape((batch_size, n_steps, n_input))
    return lstm_input

def get_LSTM_training_output(data, col_index, step, batch_size, n_steps, n_outputs, min_lag):
    array = np.ndarray.flatten(data.ix[:,col_index].as_matrix())
    output_data = np.zeros((batch_size, n_outputs))
    for i in range(batch_size):
        output_data[i] = array[(step - 1) * n_steps + i + min_lag: (step - 1) * n_steps + i + min_lag + n_outputs]
    output_data = output_data.reshape((batch_size, n_outputs))
    return output_data

def plot_comparison(pred_value, y_value, start_time, end_time):
    indices = np.arange(start_time, end_time, MIN_INTERVALS)
    plt.plot(indices, pred_value[0], label='Prediction')
    plt.plot(indices, y_value[0], label='Actual')
    plt.xlabel('Time of day (minutes from midnight)')
    plt.ylabel('Congestion value (accumulation/output)(min)')
    plt.legend()
    plt.show()

def plot_loss_vs_iter(loss_values, n_iter_per_day, plot_loss_iter_num):
    plt.xlabel('Iteration #')
    plt.ylabel('Mini-batch loss')
    plt.plot(np.arange(start=0, stop=(step + (n_iter_per_day - plot_loss_iter_num)) / n_iter_per_day), loss_values)
    plt.show()

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
    plt.title('Spatial Attention, Start Time = ' + str(start_time) + ' mins, End Time = ' + str(end_time) + ' mins',
              x=-1)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':

    #Input config file
    config_file_path = sys.argv[1]
    with open(config_file_path) as json_file:
        dict = json.load(json_file)

        # Data
        input_file_path = dict.get('input_file_path')
        data = pd.read_csv(input_file_path)
        input_data_column_index_ranges = dict.get('input_data_column_index_ranges')
        output_column_index = dict.get('output_column_index')

        # Overall Parameters
        learning_rate = dict.get('learning_rate')
        n_days = dict.get('n_days')
        batch_size = dict.get('batch_size')
        dropout = dict.get('dropout')

        # Network Parameters
        n_input = dict.get('n_input')
        n_steps = dict.get('n_steps')
        n_hidden = dict.get('n_hidden')
        n_outputs = dict.get('n_outputs')
        min_lag = dict.get('min_lag')
        n_layers = dict.get('n_layers')

        # Plotting parameters
        display_step = dict.get('display_step')
        n_iter_per_day = NUM_BINS / n_steps
        n_plot_loss_iter = dict.get('n_plot_loss_iter')
        n_plot_loss_iter = min(n_plot_loss_iter, n_iter_per_day)
        training_iters = NUM_BINS * (n_days * n_iter_per_day - 2)
        attention_display_step = dict.get('attention_display_step')

        # tf Graph input
        lr = tf.placeholder(tf.float32, [])
        x = tf.placeholder("float", [None, n_steps, n_input])
        y = tf.placeholder("float", [None, n_outputs])
        keep_prob = tf.placeholder(tf.float32)
        h = tf.placeholder("float",[n_outputs, None, n_hidden])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_outputs]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_outputs]))
        }

        # Plot demand and output data for first few days
        #plot_data_first_few_days(data,input_data_column_index_ranges, output_column_index, 0,10)

        # Define LSTM prediction framework
        pred, states = LSTM(x, weights, biases, dropout)
        temp_att_pred = temporal_attention_pred(h, weights, biases, n_steps)

        # Define loss (Euclidean distance) and optimizer
        loss, optimizer = define_loss_and_optimizer(pred, y)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            loss_values = []

            # Keep training until reach max iterations
            while step * n_steps < training_iters:
                lstm_training_input_batch = get_LSTM_training_input(data, input_data_column_index_ranges, batch_size,
                                                                    n_steps, n_input)
                lstm_training_output_batch = get_LSTM_training_output(data, output_column_index, step, batch_size,
                                                                      n_steps, n_outputs, min_lag)

                print("Iteration " + str(step))
                print ("Starting optimizer....")

                # Run optimization operation (backprop)
                sess.run(optimizer,
                         feed_dict={x: lstm_training_input_batch, y: lstm_training_output_batch, lr: learning_rate})

                print ("Optimized")

                # Calculate mini-batch loss value
                loss_value = sess.run(loss, feed_dict={x: lstm_training_input_batch, y: lstm_training_output_batch})

                # Store loss value for plotting
                if n_iter_per_day == 1:
                    loss_values.append(np.sqrt(loss_value / n_outputs))
                elif step % n_iter_per_day == n_plot_loss_iter:
                    loss_values.append(np.sqrt(loss_value / n_outputs))

                if step % display_step == 0 or (step > attention_display_step and step <= attention_display_step + n_iter_per_day):
                    # Get prediction and training output
                    pred_value, hidden_states_value, y_value = sess.run((pred, states, y), feed_dict={x: lstm_training_input_batch,
                                                                                                      y: lstm_training_output_batch})
                    pred_value_2 = sess.run(temp_att_pred, feed_dict={h:hidden_states_value})
                    # Display comparison of training prediction and training output
                    print ("Comparing taining prediction to training output...")
                    print ("Training prediction = " + str(pred_value))
                    print ("Training output = " + str(y_value))

                    print "Recovering trained variables"
                    for v in tf.trainable_variables():
                        if ('weights:0' in v.name):
                            v1 = v.eval()
                            break

                    # Plot training prediction vs actual training output
                    start_time = (step * MIN_INTERVALS * n_steps + MIN_INTERVALS * min_lag) % MIN_PER_DAY
                    end_time = start_time + MIN_INTERVALS * n_outputs
                    plot_comparison(pred_value, y_value, start_time, end_time)

                    # Plot loss vs number of iterations
                    plot_loss_vs_iter(loss_values, n_iter_per_day, n_plot_loss_iter)

                    # Display average mini-batch error
                    print("Average Minibatch Error = " + "{:.6f}".format(np.sqrt(loss_value / n_outputs)))

                    if step > attention_display_step  and step <= attention_display_step + n_iter_per_day:
                        temp_att = temporal_attention(pred_value_2,pred_value, n_steps)
                        print "Temporal attentions:"
                        print temp_att
                        plot_temporal_attention(temp_att, start_time, end_time)

                        spatial_att = spatial_attention(v1,lstm_training_input_batch, n_input, n_hidden, n_steps)
                        print "Spatial attentions:"
                        print spatial_att
                        plot_spatial_attention(spatial_att, start_time, end_time)

                step += 1

            print("Optimization Finished!")

    json_file.close()
