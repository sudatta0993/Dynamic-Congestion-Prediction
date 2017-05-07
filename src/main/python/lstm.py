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

def plot_data_first_few_days(data, start_day, end_day):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data['Link 0 demand'][NUM_BINS*start_day:NUM_BINS*end_day])
    ax1.plot(data['Link 1 demand'][NUM_BINS*start_day:NUM_BINS*end_day])
    ax1.plot(data['Link 2 demand'][NUM_BINS*start_day:NUM_BINS*end_day])
    ax2.plot(data['Congestion value'][NUM_BINS*start_day:NUM_BINS*end_day],'-y')
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
    return pred

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
    input_features_batch = get_input_features_batch(input_features_data, step, batch_size, n_steps, n_input)
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

        # tf Graph input
        lr = tf.placeholder(tf.float32, [])
        x = tf.placeholder("float", [None, n_steps, n_input])
        y = tf.placeholder("float", [None, n_outputs])
        keep_prob = tf.placeholder(tf.float32)

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_outputs]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_outputs]))
        }

        # Plot demand and output data for first few days
        # plot_data_first_few_days(data,0,10)

        # Define LSTM prediction framework
        pred = LSTM(x, weights, biases, dropout)

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

                if step % display_step == 0:
                    # Get prediction and training output
                    pred_value, y_value = sess.run((pred, y), feed_dict={x: lstm_training_input_batch,
                                                                         y: lstm_training_output_batch})

                    # Display comparison of training prediction and training output
                    print ("Comparing taining prediction to training output...")
                    print ("Training prediction = " + str(pred_value))
                    print ("Training output = " + str(y_value))

                    # Plot training prediction vs actual training output
                    start_time = (step * MIN_INTERVALS * n_steps + MIN_INTERVALS * min_lag) % MIN_PER_DAY
                    end_time = start_time + MIN_INTERVALS * n_outputs
                    plot_comparison(pred_value, y_value, start_time, end_time)

                    # Plot loss vs number of iterations
                    plot_loss_vs_iter(loss_values, n_iter_per_day, n_plot_loss_iter)

                    # Display average mini-batch error
                    print("Average Minibatch Error = " + "{:.6f}".format(np.sqrt(loss_value / n_outputs)))
                step += 1

            print("Optimization Finished!")

            # print weights['out'].eval()
            # print biases['out'].eval()
            for v in tf.trainable_variables():
                print (v.name)
                print (v.eval())

    json_file.close()
