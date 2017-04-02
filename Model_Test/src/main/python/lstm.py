import sys
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

def get_input_features_data(data,feature_cols,batch_size,n_steps,n_input):
    input_features_data = data[feature_cols].copy()
    input_features_data = add_input_features_first_diff(input_features_data)
    input_features_batch = get_input_features_batch(input_features_data, step, batch_size, n_steps, n_input)
    return input_features_batch

def get_LSTM_training_input(data,feature_cols,batch_size,n_steps,n_input):
    input_features_data = get_input_features_data(data,feature_cols, batch_size, n_steps, n_input)
    lstm_input = input_features_data.reshape((batch_size, n_steps, n_input))
    return lstm_input

def get_LSTM_training_output(data, col, step, batch_size, n_steps, n_outputs, min_lag):
    array = np.ndarray.flatten(data[col].as_matrix())
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

    # Parameters
    learning_rate = 0.01
    n_days = 5000
    batch_size = 10
    dropout = 0.75

    # Network Parameters
    n_input = 6
    n_steps = 24
    n_hidden = 100
    n_outputs = 24
    min_lag = 24
    n_layers = 1

    # Plotting parameters
    display_step = 1000
    n_iter_per_day = NUM_BINS / n_steps
    n_plot_loss_iter = 6
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

    # Read input data and define input columns
    data = pd.read_csv(sys.argv[1])
    input_cols =  ['Link 0 demand','Link 1 demand', 'Link 2 demand',
             'Congestion Link from zone 0 input count',
             'Congestion Link from zone 1 input count',
             'Congestion Link from zone 2 input count',
             'Freeway Link 3 input count',
             'Freeway Link 7 input count',
             'Freeway Link 11 input count',
             'Freeway Link 3 output count',
             'Freeway Link 7 output count',
             'Freeway Link 11 output count',
             'Freeway Link 2 input count',
             'Freeway Link 2 output count'
             ]
    output_col = 'Congestion value'

    # Plot demand and output data for first few days
    #plot_data_first_few_days(data,0,10)

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
            lstm_training_input_batch = get_LSTM_training_input(data,input_cols[:n_input/2],batch_size, n_steps, n_input)
            lstm_training_output_batch = get_LSTM_training_output(data, [output_col],step,batch_size,n_steps,n_outputs,min_lag)


            print("Iteration " + str(step))
            print ("Starting optimizer....")

            # Run optimization operation (backprop)
            sess.run(optimizer, feed_dict={x: lstm_training_input_batch, y: lstm_training_output_batch, lr: learning_rate})

            print ("Optimized")

            # Calculate mini-batch loss value
            loss_value = sess.run(loss, feed_dict={x: lstm_training_input_batch, y: lstm_training_output_batch})

            # Store loss value for plotting
            if n_iter_per_day == 1:
                loss_values.append(np.sqrt(loss_value / n_outputs))
            elif step % n_iter_per_day == n_plot_loss_iter:
                loss_values.append(np.sqrt(loss_value/n_outputs))

            if step % display_step == 0:

                # Get prediction and training output
                pred_value, y_value = sess.run((pred, y), feed_dict={x: lstm_training_input_batch, y: lstm_training_output_batch})

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
                print("Average Minibatch Error = " + "{:.6f}".format(np.sqrt(loss_value/n_outputs)))
            step += 1

        print("Optimization Finished!")

        #print weights['out'].eval()
        #print biases['out'].eval()
        for v in tf.trainable_variables():
            print (v.name)
            print (v.eval())