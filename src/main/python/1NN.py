import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MIN_PER_DAY = 1440
MIN_INTERVALS = 5
NUM_BINS = MIN_PER_DAY / MIN_INTERVALS

def rmse(output, prediction):
    return np.sqrt(np.mean(np.square(output - prediction)))

def plot_comparison(pred_value, y_value, start_time, end_time):
    indices = np.arange(start_time, end_time, MIN_INTERVALS)
    plt.plot(indices, pred_value, label='Prediction')
    plt.plot(indices, y_value, label='Actual')
    plt.xlabel('Time of day (minutes from midnight)')
    plt.ylabel('Congestion value (accumulation/output)(min)')
    plt.legend()
    plt.show()

def plot_loss_vs_iter(step, loss_values):
    plt.xlabel('Iteration #')
    plt.ylabel('Mini-batch loss')
    plt.plot(np.arange(start=0, stop=step+1), loss_values)
    plt.show()

if __name__ == '__main__':

    # Read input data and define input columns
    data = pd.read_csv(sys.argv[1])

    # Parameters
    n_days = 101
    n_outputs = 288
    display_step = 100
    n_iter_per_day = NUM_BINS / n_outputs
    n_plot_loss_iter = 6
    n_plot_loss_iter = min(n_plot_loss_iter, n_iter_per_day)
    training_iters = NUM_BINS * ((n_days - 1) * n_iter_per_day)
    output_col = 'Congestion value'

    # Initialize
    loss_values = []

    # Keep training until reach max iterations
    for i in range(n_days):
        output = np.array(data[output_col][(i+1)*NUM_BINS + (n_plot_loss_iter - 1)*n_outputs:
                      (i+1)*NUM_BINS + n_plot_loss_iter*n_outputs])
        prediction = np.array(data[output_col][i*NUM_BINS + (n_plot_loss_iter - 1)*n_outputs:
                      i*NUM_BINS + n_plot_loss_iter*n_outputs])
        loss_values.append(rmse(output, prediction))
        if i > 0 and i % display_step == 0:
            # Plot training prediction vs actual training output
            start_time = (n_plot_loss_iter - 1)*n_outputs * MIN_INTERVALS
            end_time = start_time + MIN_INTERVALS * n_outputs
            plot_comparison(prediction, output, start_time, end_time)

            # Plot loss vs number of iterations
            plot_loss_vs_iter(i, loss_values)

            # Loss value
            print ("RMSE = "+str(np.mean(loss_values)))