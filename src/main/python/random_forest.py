import json
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

def rmse(output, prediction):
    return np.sqrt(np.mean(np.square(output - prediction)))

def get_input_features_data(data,col_index_ranges,n_input):
    frames = []
    for i in range(int(len(col_index_ranges)/2)):
        start_col_index = col_index_ranges[2*i]
        end_col_index = col_index_ranges[2*i+1] + 1
        cols = data[data.columns[start_col_index:end_col_index]].copy()
        frames.append(cols)
    frames = pd.concat(frames,axis=1)
    return get_input_data(frames.values, n_input)

def get_input_data(input_features_data,n_input):
    input = np.zeros((int(len(input_features_data)/n_input),int(n_input*len(input_features_data[0]))))
    for i in range(int(len(input_features_data) / n_input)):
        input[i] = np.array(input_features_data[i*n_input:i*n_input+n_input]).flatten()
    return input

def get_output_data(data,col_index,n_input):
    output = np.zeros((int(len(data)/n_input),int(n_input)))
    for i in range(int(len(data)/n_input)):
        output[i] = data[data.columns[col_index]][i*n_input:i*n_input+n_input]
    return output

if __name__ == '__main__':

    config_file_path = sys.argv[1]
    with open(config_file_path) as json_file:
        dict = json.load(json_file)

        # Define input columns
        input_file_path = dict.get('input_file_path')
        data = pd.read_csv(input_file_path)
        input_data_column_index_ranges = dict.get('input_data_column_index_ranges')
        output_column_index = dict.get('output_column_index')

        # Parameters
        n_input = 288
        n_training_samples = 300

        # Training data
        X = get_input_features_data(data, input_data_column_index_ranges,n_input)
        y = get_output_data(data, output_column_index,n_input)

        # Fit model
        clf = RandomForestRegressor()
        clf.fit(X[:n_training_samples-1],y[1:n_training_samples])

        # Calculate RMSE on test sample
        prediction = clf.predict(X[n_training_samples:n_training_samples+1]).flatten()
        output = y[n_training_samples]
        plt.plot(prediction)
        plt.plot(output)
        plt.show()
        print("RMSE = " + str(rmse(output, prediction)))

