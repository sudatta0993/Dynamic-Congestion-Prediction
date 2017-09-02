# Dynamic-Congestion-Prediction
This code is the repository for implementing algorithms for real-time prediction of macroscopic congestion from network state variables using Deep Learning. The corresponding work is detailed in the following papers (in review):  
* Sudatta Mohanty, Alexey Pozdnukhov, *Real-Time Macroscopic Congestion Prediction Using Deep Learning*, Transportation Research Part c, 2017  
* Sudatta Mohanty, Alexey Pozdnukhov, *GCN-LSTM Framework For Real-Time Macroscopic Congestion Prediction*, Bay Area Machine Learning Symposium (BayLearn), 2017  
  
There is implementation of Graph Convolutional Network (GCN) based on the opensource codebase corresponding to:  
* Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), Neural Information Processing Systems (NIPS), 2016.
  
The Neural Attention Framework used is based on the concepts discussed in:  
* Vasili Ramanishka, Abir Das, Jianming Zhang, and Kate Saenko, [Top-down Visual Saliency Guided by Captions](https://arxiv.org/abs/1612.07360), Conference on Computer Vision and Pattern Recognition (CVPR), 2017  
  
## Generating Test Network And Test Scenario Plots For Single Day 
The base default test network is represented by the following figure:  
![overview_setting](https://github.com/sudatta0993/Dynamic-Congestion-Prediction/blob/master/src/main/python/model_test/overview_setting.png)  
  
To generate plots for test scenarios on test network:    
- Run `/src/test/python/model_test/test_run_scenarios.py`  
- Visualize output files in `/src/test/python/model_test/scenario_<Scenario_Index>`  
    
.json config parameters:  
{  
  `"freeway_links_jam_density"`: List of jam density values (veh/km) for freeway links (between zones),  
  `"check_queue_spillover"`: Update curves after checking queue spillover condition (true/false),  
  `"plot_congestion_io_curves"`: Plot I-0 curves for congestion links (within zones) (true/false),
  `"freeway_links_length"`: List of lengths (km) for freeway links(between zones),  
  `"get_curves_data"`: Output .csv files for data generated for each curve during simulation (true/false),  
  `"plot_demand_congestion_curves"`: Plot curves for demand vs time and congestion vs time (true/false),  
  `"congestion_links_fftt"`: List of free flow travel times (mins) for congestion links (within zones),  
  `"num_bins"`: Number of time bins within a day,  
  `"freeway_links_fftt"`: List of free flow travel times (mins) for freeway links (between zones),  
  `"demand_start_times"`: List of start times (minutes from midnight) for OD demand from each input zone,  
  `"congestion_links_length"`: List of lengths (km) for congestion links (within zones),  
  `"threshold_output_for_congestion"`: List of minimum congestion metric value (mins) below which congestion is treated as zero,  
  `"min_intervals"`: Time interval (mins) for each time bin,  
  `"congestion_links_capacity"`: List of capacity values (veh/hr) for congestion links (within zones),  
  `"demand_slopes"`: List of slopes (veh/hr/min) of OD demand curves (equal positive and negative slopes),  
  `"demand_end_times"`:List of end times (minutes from midnight) for OD demand from each input zone,  
  `"file_directory"`: File path for base directory,  
  `"num_zones"`: Total number of zones,  
  `"plot_route_choice_io_curves"`: Plot IO curves for freeway links to compare route choice (true/false),  
  `"check_route_choice"`: Update curves if users choose route to satisfy Deterministic User Equilibrium (DUE) (true/false),  
  `"congestion_links_jam_density"`: List of jam density values (veh/km) for congestion links (within zones),  
  `"freeway_links_capacity"`: List of capacity values (veh/hr) for freeway links (between zones),  
  `"congestion_nn_smoothening_number"`: Number of nearest neighbors used for smoothening congestion vs time curves  
}
## Reproducing Model Results  
1. 
### Generating Data For Test Network And Test Scenarios  
Run `/src/main/python/model_test/generate_model_data.py` with following parameters:  
- Config .json file path (For example, `./scenario_<Scenario_Index>/config_generate_model_data.json`)  
- Output .csv file path inside base directory defined in config .json file (For example `lstm_input_data/case_<Case_Index>.csv`)
- Variable slope across days - Boolean parameter which specifies whether there is 10% stdev in the slope of demand function across days or not (T/F)  
- Variable start time across days - Boolean parameter which specifies whether there is 30 mins stdev in start time of demand function across days or not (T/F)  
- Realistic demand - Boolean parameter which specifies start times from the three zones at 6AM +/- 30 mins, 7AM +/- 30 mins and 8AM +/ 30 mins respectively with 30 mins stdev in start time and 10% stdev in slope across days, if set `T`, then the previous two parameters are overwritten (T/F)  
- Start day index
- End day index  
  
### Generating Data For Simplified Bay Area Freeway Network  
The simplified Bay Area freeway network is represented by the following figure:  
<img src="https://github.com/sudatta0993/Dynamic-Congestion-Prediction/blob/master/src/main/python/bay_area_simplified_freeway_network/Simplified_Network.png" width="500">  
  
To generate data on simplified Bay Area freeway network for fitting models:    
Contact corresponding author at sudatta.mohanty@berkeley.edu for H-W OD demand files, link count files and arrival count files generated for each scenario and run `/src/main/python/bay_area_simplified_freeway_network/generate_model_data.py` with following parameters:  
- Config .json file path (For example, `config_generate_model_data.json`)    
  
.json config parameters:  
{  
  `"num_zones"`: Total number of zones,  
  `"num_links"`: Total number of (forward + backward) links,  
  `"congestion_zone_nos"`: List of indices of zones for which congestion metric is calculated over time,
  `"congestion_nn_smoothening_number"`: Number of nearest neighbors used for smoothening congestion vs time curves,  
  `"threshold_output_for_congestion"`:  List of minimum congestion metric value (mins) below which congestion is treated as zero,  
  `"start_day"`: Start day index for simulation,  
  `"end_day"`: End day index for simulation,  
  `"num_profiles"`: Number of possible simulation scanerios for each day (currently 10),  
  `"arrival_count_dir_base_path"`: Base file path for directory containing arrival counts .csv files,  
  `"arrival_count_file_name"`: File name (with extension) for file containing arrival counts,  
  `"link_count_dir_base_path"`: Base file path for directory containing link counts .csv files,  
  `"link_count_file_name"`: File name (with extension) for file containing link counts,  
  `"od_count_dir_base_path"`: Base file path for directory containing OD counts .csv files,    
  `"od_count_dir_name"`: Directory name for OD counts .csv files,  
  `"od_count_base_file_name"`: Base file name (to be appended with index) (without extension) for OD counts .csv files,    
  `"output_file_name"`:Output file name (with .csv extension),  
  `"min_intervals"`: Length (mins) of each time bin    
}  
  
2. ### Running 1-NN Model  
Run `/src/main/python/model_test/1NN.py` with following parameters:  
- .csv file path for file generated in the previous step  
  
Output includes:  
- display of plot showing comparison for congestion values for actual data and 1NN prediction 
- display of plot showing RMSE vs iteration number
- average RMSE value

3. ### Running LSTM-only Model 
Run `/src/main/python/lstm.py` with following parameters:  
- Config .json file path (For example, `./model_test/config_lstm.json` or `./bay_area_simplified_freeway_network/config_lstm_zone_0.json`)  
  
.json config parameters:  
{  
  `"input_file_path"`: File path of file generated in Step 1,  
  `"input_data_column_index_ranges"`: List of numbers of even size for input column indices(Each consecutive pair is considered a start and end column for inputs into the model. For example [1,3,5,7] implies that the input columns are 1,2,3,5,6,7),  
  `"output_column_index"`: Column index for target output variable,  
  `"n_days"`: Total number of days of data for running model,  
  `"learning_rate"`: Learning rate for gradient descent during optimization,  
  `"batch_size"`: Batch size (continuous indices in time),  
  `"dropout"`: Dropout rate (currently not implemented),  
  `"n_input"`: Number of inputs (must be equal to number of input columns * 2),  
  `"n_steps"`: Number of time intervals per input data,  
  `"n_hidden"`: Number of hidden nodes per hidden layer,  
  `"n_outputs"`: Number of time intervals per output data,  
  `"min_lag"`: Number of time intervals between between first time bin of input and first time bin of output,  
  `"n_layers"`: Number of hidden layers,  
  `"display_step"`: Display outputs at this iteration interval,  
  `"n_plot_loss_iter"`: If predicting less than a day, the loss is calculated only at the time intervals at `n_steps * [n_plot_loss_iter,n_plot_loss_iter+1]` (this is to ensure we compare apples to apples!),  
  `"attention_display_step"`:Number of iterations after which attention is displayed (for all consecutive iterations until one full day is covered)  
}    
  
Output includes:
- display of plot showing comparison for congestion values for actual data and LSTM prediction every `display_step` (defined in config .json file) iterations 
- display of plot showing RMSE vs iteration number (if prediction done for less than 1 day, then RMSE is calculated for part of day at `n_steps * [n_plot_loss_iter, n_plot_loss_iter + 1]` time bins) every `display_step` iterations
- average RMSE value every `display_step` iterations  
- display of temporal attention model heatmap at `attention_display_step` iterations (the plots are displayed for all subsequent iterations until temporal attention for all times in the day are covered)
- display of spatial attention model heatmap at `attention_display_step` iterations (the plots are displayed for all subsequent iterations until spatial attention for all times in the day are covered)
    
4. ### Running GCN-LSTM Model  
Run `/src/main/python/graph_cnn_lstm.py` with following parameters:  
- Config .json file path (For example, `./model_test/config_graph_cnn_lstm.json` or `./bay_area_simplified_freeway_network/config_graph_cnn_lstm_zone_0.json`)  
  
.json config parameters:  
{  
  `"input_file_path"`: File path of file generated in Step 1,  
  `"input_data_column_index_ranges"`: List of numbers of even size for input column indices(Each consecutive pair is considered a start and end column for inputs into the model. For example [1,3,5,7] implies that the input columns are 1,2,3,5,6,7),  
  `"output_column_index"`: Column index for target output variable,  
  `"n_days"`: Total number of days of data for running model,  
  `"learning_rate"`: Learning rate for gradient descent during optimization,  
  `"decay_rate"`: Decay rate of learning rate per iteration,  
  `"momentum"`: Momentum value for learning rate of previous iteration,  
  `"batch_size"`: Batch size (continuous indices in time),  
  `"eval_frequency"`: Display outputs at this iteration interval,  
  `"regularization"`: L2 regularizations of weights and biases,  
  `"dropout"`: Dropout rate (currently not implemented),  
  `"lstm_n_hidden"`: Number of hidden nodes per hidden LSTM layer,  
  `"lstm_n_outputs"`: Number of time intervals per output data,  
  `"lstm_min_lag"`: Number of time intervals between between first time bin of input and first time bin of output,  
  `"lstm_n_layers"`: Number of hidden LSTM layers,  
  `"display_step"`: Display outputs at this iteration interval,  
  `"n_plot_loss_iter"`: If predicting less than a day, the loss is calculated only at the time intervals at `n_steps * [n_plot_loss_iter,n_plot_loss_iter+1]` (this is to ensure we compare apples to apples!),  
  `"cnn_filter"`: Filter type for GCN (Currently only implemented "chebyshev5"),  
  `"cnn_brelu"`: Bias and Relu for GCN (Currently only implemented "b1relu"),  
  `"cnn_pool"`: Pooling for GCN (Currently only implemented maxpooling "mpool1"),  
  `"cnn_num_conv_filters"`: List of number of convolutional filters for each layer of GCN,  
  `"cnn_poly_order"`: List of polynomial orders (filter sizes) for each layer of GCN,  
  `"cnn_pool_size"`: List of pooling size (1 for no pooling and power of 2 to make graph coarser),  
  `"cnn_output_dim"`: Number of features per sample for GCN,  
  `"attention_display_step"`:Number of iterations after which attention is displayed (for all consecutive iterations)  
}   
  
Output includes:
- display of relative location of points in the original graph  
- display of spectrum of a Laplacians for original and coarsened graphs  
- display of plot showing comparison for congestion values for actual data and LSTM prediction every `display_step` (defined in config .json file) iterations 
- display of plot showing RMSE vs iteration number (if prediction done for less than 1 day, then RMSE is calculated for part of day at `n_steps * [n_plot_loss_iter, n_plot_loss_iter + 1]` time bins) every `display_step` iterations
- average RMSE value every `display_step` iterations 
- display of temporal attention model heatmap at `attention_display_step` iterations (the plots are displayed for all subsequent iterations)
- display of spatial attention model heatmap at `attention_display_step` iterations (the plots are displayed for all subsequent iterations)
