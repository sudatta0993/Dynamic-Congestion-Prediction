import os
import sys
import numpy as np
import json
import csv

MIN_PER_DAY = 1440

def get_arrival_counts(num_zones, arrival_counts_file_path, NUM_BINS):
    lines = [line.rstrip('\n')[:-1] for line in
             open(arrival_counts_file_path)]
    arrival_counts = np.zeros((NUM_BINS, num_zones))
    for i in range(len(lines)):
        arrival_counts[i] = [float(j) for j in lines[i].split(',')]
    arrival_counts = np.transpose(arrival_counts)
    return arrival_counts

def get_link_counts(num_links, link_counts_file_path, NUM_BINS):
    lines = [line.rstrip('\n')[:-1] for line in
             open(link_counts_file_path)]
    link_counts = np.zeros((NUM_BINS, num_links))
    for i in range(len(lines)):
        link_counts[i] = [float(j) for j in lines[i].split(',')]
    link_counts = np.transpose(link_counts)
    return link_counts

def get_od_counts(num_zones, od_counts_dir_path,od_count_base_file_name,
                  NUM_BINS):
    od_counts = np.zeros((NUM_BINS,num_zones, num_zones))
    for i in range(NUM_BINS):
        lines = [line.rstrip('\n')[:-1] for line in
                 open(od_counts_dir_path+os.sep + od_count_base_file_name + str(i) + '.csv')]
        for j in range(len(lines)):
            od_counts[i,j] = [float(k) for k in lines[j].split(',')]
    return od_counts

def get_congestion(zone_no, link_counts, arrival_counts,congestion_nn_smoothening_number,
                   threshold_output_for_congestion, NUM_BINS):
    congestion = np.zeros(NUM_BINS)
    for i in range(NUM_BINS):
        congestion[i] = link_counts[zone_no,i]/arrival_counts[zone_no,i] if \
            arrival_counts[zone_no,i] > threshold_output_for_congestion else 0
    for i in range(len(congestion)):
        congestion[i] = np.average(congestion[max(i - congestion_nn_smoothening_number, 0):
        min(i + congestion_nn_smoothening_number + 1, len(congestion) + 1)])
    return congestion

def get_header(od_counts, link_counts, congestion_zone_nos):
    header_row = ['Time(min)']
    for i in range(len(congestion_zone_nos)):
        for j in range(len(od_counts[0])):
            header_row.append('OD Count from zone ' + str(j) + ' to zone ' + str(congestion_zone_nos[i]))
    for i in range(len(link_counts)):
        header_row.append('Link count for link ' + str(i))
    for i in range(len(congestion_zone_nos)):
        header_row.append('Congestion for zone ' + str(congestion_zone_nos[i]))
    return header_row

def run(num_zones, num_links, congestion_zone_nos ,congestion_nn_smoothening_number,
                   threshold_output_for_congestion, num_profiles,arrival_count_dir_base_path,
        arrival_count_file_name,link_count_dir_base_path, link_count_file_name,
        od_count_dir_base_path, od_count_dir_name, od_count_base_file_name, NUM_BINS):

    profile_no = np.random.randint(1, num_profiles + 1)
    arrival_counts_file_path = arrival_count_dir_base_path + str(profile_no) + os.sep + arrival_count_file_name
    link_counts_file_path = link_count_dir_base_path + str(profile_no) + os.sep + link_count_file_name
    od_counts_dir_path = od_count_dir_base_path + str(profile_no) + os.sep + od_count_dir_name
    arrival_counts = get_arrival_counts(num_zones, arrival_counts_file_path, NUM_BINS)
    link_counts = get_link_counts(num_links, link_counts_file_path, NUM_BINS)
    od_counts = get_od_counts(num_zones, od_counts_dir_path, od_count_base_file_name, NUM_BINS)
    congestion = np.zeros((len(congestion_zone_nos), NUM_BINS))
    for i in range(len(congestion_zone_nos)):
        congestion[i] = get_congestion(congestion_zone_nos[i], link_counts, arrival_counts,
                                       congestion_nn_smoothening_number, threshold_output_for_congestion,
                                       NUM_BINS)
    return od_counts, link_counts, congestion

def write_csv(num_zones, num_links, congestion_zone_nos ,congestion_nn_smoothening_number,
                   threshold_output_for_congestion, start_day, end_day ,num_profiles,arrival_count_dir_base_path,
        arrival_count_file_name,link_count_dir_base_path, link_count_file_name,
        od_count_dir_base_path, od_count_dir_name, od_count_base_file_name, output_file_name,
              min_intervals):
    MIN_INTERVALS = min_intervals
    NUM_BINS = MIN_PER_DAY / MIN_INTERVALS
    od_counts, link_counts, congestion = run(num_zones, num_links, congestion_zone_nos ,
                    congestion_nn_smoothening_number,threshold_output_for_congestion,
                    num_profiles,arrival_count_dir_base_path, arrival_count_file_name,
                    link_count_dir_base_path, link_count_file_name,od_count_dir_base_path,
                    od_count_dir_name, od_count_base_file_name, NUM_BINS)
    with open(output_file_name, 'a+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if start_day == 0:
            header_row = get_header(od_counts, link_counts, congestion_zone_nos)
            writer.writerow(header_row)
        for i in range(start_day,end_day+1):
            od_counts, link_counts, congestion = run(num_zones, num_links, congestion_zone_nos ,
                    congestion_nn_smoothening_number,threshold_output_for_congestion,
                    num_profiles,arrival_count_dir_base_path, arrival_count_file_name,
                    link_count_dir_base_path, link_count_file_name,od_count_dir_base_path,
                    od_count_dir_name, od_count_base_file_name, NUM_BINS)
            for j in range(0, MIN_PER_DAY, MIN_INTERVALS):
                row_data = [i * MIN_PER_DAY + j]
                for k in range(len(congestion_zone_nos)):
                    row_data.extend(od_counts[j/MIN_INTERVALS,:,congestion_zone_nos[k]])
                row_data.extend(link_counts[:,j/MIN_INTERVALS])
                for k in range(len(congestion_zone_nos)):
                    row_data.append(congestion[k,j/MIN_INTERVALS])
                writer.writerow(row_data)
        csvfile.close()


if __name__ == '__main__':
    config_file_path = sys.argv[1]
    with open(config_file_path) as json_file:
        dict = json.load(json_file)
        num_zones = dict.get('num_zones')
        num_links = dict.get('num_links')
        congestion_zone_nos = dict.get('congestion_zone_nos')
        congestion_nn_smoothening_number = dict.get('congestion_nn_smoothening_number')
        threshold_output_for_congestion = dict.get('threshold_output_for_congestion')
        start_day = dict.get('start_day')
        end_day = dict.get('end_day')
        num_profiles = dict.get('num_profiles')
        arrival_count_dir_base_path = dict.get('arrival_count_dir_base_path')
        arrival_count_file_name = dict.get('arrival_count_file_name')
        link_count_dir_base_path = dict.get('link_count_dir_base_path')
        link_count_file_name = dict.get('link_count_file_name')
        od_count_dir_base_path = dict.get('od_count_dir_base_path')
        od_count_dir_name = dict.get('od_count_dir_name')
        od_count_base_file_name = dict.get('od_count_base_file_name')
        output_file_name = dict.get('output_file_name')
        min_intervals = dict.get('min_intervals')
        write_csv(num_zones, num_links, congestion_zone_nos, congestion_nn_smoothening_number,
                  threshold_output_for_congestion, start_day, end_day, num_profiles, arrival_count_dir_base_path,
                  arrival_count_file_name, link_count_dir_base_path, link_count_file_name,
                  od_count_dir_base_path, od_count_dir_name, od_count_base_file_name, output_file_name,
                  min_intervals)
    json_file.close()