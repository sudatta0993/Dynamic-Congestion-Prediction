import sys
import json
import numpy as np
from sklearn.cluster import KMeans
import csv

def get_W_sp(sp_adjacency_file_path):
    W_sp = np.loadtxt(sp_adjacency_file_path, delimiter=',')
    d = len(W_sp)
    W_sp = np.hstack((W_sp, W_sp))
    W_sp = np.vstack((W_sp, W_sp))
    return W_sp, d

def get_trips(trajectories_file_path, n_trips):
    with open(trajectories_file_path) as trajectories_file:
        trajectories = [line.split(',') for line in trajectories_file]
    return trajectories[:n_trips]

def dtw(x, y, W):
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = W[int(x[i]), int(y[j])] + min(D0[i, j], D0[i, j+1], D0[i+1, j])
    return D1[-1, -1] / sum(D1.shape)

def create_route_adjacency(trajectories,W_sp, n_trips):
    trajectories_adjacency = [dtw(x, y, W_sp) for x in trajectories for y in trajectories]
    return np.reshape(trajectories_adjacency, (n_trips, n_trips))

def cluster_trajectories(trajectories_adjacency, n_clusters):
    k_means = KMeans(n_clusters=n_clusters, precompute_distances=True)
    k_means.fit(trajectories_adjacency)
    cluster_values = k_means.predict(trajectories_adjacency)
    return cluster_values

def calculate_prob_dist(freq_dict, n_clusters):
    tot = sum(freq_dict.values())
    prob = [freq_dict.get(i,0)/float(tot) for i in range(n_clusters)]
    return prob

def calculate_OD_cluster_prob_dist(trajectories,cluster_values,n_trips, d,n_clusters):
    OD_cluster_freq = {}
    for i in range(n_trips):
        cluster_value = cluster_values[i]
        origin_index = int(trajectories[i][0]) % d
        dest_index = int(trajectories[i][-1]) % d
        if (origin_index, dest_index) not in OD_cluster_freq:
            OD_cluster_freq[(origin_index, dest_index)] = {cluster_value: 1}
        else:
            if cluster_value not in OD_cluster_freq[(origin_index, dest_index)]:
                OD_cluster_freq[(origin_index, dest_index)][cluster_value] = 1
            else:
                old_value = OD_cluster_freq[(origin_index, dest_index)][cluster_value]
                OD_cluster_freq[(origin_index, dest_index)][cluster_value] = old_value + 1
    OD_cluster_prob_dist = {k:calculate_prob_dist(v,n_clusters) for (k,v) in OD_cluster_freq.iteritems()}
    return OD_cluster_prob_dist

def calculate_OD_adjacency(OD_cluster_prob_dist, d):
    OD_adjacency = np.full((d*d,d*d),100)
    np.fill_diagonal(OD_adjacency,0)
    for (i,j) in OD_cluster_prob_dist.keys():
        for (k,l) in OD_cluster_prob_dist.keys():
            prob_dist_1 = OD_cluster_prob_dist[(i,j)]
            prob_dist_2 = OD_cluster_prob_dist[(k, l)]
            OD_adjacency[i*d+j,k*d+l] = np.linalg.norm(np.subtract(prob_dist_1,prob_dist_2), 2)
    return OD_adjacency

def write_csv(OD_adjacency,output_file_path):
    with open(output_file_path,'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(OD_adjacency)):
            writer.writerow(OD_adjacency[i])
    csvfile.close()

def run(trajectories_file_path, sp_adjacency_file_path, n_trips, n_clusters, output_file_path):
    W_sp, d = get_W_sp(sp_adjacency_file_path)
    trajectories = get_trips(trajectories_file_path, n_trips)
    print "Creating trajectory adjacency matrix..."
    trajectories_adjacency = create_route_adjacency(trajectories,W_sp,n_trips)
    print "Clustering trajectories..."
    cluster_values = cluster_trajectories(trajectories_adjacency,n_clusters)
    print "Calculating O-D probability distributions..."
    OD_cluster_prob_dist = calculate_OD_cluster_prob_dist(trajectories, cluster_values, n_trips, d,n_clusters)
    print "Creating OD adjacency matrix"
    OD_adjacency = calculate_OD_adjacency(OD_cluster_prob_dist, d)
    print "Writing to CSV file"
    write_csv(OD_adjacency, output_file_path)
    print "CSV file created"


if __name__ == '__main__':
    config_file_path = sys.argv[1]
    with open(config_file_path) as json_file:
        dict = json.load(json_file)
        trajectories_file_path = dict.get('trajectories_file_path')
        sp_adjacency_file_path = dict.get('sp_adjacency_file_path')
        n_trips = dict.get('n_trips')
        n_clusters = dict.get('n_clusters')
        output_file_path = dict.get('output_file_path')
        run(trajectories_file_path,sp_adjacency_file_path,n_trips,n_clusters,output_file_path)