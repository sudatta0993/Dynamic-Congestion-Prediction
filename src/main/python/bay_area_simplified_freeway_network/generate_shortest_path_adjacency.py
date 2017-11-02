import sys
import json
import fiona
import networkx as nx
import numpy as np
import csv

def node_graph(node_file_path):
    node_file = fiona.open(node_file_path)
    G = nx.DiGraph()
    for record in node_file:
        (lng,lat) = record['geometry']['coordinates']
        # Hashing
        id = (int(record['id'])+100)*100
        G.add_node(id,longitude = lng, latitude= lat)
    return G

def add_sp_edges(G, edge_file_path):
    edge_file = fiona.open(edge_file_path)
    d = 0
    for record in edge_file:
        id = int(record['id'])
        properties = record['properties']
        lat_1, lng_1, lat_2, lng_2 = properties['Lat_end_1'], properties['Long_end_1'], \
                                     properties['Lat_end_2'], properties['Long_end_2']
        node_1 = filter(lambda (n,d): d['latitude'] == lat_1 and d['longitude'] == lng_1,G.nodes(data=True))[0][0]
        node_2 = filter(lambda (n,d): d['latitude'] == lat_2 and d['longitude'] == lng_2,G.nodes(data=True))[0][0]
        length_m = float(record['properties']['length_km'])*1000
        G.add_node(id, longitude = (lng_1 + lng_2)/2, latitude = (lat_1+lat_2)/2)
        G.add_weighted_edges_from([(id, node_1, length_m / 2),(node_1, id, length_m/2),
                                   (id, node_2, length_m / 2),(node_2, id, length_m/2)])
        d += 1
    return G, d

def shortest_path_graph(node_file_path, edge_file_path):
    G = node_graph(node_file_path)
    G, d = add_sp_edges(G, edge_file_path)
    return G, d

def shortest_path_adjacency(G,d):
    shortest_path_lengths = nx.all_pairs_dijkstra_path_length(G)
    shortest_path_lengths_dict = {}
    for (id, dict) in shortest_path_lengths:
        shortest_path_lengths_dict[id] = dict
    W = np.zeros((d, d))
    for i in range(d):
        length_dict = shortest_path_lengths_dict[i]
        for j in range(d):
            W[i,j] = length_dict[j]
    return W

def write_csv(W,output_file_path):
    with open(output_file_path,'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(W)):
            writer.writerow(W[i])
    csvfile.close()

def run(node_file_path, edge_file_path, output_file_path):
    G, d = shortest_path_graph(node_file_path, edge_file_path)
    W = shortest_path_adjacency(G,d)
    write_csv(W, output_file_path)

if __name__ == '__main__':
    config_file_path = sys.argv[1]
    with open(config_file_path) as json_file:
        dict = json.load(json_file)
        node_file_path = dict.get('node_file_path')
        edge_file_path = dict.get('edge_file_path')
        output_file_path = dict.get('output_file_path')
        run(node_file_path, edge_file_path, output_file_path)
    json_file.close()