import pandas as pd
import networkx as nx
from community import community_louvain
import os
import json
from datetime import datetime


def community_detector(algorithm_name, network, most_valualble_edge=None):
    """
    A function to detect communities in a network using different algorithms.

    Parameters:
    algorithm_name (str): Name of the algorithm to run. Can be either 'girvin_newman', 'louvain' or 'clique_percolation'.
    network (networkx.Graph): The network to run the detection over.
    most_valuable_edge (function or None): A parameter that is used only by the 'girvin_newman' algorithm, the function
                                            should return an edge from the graph.

    Returns:
    dict: A dictionary with the following key/values:
        - num_partitions (int): Number of partitions the network was divided into.
        - modularity (float): The modularity value of the partition.
        - partition (list of lists): The partition of the network. Each element in the list is a community detected (with node names).
    """
    if algorithm_name == 'girvin_newman':
        communities_generator = nx.community.girvan_newman(network, most_valualble_edge)
        initial_communities = next(communities_generator)
        num_modularity_partition = []

        # Define a default edge selector if none is provided

        # We continue removing edges and getting new community partitions
        try:
            modularity = nx.algorithms.community.quality.modularity(network, initial_communities)
            num_modularity_partition.append((len(initial_communities), modularity, initial_communities))
            # yield all of the possible partitions and store them in the list along with other relevant information
            while True:
                # Get the next community partition
                new_communities = next(communities_generator)
                num_communities = len(new_communities)
                modularity = nx.algorithms.community.quality.modularity(network, new_communities)
                num_modularity_partition.append((num_communities, modularity, new_communities))

                # Optionally, break the loop after some condition or keep it running until no edges are left
                if network.number_of_edges() == 0:
                    break
        except StopIteration:
            pass
        # get the bet partition base on the modularity
        best_values = max(num_modularity_partition, key=lambda x: x[1])
        partitions = []
        # cast the partitions data structure from a set to a list
        for partition in best_values[2]:
            new_partition = list(partition)
            partitions.append(new_partition)
        # return the dictionary
        return {'num_partitions': best_values[0], 'modularity': best_values[1], 'partition': partitions}

    elif algorithm_name == 'louvain':
        partition = community_louvain.best_partition(network)
        communities = []
        for com in set(partition.values()):
            # get the partitions and store them in a list
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            communities.append(list_nodes)
        modularity = nx.algorithms.community.quality.modularity(network, communities)
        # return the dictionary
        return {'num_partitions': len(communities), 'modularity': modularity, 'partition': communities}

    elif algorithm_name == 'clique_percolation':
        def calculate_overlapping_modularity(G, communities, weight=True):
            """
            Calculate the modularity of a graph with overlapping communities.

            Args:
            - G (nx.Graph): The graph for which to calculate modularity.
            - communities (list of lists): A list where each sublist represents a community and contains the nodes in
                                           that community.
            - weight (str or None): The edge weight attribute name.

            Returns:
            - float: The modularity value.
            """
            if weight:
                total_weight = sum(data.get(weight, 1) for u, v, data in G.edges(data=True))
            else:
                total_weight = len(G.edges())

            node_community_count = {node: 0 for node in G}
            for community in communities:
                for node in community:
                    node_community_count[node] += 1

            # Contribution from within-community edges
            modularity = 0
            # loop over the community nodes and add each community's modularity to total modularity in the network
            for community in communities:
                subgraph = G.subgraph(community)
                if weight:
                    L_c = sum(data.get(weight, 1) for u, v, data in subgraph.edges(data=True))
                else:
                    L_c = len(subgraph.edges())

                k_c = sum(G.degree(n, weight=weight) / node_community_count[n] for n in community)
                modularity += (L_c / total_weight) - (k_c / (2 * total_weight)) ** 2

            return modularity
        # find the biggest clique in the graph as we attempt to check every possible clique size
        biggest_k_clique = nx.algorithms.approximation.clique.large_clique_size(network)
        num_modularity_partition = []
        # go on every possible value of k, starts at 3 since 2 regards the entire graph as a community
        for k in range(3, biggest_k_clique + 1):
            try:
                # find communities based on the cliques and calculate the overlapping modularity, store for each k the
                # the respective partitions and modularity
                new_communities = list(list(c) for c in list(nx.community.k_clique_communities(network, k)))
                modularity = calculate_overlapping_modularity(network, new_communities)
                num_communities = len(new_communities)
                num_modularity_partition.append((num_communities, modularity, new_communities))
            except nx.NetworkXError:
                pass
        # if no partitions were found
        if len(num_modularity_partition) == 0:
            return {'num_partitions': 0, 'modularity': None, 'partition': []}
        # get the best partitioning and return the values
        best_values = max(num_modularity_partition, key=lambda x: x[1])
        return {'num_partitions': best_values[0], 'modularity': best_values[1], 'partition': best_values[2]}
    # invalid input for the algorithm name
    else:
        raise ValueError("Unsupported algorithm name")


def edge_selector_optimizer(G):
    """
    A custom edge selector function to be used with the Girvan-Newman algorithm in the community_detector function.
    This function selects the most valuable edge to be removed based on communicability betweenness centrality and
    second order centrality.

    Parameters:
    G (networkx.Graph): The network for which to find the most valuable edge to remove.

    Returns:
    tuple: A tuple representing the edge to be removed (node1, node2).
    """
    # Check if the graph is connected
    if nx.is_connected(G):
        # Calculate communicability betweenness and second order centrality for each node
        com_betweenness = {node: val for node, val in nx.communicability_betweenness_centrality(G).items()}
        second_order = {node: val for node, val in nx.second_order_centrality(G).items()}

        # Sort nodes based on second order centrality and communicability betweenness centrality
        sorted_by_second_order = sorted(second_order, key=second_order.get)
        sorted_by_com_betweenness = sorted(com_betweenness, key=com_betweenness.get, reverse=True)

        # Try to find an edge to remove based on the sorted lists
        for node1 in sorted_by_second_order:
            for node2 in sorted_by_com_betweenness:
                if G.has_edge(node1, node2):
                    return (node1, node2)
    else:
        # Calculate edge betweenness centrality when the graph is not connected
        edge_betweenness = nx.edge_betweenness_centrality(G)
        return max(edge_betweenness, key=edge_betweenness.get)


def construct_heb_edges(files_path, start_date='2019-03-15', end_date='2019-04-15', non_parliamentarians_nodes=0):
    """
    A function to construct an edge dictionary based on retweeting relations from the given txt-json files.

    Parameters:
    files_path (str): Location of all files the function requires. These are the 90 txt files and the csv file.
    start_date (str): First day to include (format: YYYY-MM-DD). Default value is '2019-03-15'.
    end_date (str): Last day to include (format: YYYY-MM-DD). Default value is '2019-04-15'.
    non_parliamentarians_nodes (int): Number of non-parliamentarian nodes to keep in the network on top of the central
                                      political players listed in the CSV file. Default value is 0.

    Returns:
    dict: An edge dictionary where the keys are tuples of user IDs (USER_X_ID, USER_Y_ID) and the values are the
          retweets counter.
    """
    # Load the central political players
    central_players_df = pd.read_csv(os.path.join(files_path, 'central_political_players.csv'))
    central_players = set(central_players_df['id'].astype(str))

    non_parliamentarians = {}
    # Initialize an empty dictionary to store edges
    edges = {}

    # Define the date range
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Process each file within the date range
    for filename in os.listdir(files_path):
        if filename.endswith(".txt") and 'tweets' in filename:
            date = filename.split('_')[1].split('.')[2]  # Assuming files named "tweets_YYYY-MM-DD.txt"
            file_date = datetime.strptime(date, "%Y-%m-%d")
            if start_date <= file_date and  file_date <= end_date:
                with open(os.path.join(files_path, filename), 'r', encoding='utf-8') as file:
                    for line in file:
                        tweet = json.loads(line)
                        user_id = str(tweet['user']['id'])
                        if 'retweeted_status' in tweet:
                            retweeted_user_id = str(tweet['retweeted_status']['user']['id'])
                            # add edge to the dictionary
                            edge = (user_id, retweeted_user_id)
                            if edge in edges:
                                edges[edge] += 1
                            else:
                                edges[edge] = 1
                            # count stats about not central players, retweets from a central players get 10 times the
                            # weight of none central players retweets
                            if retweeted_user_id in central_players and user_id not in central_players:
                                if user_id in non_parliamentarians:
                                    non_parliamentarians[user_id] += 1
                                else:
                                    non_parliamentarians[user_id] = 1
                            if retweeted_user_id not in central_players and user_id not in central_players:
                                if user_id in non_parliamentarians:
                                    non_parliamentarians[user_id] += 0.1
                                else:
                                    non_parliamentarians[user_id] = 0.1
    # don't add non parliamentarians
    if non_parliamentarians_nodes == 0:
        relevant_players = list(central_players)
        # construct the dictionary based on the central players
        final_edges = {edge: weight for edge, weight in edges.items() if set(edge).issubset(relevant_players)}
        return final_edges
    # rank the the non parliamentarians  and pick the top players according to the non_parliamentarians_nodes parameter
    sorted_non_parliamentarians = [k for k, v in sorted(non_parliamentarians.items(), key=lambda item: item[1], reverse=True)]
    if non_parliamentarians_nodes > len(sorted_non_parliamentarians):
        non_parliamentarians_to_keep = sorted_non_parliamentarians
    else:
        non_parliamentarians_to_keep = sorted_non_parliamentarians[:non_parliamentarians_nodes]
    relevant_players = list(central_players) + non_parliamentarians_to_keep
    # construct the dictionary based on the central players and the top ranked non parliamentarians
    final_edges = {edge: weight for edge, weight in edges.items() if set(edge).issubset(relevant_players)}
    return final_edges


def construct_heb_network(edge_dictionary):
    """
    A function to construct a directed and weighted networkX graph from an edge dictionary.

    Parameters:
    edge_dictionary (dict): A dictionary where keys are tuples of user IDs (USER_X_ID, USER_Y_ID) and values are the
                            retweets counter.

    Returns:
    networkx.DiGraph: A directed and weighted networkX graph constructed from the input edge dictionary.
    """
    # Create a directed graph
    G = nx.DiGraph()
    # Add edges to the graph with weights
    for edge, weight in edge_dictionary.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    return G
