import networkx as nx


def centrality_measures(network, node, iterations=100):
    """
    Calculate basic centrality measures for a given network and a specific node.

    Parameters:
    - network: networkX object, the network to run the analysis over
    - node: int, the node name (represented as an integer) to retrieve the centrality measures for
    - iterations: int, to be used by the page-rank and the authority score algorithms

    Returns:
    - A dictionary with the following key/values:
      - 'dc': Degree centrality of the given node
      - 'cs': Closeness centrality of the given node
      - 'nbc': Normalized betweenness centrality of the given node
      - 'pr': Normalized page-rank score of the given node
      - 'auth': Normalized authority score of the given node
    """
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(network)[node]

    # Calculate closeness centrality
    closeness_centrality = nx.closeness_centrality(network)[node]

    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(network, normalized=True)[node]

    # Calculate page-rank score
    page_rank = nx.pagerank(network, alpha=0.85, max_iter=iterations)[node]

    # Calculate authority score using HITS algorithm
    _, authority_scores = nx.hits(network, max_iter=iterations, normalized=True)
    authority_score = authority_scores[node]

    # Return the results in a dictionary
    return {
        'dc': degree_centrality,
        'cs': closeness_centrality,
        'nbc': betweenness_centrality,
        'pr': page_rank,
        'auth': authority_score
    }


def single_step_voucher(network):
    """
    Find the best node to send the voucher to such that it can reach the maximum number of nodes in one step.

    Parameters:
    - network: networkX object, the network to run the analysis over

    Returns:
    - An integer representing the best node to send the voucher to
    """
    # Find the node with the highest degree (most immediate neighbors)
    degree_centrality = nx.degree_centrality(network)

    # Find the node with the highest degree centrality
    best_node = max(degree_centrality, key=degree_centrality.get)

    return best_node


def multiple_steps_voucher(network):
    """
    Find the best node to send the voucher to such that it reaches all nodes while minimizing the total number of steps.

    Parameters:
    - network: networkX object, the network to run the analysis over

    Returns:
    - An integer representing the best node to send the voucher to
    """
    # Closeness centrality measures how close a node is to all other nodes in the network
    closeness_centrality = nx.closeness_centrality(network)

    # The node with the highest closeness centrality is the one that minimizes the total number of steps required to
    # reach all other nodes in the network
    lowest_num_of_steps_node = max(closeness_centrality, key=closeness_centrality.get)

    return lowest_num_of_steps_node


def multiple_steps_diminished_voucher(network):
    """
    Find the best node to send the voucher to such that it maximizes the total benefit to the network,
    considering the voucher's value diminishes by 6% in every step and nullifies after 4 steps.

    Parameters:
    - network: networkX object, the network to run the analysis over

    Returns:
    - An integer representing the best node to send the voucher to
    """
    max_benefit = 0
    best_node = None
    value_decay = 0.94  # 6% diminution per step

    # Iterate through each node in the network to evaluate the total benefit if the voucher starts from that node
    for node in network.nodes:
        total_benefit = 0
        visited = {node}
        current_level = [node]
        current_value = 1  # initial voucher value
        step = 0

        # Perform BFS up to 4 steps
        while current_level and step < 4:
            next_level = []
            for current_node in current_level:
                for neighbor in network.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                        total_benefit += current_value * value_decay ** step
            current_level = next_level
            step += 1

        # Update the maximum benefit and best node if the current node provides a higher benefit
        if total_benefit > max_benefit:
            max_benefit = total_benefit
            best_node = node

    return best_node


def find_most_valuable(network):
    """
    Identify the single node that is most valuable to the marketing strategy, i.e., the node with the highest betweenness centrality.

    Parameters:
    - network: networkX object, the network to run the analysis over

    Returns:
    - An integer representing the most valuable node in the network
    """
    # Calculate betweenness centrality for all nodes
    betweenness_centrality = nx.betweenness_centrality(network)

    # Find the node with the highest betweenness centrality
    most_valuable_node = max(betweenness_centrality, key=betweenness_centrality.get)

    return most_valuable_node
