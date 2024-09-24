import networkx as nx
import numpy as np
import scipy as sp
from scipy.stats import binomtest
import powerlaw




def random_networks_generator(n, p, num_networks=1, directed=False, seed=316202555):
    """
    Generates a list of random networks.

    Parameters:
    n (int): Number of nodes in each network, must be positive.
    p (float): Probability for each pair of nodes to be connected, must be between 0 and 1.
    num_networks (int, optional): Number of networks to generate. Defaults to 1.
    directed (bool, optional): Indicates whether each generated network is directed. Defaults to False.
    seed (int, optional): Seed for the random number generator. Defaults to 316202555.

    Returns:
    list: A list containing `num_networks` networkX graph objects.
    """

    networks = []
    for i in range(num_networks):
        network = nx.gnp_random_graph(n, p, seed=seed, directed=directed)
        networks.append(network)
        seed += 1  # changing the seed to create more varied array of networks
    return networks

def network_stats(network):
    """
    Calculate statistics for a given network.

    This function computes several statistics related to the degree of nodes
    and connectivity metrics.

    Parameters:
    network (networkx.Graph): A NetworkX graph object.

    Returns:
    dict: A dictionary containing the calculated statistics with the following keys:
        - 'degrees_avg': float, average degree of nodes in the graph.
        - 'degrees_std': float, standard deviation of degrees of nodes.
        - 'degrees_min': float, minimum degree of nodes.
        - 'degrees_max': float, maximum degree of nodes.
        - 'spl': float, average shortest path length between nodes.
        - 'diameter': float, diameter of the network (longest shortest path).
    """

    degrees = [degree for node, degree in network.degree()]
    degrees_avg = np.mean(degrees)
    degrees_std = np.std(degrees)
    degrees_min = np.min(degrees)
    degrees_max = np.max(degrees)

    stats = {
        'degrees_avg': degrees_avg,
        'degrees_std': degrees_std,
        'degrees_min': degrees_min,
        'degrees_max': degrees_max
    }

    # Calculate connectivity measures depending on the type of the network
    if network.is_directed():
        # Handle directed networks
        if nx.is_strongly_connected(network):
            # Calculate if the network is strongly connected
            spl = nx.average_shortest_path_length(network)
            diameter = nx.diameter(network)
        else:
            # If not strongly connected, set to infinity
            spl = float('inf')
            diameter = float('inf')
    else:
        # Handle undirected networks
        if nx.is_connected(network):
            # Calculate if the network is connected
            spl = nx.average_shortest_path_length(network)
            diameter = nx.diameter(network)
        else:
            # If not connected, set to infinity
            spl = float('inf')
            diameter = float('inf')

    stats['spl'] = spl
    stats['diameter'] = diameter

    return stats


def networks_avg_stats(networks):
    """
    Calculate average statistics across a list of networkX graphs.

    This function computes average values of several statistics including the degree
    distribution metrics and connectivity metrics such as the shortest path length and diameter
    for a list of networks. If only one network is provided, it returns the statistics for that network.

    Parameters:
    networks (list of networkx.Graph): A list of NetworkX graph objects.

    Returns:
    dict: A dictionary containing the averaged statistics with keys:
        - 'degrees_avg': float, average of the average degree of nodes across networks.
        - 'degrees_std': float, average of the standard deviation of degrees across networks.
        - 'degrees_min': float, average of the minimum degree of nodes across networks.
        - 'degrees_max': float, average of the maximum degree of nodes across networks.
        - 'spl': float, average shortest path length across networks.
        - 'diameter': float, average diameter across networks.
    """

    # If only one network is provided in the list, directly return its statistics
    if len(networks) == 1:
        return network_stats(networks[0])
    avg = []
    std = []
    minimum = []
    maximum = []
    spls = []
    diameters = []
    # Compute statistics for each network and store in lists
    for network in networks:
        stats = network_stats(network)
        avg.append(stats['degrees_avg'])
        std.append(stats['degrees_std'])
        minimum.append(stats['degrees_min'])
        maximum.append(stats['degrees_max'])
        spls.append(stats['spl'])
        diameters.append(stats['diameter'])

    # Calculate the mean of each statistic across all networks
    degrees_avg = np.mean(avg)
    degrees_std = np.mean(std)
    degrees_min = np.mean(minimum)
    degrees_max = np.mean(maximum)
    spl = np.mean(spls)
    diameter = np.mean(diameters)

    # Compile the averaged statistics into a dictionary
    stats = {
        'degrees_avg': degrees_avg,
        'degrees_std': degrees_std,
        'degrees_min': degrees_min,
        'degrees_max': degrees_max,
        'spl': spl,
        'diameter': diameter
    }
    return stats


def rand_net_hypothesis_testing(network, theoretical_p, alpha=0.05):
    """
    Perform a hypothesis test on the 'p' parameter of a network generated with the G(n, p) model.

    This function tests the null hypothesis that the probability of an edge existing between any two
    nodes in the network is equal to a given theoretical probability `theoretical_p`.

    Parameters:
    network (networkx.Graph): The networkX graph object to be tested.
    theoretical_p (float): The theoretical probability of an edge existing between any two nodes.
    alpha (float, optional): The significance level used to decide whether to reject the null hypothesis. Defaults to 0.05.

    Returns:
    tuple: A tuple containing:
        - p_value (float): The p-value of the test.
        - result (str): 'accept' if the null hypothesis is accepted, 'reject' if it is rejected.
    """

    # Calculate the total number of possible edges in an undirected graph without self-loops
    n = len(network.nodes())
    # Count the actual number of edges in the graph
    links = len(network.edges())
    max_edges = n * (n - 1) / 2  # Calculate the maximum number of edges for undirected graph

    # Perform a binomial test
    test_results = binomtest(links, n=int(max_edges), p=theoretical_p)
    p_value = test_results.pvalue
    # Determine the result based on the p-value and the significance level
    result = 'reject' if p_value < alpha else 'accept'
    return (p_value, result)


def most_probable_p(graph):
    """
    Determine the most probable 'p' value for a network based on hypothesis testing for each probability in [0.01, 0.1, 0.3, 0.6].

    Parameters:
    graph (networkx.Graph): A graph representing the network for which the probability of connection 'p' is to be estimated.

    Returns:
    float: The most probable 'p' value from predefined candidates [0.01, 0.1, 0.3, 0.6]. Returns -1 if no candidate is fits.
    """

    candidate_p = [0.01, 0.1, 0.3, 0.6]
    all_tests_results = []

    for prob in candidate_p:
        # Test the current 'p' using the hypothesis testing function
        test_res = rand_net_hypothesis_testing(graph, prob)
        # If hypothesis is accepted, append the result and corresponding 'p' to the list
        if test_res[1] == 'accept':
            all_tests_results.append([test_res, prob])

    # If no hypotheses were accepted, return -1
    if not all_tests_results:
        return -1

    # Initialize variables to find the 'p' with the highest test p value
    best_p = 0
    best_p_value = 0
    for result in all_tests_results:
        # if the p value is greater update the best_p variable to the current p
        if result[0][0] > best_p_value:
            best_p = result[1]

    return best_p


def find_opt_gamma(network, treat_as_social_network=True):
    """
    Estimates the optimal gamma parameter for a given network using the power law distribution.

    Parameters:
    network (networkx.Graph): The network for which gamma is to be calculated. It is treated as a networkX object.
    treat_as_social_network (bool): Determines whether the network should be treated as a social network. This affects
                                    the fitting process since social networks typically have discrete degree distributions.
                                    Default is True.

    Returns:
    float: The optimal gamma value for the network's degree distribution.
    """

    # Extract degree sequence
    degrees = [d for n, d in network.degree()]

    # Fit the power law distribution to the degree sequence
    results = powerlaw.Fit(degrees, discrete=treat_as_social_network)

    # The 'alpha' parameter from powerlaw is gamma
    gamma = results.power_law.alpha

    return gamma


def netwrok_classifier(network):
    """
    Classifies a given network as either a random network or a scale-free network.

    This function determines the type of network by estimating the gamma parameter of the network's degree distribution.
    If the gamma value is between 2 and 3, the network is classified as scale-free. Otherwise, the network is classified
    as a random network.

    Parameters:
    network (networkx.Graph): The network to be classified.

    Returns:
    int: Returns 1 if the network is classified as a random network, or 2 if it is classified as a scale-free network.
    """

    # Estimate the optimal gamma value for the network
    gamma = find_opt_gamma(network)
    # for scale-free networks the value of gamma is between 2 to 3, gamma value of random network is over 3
    if 2 < gamma < 3:
        return 2  # Scale-free
    return 1  # Random


