# Social-Network-Analysis
The main goal of this project is to explore social networks and social phenomena by modeling networks as graphs using the NetworkX module. The project involves performing various analyses based on theoretical knowledge about social networks and their behavior. Each Python file in this repository focuses on a different aspect of network analysis.  
## File Descriptions
### 1. Random Networks and Scale-Free Networks
This project aims to highlight the differences between random networks and scale-free networks. The functions in this file create instances of random networks, measure network metrics, conduct hypothesis testing, estimate the gamma parameter of scale-free networks, and classify networks as either random or scale-free. While random networks can be generated using this file, analyzing scale-free networks requires loading pre-existing instances of such networks.    
  
  #### How to Use:
  * Generate random networks directly using the provided functions.
  * Load and analyze scale-free networks by importing pre-existing network data.
  * Utilize the functions to measure network properties and classify the networks accordingly.
### 2. Community Detection
This project focuses on identifying communities within social networks through graph modeling. Using data from tweets by Knesset members, we aimed to identify the political affiliations of these members based on their retweets. Community detection algorithms such as Girvan-Newman, Louvain, and Clique Percolation were employed to identify the political sides of users and categorize them accordingly.  
  
 #### How to Use:
 * Load the dataset containing Knesset members' tweet data.
 * Apply the community detection algorithms to identify communities.
 * Visualize and interpret the results to understand the political landscape within the network.
### 3. Centrality Measurements
This project explores various centrality measurement algorithms in network analysis. The centrality metrics studied include degree centrality, closeness centrality, betweenness centrality, and PageRank. These metrics help in understanding the importance and influence of nodes within a network.  

  #### How to Use:
* Load a network dataset to analyze.
* Use the provided functions to compute different centrality metrics.
* Compare and interpret the centrality scores to identify key nodes within the network.
  
  ## Dependencies
  * Python 3.x
  * networkx
  * numpy
  * matplotlib
