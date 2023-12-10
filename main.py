import datetime

import numpy as np
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def extract_costs(xml_file):
    """
    Description: data is read from a file and loaded into the code.
    Parameters: A string (file_path): The path of the file to be read
    Returns: a 2D array (edge_costs), the distance matrix for all cities.
             an integer (cities), containing the amount of cities loaded in.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    edge_costs = []
    cities = 0

    for vertex in root.findall('.//vertex'):
        cities += 1
        temp = [float(edge.attrib['cost']) for edge in vertex.findall('edge')]
        edge_costs.append(temp)

    for i in range(len(edge_costs)):
        edge_costs[i].insert(i, 0)  # add in cost for travelling to current node (0)

    return edge_costs, cities


def generate_path(cities, start, pheromones):
    """
    Description: Calculates the path of an ant.
    Parameters: An integer (n), which determines the number of locations in the graph
                An integer (start), containing the value of the first city to be visited
                A numpy array (pheromones), which stores the values of the pheromone drops on each graph link
    Returns: The ordered list of nodes that the ant will visit
    """

    path = np.array([start])  # Set the path equal to only the start node
    all_cities = np.arange(0, cities)  # Create an ordered list of all nodes

    for city in range(1, cities):
        unvisited = np.delete(all_cities, path)  # Remove visited nodes
        if city == (cities - 1):
            choice = unvisited[0]
        else:
            next_pheromones = np.delete(pheromones[path[-1]], path)  # Get pheromones for all potential next nodes
            choice = random.choices(unvisited, weights=next_pheromones, cum_weights=None,k=1)  # Picks a next city at random, but weighted
        path = np.append(path, choice)  # Add city to path
    return path


def evaluate_path(path, edge_costs):
    """
    Description: Evaluates the fitness of an ant path
    Parameters: A string (path), defining an ant path
                An array (edge_costs), containing all the travel costs between cities
    Returns:
            fitness (int): The calculated fitness of the path
    """
    fitness = np.sum([edge_costs[path[i]][path[i + 1]] for i in range(len(path) - 1)])
    return fitness


def update_pheromones(reward, path, n):
    """
    Description: Calculates the amount which pheromones should change on graph links.
    Parameters: A numpy longdouble (reward), that stores the reward (Q/fitness)
                A list (path), containing an ant path
                An integer (n), that denotes the number of nodes in the graph
    Returns: A numpy array (p), in the form of a matrix containing the evaporation rate of the pheromones
    """
    probability = np.zeros((n, n), dtype='longdouble')
    for x in range(len(path) - 1):
        probability[path[x]][path[x + 1]] += np.longdouble(reward)  # Add 1/fitness to correct links
    return probability


def plot_results(fitness_list, m, iterations):
    """
    Description: Plots the fitnesses of each ant against their respective IDs
    Parameters: An array (fitness_levels) with all ant fitnesses
    Returns: none, but displays a graph
    """

    x = [i for i in range(1, iterations+1)]
    plt.scatter(x, fitness_list)
    plt.ylabel("Average fitness")
    plt.xlabel("Iteration")
    plt.show()
    name = "fitness" + str(m)
    #plt.savefig(name)
    return


def main(m, e, Q):
    edge_costs, cities = extract_costs('burma.xml')  # load the cost matrix and the amount of cities
    pheromones = np.random.rand(cities, cities)  # Initialise random pheromones
    iterations = int(10000 / m)  # Easier to keep track of iterations, given by the number of evaluations divided by the number of ants

    start_node = np.random.randint(cities)  # Randomly pick a start node
    fitness_list = []  # List that will store the average fitness per iteration

    for x in range(iterations):
        total_fitness = 0
        pheromone_update = np.zeros((cities, cities), dtype='longdouble')  # no pheromone drops as no ants have moved

        for ant in range(m):
            path = generate_path(cities, start_node, pheromones)  # Generate the path
            fitness = evaluate_path(path, edge_costs)  # Evaluate the path

            total_fitness += fitness
            reward = np.longdouble(Q) / np.longdouble(
                fitness)  # Calculate Q/fitness, in order to reward high fitness more, decrease Q
            pheromone_update = np.add(pheromone_update, update_pheromones(reward, path, cities))

        pheromones = np.add(pheromones, pheromone_update)
        pheromones *= e  # Evaporate all pheromone links

        fitness_list.append(total_fitness / m)

    plot_results(fitness_list, m, iterations)
    best_fitness = min(fitness_list)
    return best_fitness

# pheromone * (1/cost) / total_pheromone*(1/total_cost)
print(main(100, 0.5, 1))
