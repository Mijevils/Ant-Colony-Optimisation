import numpy as np
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_costs(xml_file):
    """
    Description: data is read from a file and loaded into the code.
    Parameters: A string (file_path): The path of the file to be read
    Returns: a 2D array (distances), the distance matrix for all cities.
             an integer (cities), containing the amount of cities loaded in.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    distances = []
    cities = 0

    for vertex in root.findall('.//vertex'):
        cities += 1
        temp = [float(edge.attrib['cost']) for edge in vertex.findall('edge')]
        distances.append(temp)

    for i in range(len(distances)):
        distances[i].insert(i, 0)  # add in cost for travelling to current node (0)

    return np.array(distances), cities


def generate_path(num_cities, start, pheromones, distances):
    """
    Description: Calculates the path of an ant.
    Parameters: An integer (n), which determines the number of locations in the graph
                An integer (start), containing the value of the first city to be visited
                A numpy array (pheromones), which stores the values of the pheromone drops on each graph link
    Returns: The ordered list of nodes that the ant will visit
    """

    path = np.array([start])  # Set the path equal to only the start node
    all_cities = np.arange(0, num_cities)  # Create an ordered list of all nodes

    for city in range(1, num_cities):
        unvisited = np.delete(all_cities, path)  # Remove visited nodes
        if len(unvisited) == 1:
            choice = unvisited[0]
        else:
            next_pheromones = np.delete(pheromones[path[-1]], path)  # Get pheromones for all potential next nodes
            next_edges = np.delete(distances[path[-1]], path)
            heuristic = np.divide(np.divide(next_pheromones, next_edges), np.divide(np.sum(next_pheromones), np.sum(next_edges)))
            
            choice = random.choices(unvisited, weights=heuristic, cum_weights=None,k=1)  # Picks a next city at random, but weighted
        path = np.append(path, choice)  # Add city to path
    return path

def update_pheromones(reward, path, n):
    """
    Description: Calculates the amount which pheromones should change on graph links.
    Parameters: A numpy longdouble (reward), that stores the reward (Q/fitness)
                A list (path), containing an ant path
                An integer (n), that denotes the number of nodes in the graph
    Returns: A numpy array (p), in the form of a matrix containing the evaporation rate of the pheromones
    """
    probability = np.zeros((len(path), len(path)), dtype='longdouble')
    x_indices = path[:-1]
    y_indices = path[1:]
    probability[x_indices, y_indices] += np.longdouble(reward)
    return probability

def tests():

    test_dic = {'1': [50, 0.5, 0.5], '2': [50, 1.0, 1.0], '3': [50, 1.5, 1.5],
                '4': [100, 0.5, 0.5], '5': [100, 1.0, 1.0], '6': [100, 1.5, 1.5],
                '7': [150, 0.5, 0.5], '8': [150, 1.0, 1.0], '9': [150, 1.5, 1.5]}      
    
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(14, 7), gridspec_kw={'hspace': 0.6, 'wspace': 0.4})
    fig.tight_layout()
    
    for i, ax in enumerate(fig.axes):
        row = test_dic[str(i + 1)]
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
        fitness_list = main(row[0], row[1], row[2])
        ax.set_title(label="M = %s, E = %s, Q = %s | Best = %s" % (row[0], row[1], row[2], int(min(fitness_list))))
        ax.plot(np.arange(len(fitness_list)), fitness_list)

    plt.show()
                

def main(m, e, Q):
    distances, num_cities = extract_costs('burma.xml')  # load the cost matrix and the amount of cities
    pheromones = np.random.rand(num_cities, num_cities)  # Initialise random pheromones
    np.fill_diagonal(pheromones, 0)
    iterations = int(10000 / m)  # Easier to keep track of iterations, given by the number of evaluations divided by the number of ants

    start_node = np.random.randint(num_cities)  # Randomly pick a start node
    fitness_list = []  # List that will store the average fitness per iteration

    for x in tqdm(range(iterations)):
        total_fitness = 0
        pheromone_update = np.zeros((num_cities, num_cities), dtype='longdouble')  # no pheromone drops as no ants have moved

        for ant in range(m):
            path = generate_path(num_cities, start_node, pheromones, distances)  # Generate the path
            fitness = np.sum(distances[path[:-1], path[1:]])

            total_fitness += fitness
            reward = np.longdouble(Q) / np.longdouble(fitness)
            pheromone_update = np.add(pheromone_update, update_pheromones(reward, path, num_cities))

        pheromones = np.add(pheromones, pheromone_update)
        pheromones *= e

        fitness_list.append(total_fitness / m)

    return fitness_list

if __name__ == "__main__":
    tests()
