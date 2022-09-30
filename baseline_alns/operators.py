import random
from helper_functions import *

evaluation_iterations = 1


# ----- Initial solution constructor ----------------------------------------------------------------------------------
def empty_route(state, init_node):
    state.tour = [init_node, init_node]
    return state


# ---- DESTROY OPERATORS ----------------------------------------------------------------------------------------------
def low_path_removal(current, random_state):
    """
    Removes an entire consecutive subpath, that is, a series of contiguous nodes.
    """
    if current.tour == [1, 1]:
        return current

    nodes = list(current.tour[:-1])  # only select the nodes that are in the current solution
    destroyed = current.copy()

    node_idx = random_state.choice(range(1, len(nodes)))
    node = nodes[node_idx]

    nr_nodes_to_remove = int((len(current.nodes)-1) * random.uniform(0, 0.25))
    idx_to_remove = [node + i for i in range(nr_nodes_to_remove)]

    destroyed.tour = [i for j, i in enumerate(current.tour) if j not in idx_to_remove or i == 1]
    return destroyed


def modest_path_removal(current, random_state):
    """
    Removes an entire consecutive subpath, that is, a series of contiguous nodes.
    """
    if current.tour == [1, 1]:
        return current

    nodes = list(current.tour[:-1])  # only select the nodes that are in the current solution
    destroyed = current.copy()

    node_idx = random_state.choice(range(1, len(nodes)))
    node = nodes[node_idx]

    nr_nodes_to_remove = int((len(current.nodes)-1) * random.uniform(0.2, 0.4))
    idx_to_remove = [node + i for i in range(nr_nodes_to_remove)]

    destroyed.tour = [i for j, i in enumerate(current.tour) if j not in idx_to_remove or i == 1]
    return destroyed


def low_node_removal(current, random_state):
    """
    iteratively removes nodes randomly from solution.
    """
    if current.tour == [1, 1]:
        return current

    nodes = list(current.tour[:-1])  # only select the nodes that are in the current solution
    destroyed = current.copy()

    nr_nodes_to_remove = int((len(nodes)-1) * random_state.uniform(0, 0.25))
    idx_to_remove = random_state.choice(range(1, len(nodes)), nr_nodes_to_remove, replace=False)
    destroyed.tour = [i for j, i in enumerate(current.tour) if j not in idx_to_remove]

    if destroyed.tour[-1] != 1:
        return 'something going wrong here'
    return destroyed


def modest_node_removal(current, random_state):
    """
    iteratively removes nodes randomly from solution.
    """
    if current.tour == [1, 1]:
        return current

    nodes = list(current.tour[:-1])  # only select the nodes that are in the current solution
    destroyed = current.copy()

    nr_nodes_to_remove = int((len(nodes)-1) * random_state.uniform(0.2, 0.4))
    idx_to_remove = random_state.choice(range(1, len(nodes)), nr_nodes_to_remove, replace=False)
    destroyed.tour = [i for j, i in enumerate(current.tour) if j not in idx_to_remove]

    if destroyed.tour[-1] != 1:
        return 'something going wrong here'
    return destroyed


# ---- REPAIR OPERATORS ----------------------------------------------------------------------------------------------
def random_best_distance_repair(current, random_state):
    """Randomly select a nr of nodes and add, according to a random generated sequence, at its least expensive position
    (in terms of distance) """
    visited = current.tour[:-1]  # all visited nodes
    not_visited = [x for x in current.nodes if x not in visited]  # all unvisited nodes

    # only add a random number of the nodes that are not in the destroyed solution back
    nodes_to_include = random.sample(not_visited, random_state.randint(0, len(not_visited)))
    nodes_to_include = sorted(nodes_to_include, key=lambda k: random.random())  # shuffled nodes to visit

    for node in nodes_to_include:
        # Find best position to insert node.
        index = get_best_distance_insertion_for_node(node, current.tour, current.adj)

        # Insert
        current.tour.insert(index, node)
    return current


def random_best_prize_repair(current, random_state):
    """Randomly select a nr of nodes and add, according to a random generated sequence, the nodes sequentially to their
    best positions (in terms of accumulated rewards) """
    visited = current.tour[:-1]  # all visited nodes
    not_visited = [x for x in current.nodes if x not in visited]  # all unvisited nodes

    # only add a random number of the nodes that are not in the destroyed solution back
    nodes_to_include = random.sample(not_visited, random_state.randint(0, len(not_visited)))
    nodes_to_include = sorted(nodes_to_include, key=lambda k: random.random())  # shuffled nodes to visit

    # get score and tour time of the current (destroyed) tour
    total_reward, total_pen = 0, 0
    for i in range(evaluation_iterations):
        tour_time, rewards, pen, feas = op.tour_check(current.tour, current.x, current.adj, -1.0, -1.0,
                                                       len(current.nodes))
        total_reward += rewards
        total_pen += pen

    current_score = - (total_reward + total_pen) / evaluation_iterations

    for node in nodes_to_include:
        current_score, new_tour = get_best_prize_insertion_for_node(node, current.nodes, current.tour, current_score, current.adj, current.x)
        current.tour = new_tour
    return current


def random_best_ratio_repair(current, random_state):
    """Find the best insertions to be done (in terms of additional reward/additional distance ratio)"""
    visited = current.tour[:-1]  # all visited nodes
    not_visited = [x for x in current.nodes if x not in visited]  # all unvisited nodes

    # only add a random number of the nodes that are not in the destroyed solution back
    nodes_to_include = random.sample(not_visited, random_state.randint(0, len(not_visited)))
    nodes_to_include = sorted(nodes_to_include, key=lambda k: random.random())  # shuffled nodes to visit

    # get score and tour time of the current (destroyed) tour
    total_tour_time, total_reward, total_pen = 0, 0, 0

    for i in range(evaluation_iterations):
        tour_time, rewards, pen, feas = op.tour_check(current.tour, current.x, current.adj, -1.0, -1.0,
                                                       len(current.nodes))
        total_reward += rewards
        total_pen += pen
        total_tour_time += tour_time

    current_score = - (total_reward + total_pen) / evaluation_iterations
    current_tour_time = total_tour_time / evaluation_iterations

    for node in nodes_to_include:
        current_score, current_time, new_tour = get_best_ratio_insertion_for_node(node, current.nodes, current.tour, current_score, current_tour_time, current.adj, current.x)
        current.tour = new_tour
    return current