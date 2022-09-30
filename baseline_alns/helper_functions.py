import json
import copy
import csv
import dask

from op_utils import op
from pathlib import Path
from multiprocessing.pool import Pool


# --- FILE READING AND WRITING ------------------------------
def write_output(folder, exp_name, problem_instance, seed, iterations, solution, best_objective):
    """Save outputs in files"""
    output_dir = folder
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Final pop
    with open(output_dir + exp_name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([problem_instance, seed, iterations, solution, best_objective])


def readJSONFile(file, check_if_exists=False):
    """This function reads any json file and returns a dictionary."""
    if (not Path(file).is_file()) and check_if_exists:
        return None
    with open(file) as f:
        data = json.load(f)
    return data


# --- DISTRANCE REPAIR --------------------------------------
def get_best_distance_insertion(nodes, tour, adj):
    """returns insertion of index which results in least addition distance"""
    distances = {}
    for node in nodes:
        distances[node] = {}

    for node in nodes:
        for inx in range(1, len(tour)):
            predecessor_node = tour[inx - 1]
            successor_node = tour[inx]
            distances[node][inx] = adj[node - 1][predecessor_node - 1] + adj[node - 1][successor_node - 1] \
                                   - adj[predecessor_node - 1][successor_node - 1]  # -1 because node 0 does not exist

    best_index, best_node, distance = None, None, 100000
    for key, value in distances.items():
        for key2, value2 in value.items():
            if value2 < distance:
                best_index = key2
                best_node = key

    return best_index, best_node


def get_best_distance_insertion_for_node(node, tour, adj):
    """returns insertion of index and node which results in least addition distance"""
    distances = {}
    for inx in range(1, len(tour)):
        predecessor_node = tour[inx - 1]
        successor_node = tour[inx]
        distances[inx] = adj[node - 1][predecessor_node - 1] + adj[node - 1][successor_node - 1] \
                         - adj[predecessor_node - 1][successor_node - 1]  # -1 because node 0 does not exist
    return min(distances, key=distances.get)


# ---- PRIZE REPAIR ----------------------------------------------------------------------------------------------

def multiprocess_best_prize_insertions_for_node(nodes, tour, inx, node, x, adj):
    new_tour = copy.deepcopy(tour)
    new_tour.insert(inx, node)

    total_reward, total_pen = 0, 0
    for i in range(1):
        tour_time, rewards, pen, feas = op.tour_check(new_tour, x, adj, -1.0, -1.0, len(nodes))
        total_reward += rewards
        total_pen += pen
    score = - (total_reward + total_pen) / 1
    return {tuple(new_tour): score}


def get_best_prize_insertion_for_node(node, nodes, tour, input_score, adj, x):
    """returns insertion of index and node which results in least addition distance"""

    multiprocess_results = []
    best_new_tour, best_score = None, input_score
    for inx in range(1, len(tour)):
        result = dask.delayed(multiprocess_best_prize_insertions_for_node)(nodes, tour, inx, node, x, adj)
        multiprocess_results.append(result)
    results = []

    # from multiprocessing.pool import Pool
    # pool = Pool()
    # async_results = [pool.apply_async(multiprocess_best_prize_insertions_for_node, args=(nodes, tour, inx, node, x, adj)) for inx in range(1, len(tour))]
    # results = [ar.get() for ar in async_results]


    for item in results:
        for new_tour, score in item.items():
            if best_score > score:
                best_new_tour = list(new_tour)
                best_score = score

    if best_new_tour is None:
        return input_score, tour
    else:
        return best_score, best_new_tour


# ---- RATIO REPAIR ----------------------------------------------------------------------------------------------

def multiprocess_best_ratio_insertions_for_node(nodes, tour, current_score, current_tour_time, inx, node, x, adj):
    new_tour = copy.deepcopy(tour)
    new_tour.insert(inx, node)

    total_tour_time, total_reward, total_pen = 0, 0, 0
    for i in range(1):
        tour_time, rewards, pen, feas = op.tour_check(new_tour, x, adj, -1.0, -1.0, len(nodes))
        total_tour_time += tour_time
        total_reward += rewards
        total_pen += pen
    tour_time = total_tour_time / 1
    score = - (total_reward + total_pen) / 1
    if tour_time - current_tour_time == 0:
        ratio = 0
    else:
        ratio = (score - current_score) / (tour_time - current_tour_time)
    return {tuple(new_tour): {'score': score, 'ratio': ratio, 'time': tour_time}}


def get_best_ratio_insertion_for_node(node, nodes, tour, input_score, input_time, adj, x):
    multiprocess_results = []
    for inx in range(1, len(tour)):
        result = dask.delayed(multiprocess_best_ratio_insertions_for_node)(nodes, tour, input_score, input_time, inx,
                                                                           node, x, adj)
        multiprocess_results.append(result)
    results = dask.compute(*multiprocess_results)

    # from multiprocessing.pool import Pool
    # pool = Pool()
    # async_results = [pool.apply_async(multiprocess_best_ratio_insertions_for_node, args=(nodes, tour, input_score, input_time,
    #                                                                                      inx, node, x, adj)) for inx in range(1, len(tour))]
    # results = [ar.get() for ar in async_results]

    best_new_tour, best_ratio = None, 0
    for item in results:
        for new_tour, result in item.items():
            if best_ratio > result['ratio']:
                best_new_tour = list(new_tour)
                best_ratio = result['ratio']
                best_score = result['score']
                best_time = result['time']

    if best_new_tour is None:
        return input_score, input_time, tour
    else:
        return best_score, best_time, best_new_tour
