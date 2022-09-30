import os
import time
from pathlib import Path

import numpy.random as rnd
import operators
import helper_functions

from alns import ALNS
from alns.criteria import SimulatedAnnealing
from env_alns import alnsState
from op_utils.instance import read_instance

PARAMETERS_FILE = "configs/ALNS_debug.json"
DEFAULT_RESULTS_ROOT = "single_runs/"


def run_alns(folder, exp_name, **kwargs):
    start_time = time.time()
    print('starting now :-)')

    problem_instance = kwargs['problem_instance']
    seed = kwargs['rseed']
    iterations = kwargs['iterations']

    # LOAD INSTANCE
    base_path = Path(__file__).resolve().parents[1]
    test_data_instance_path = base_path.joinpath('data/test/instances')
    test_data_adj_path = base_path.joinpath('data/test/adjs')
    x_path = os.path.join(test_data_instance_path, problem_instance + '.csv')
    adj_path = os.path.join(test_data_adj_path, 'adj-' + problem_instance + '.csv')
    x, adj, instance_name = read_instance(x_path, adj_path)

    nodes = [(i + 1) for i in range(0, len(x))]

    # INITIAL SOLUTION
    random_state = rnd.RandomState(seed)
    state = alnsState(nodes, [], x, adj, instance_name, seed)
    initial_solution = operators.empty_route(state, init_node=1)

    # ALNS
    alns = ALNS(random_state)
    alns.add_destroy_operator(operators.low_node_removal)
    alns.add_destroy_operator(operators.modest_node_removal)
    alns.add_destroy_operator(operators.low_path_removal)
    alns.add_destroy_operator(operators.modest_path_removal)

    alns.add_repair_operator(operators.random_best_distance_repair)
    alns.add_repair_operator(operators.random_best_prize_repair)
    alns.add_repair_operator(operators.random_best_ratio_repair)

    criterion = SimulatedAnnealing(1, .25, 1 / 100)  # HillClimbing()

    # START EVALUATION ALNS
    result = alns.iterate(initial_solution, [5, 3, 1, 0], 0.8, criterion, iterations=iterations, collect_stats=True)
    solution = result.best_state
    best_objective = - solution.objective()

    # Save outputs of main in files
    helper_functions.write_output(folder, exp_name, problem_instance, seed, iterations, solution.tour, best_objective)
    elapsed_time = time.time() - start_time
    print('Execution time:', elapsed_time, 'seconds')
    return problem_instance, seed, iterations, best_objective


def main(param_file=PARAMETERS_FILE):
    parameters = helper_functions.readJSONFile(param_file)

    folder = DEFAULT_RESULTS_ROOT
    exp_name = str(parameters["problem_instance"]) + str("_rseed") + str(parameters["rseed"])
    problem_instance, seed, iterations, best_objective = run_alns(
        folder,
        exp_name,
        **parameters
    )

    return problem_instance, seed, iterations, best_objective


if __name__ == "__main__":
    main()
