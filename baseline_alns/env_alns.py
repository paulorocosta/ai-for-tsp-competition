import copy
from env import Env
from pathlib import Path
import os


def evaluate_individual_solution(instance_name, solution, n_nodes, seed):
    base_path = Path(__file__).resolve().parents[1]
    test_data_instance_path = base_path.joinpath('data/test/instances')
    test_data_adj_path = base_path.joinpath('data/test/adjs')
    x_path = os.path.join(test_data_instance_path, instance_name + '.csv')
    adj_path = os.path.join(test_data_adj_path, 'adj-' + instance_name + '.csv')

    env = Env(n_nodes=n_nodes, seed=seed, from_file=True, x_path=x_path, adj_path=adj_path)

    total_reward, total_pen = 0, 0
    for _ in range(5):
        tour_time, rewards, pen, feas = env.check_solution(solution)
        total_reward += rewards
        total_pen += pen
    score = - (total_reward + total_pen) / 5
    if score == -0.0:
        return 0
    return score


class alnsState:

    def __init__(self, nodes, init_tour, x, adj, instance_name, seed):
        self.nodes = nodes  # contains all the nodes and will not be updated!
        self.tour = init_tour

        self.seed = seed
        self.instance_name = instance_name
        self.x = x
        self.adj = adj

    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        # all nodes must be present in the evaluation function: [1,2,3,1,5] --> will stop evaluation after second '1'
        tour = self.tour
        unvisited_nodes = [x for x in self.nodes if x not in tour]
        evaluation_tour = tour + unvisited_nodes

        score = evaluate_individual_solution(self.instance_name, evaluation_tour, len(self.x), self.seed)
        return score

