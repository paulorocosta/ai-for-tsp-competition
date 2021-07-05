import json
import numpy as np
import os
from env_rl import EnvRL
from pathlib import Path


def score_rl_solution(submission_filepath='example_output_rl.json', final_submission=False):
    base_path = Path(__file__).parent.absolute()
    test_data_instance_path = base_path.joinpath('data/test/instances')
    test_data_adj_path = base_path.joinpath('data/test/adjs')

    f = open(submission_filepath)
    submission = json.load(f)

    scores = []
    n_feas_sols = 0
    for instance_name in submission.keys():
        x_path = os.path.join(test_data_instance_path, instance_name + '.csv')
        adj_path = os.path.join(test_data_adj_path, 'adj-' + instance_name + '.csv')
        seed = submission[instance_name]['seed']
        env = EnvRL(from_file=True, seed=seed, x_path=x_path, adj_path=adj_path)

        instance = submission[instance_name]
        if final_submission:
            n_tours = len(instance['tours'].keys())
            assert n_tours == 100, f'each instance must have 100 tours, but found {n_tours} in {instance_name}'
        for tour_name in instance['tours'].keys():
            sol = instance['tours'][tour_name]
            for node in sol[1:]:
                env.step(node)
            rewards = env.get_collected_rewards()
            pen = env.get_incurred_penalties()
            feas = env.get_feasibility()
            assert tour_name == env.get_sim_name(), f'submission {tour_name} in {instance_name} is in the wrong order.'
            score = rewards + pen
            n_feas_sols += float(feas)
            scores.append(score)
            env.reset()

    avg_score = np.mean(scores)

    return np.round(avg_score, 5)


if __name__ == '__main__':
    print(f'Your submission scored {score_rl_solution():.05f}')
