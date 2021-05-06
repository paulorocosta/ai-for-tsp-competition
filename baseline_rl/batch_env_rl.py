import numpy as np
import os

from env_rl import EnvRL


class BatchEnvRL:

    def __init__(self, n_envs, n_nodes=50, w=(20, 40, 60, 80, 100), maxT_pen=-1.0, tw_pen=-1.0, seed=None,
                 from_file=False, x_dir=None, adj_dir=None, verbose=False, adaptive=True):
        self.n_nodes = n_nodes
        self.adaptive = adaptive
        self.envs = []
        if from_file:
            print(f'Reading instance features from {x_dir}...')
            print(f'Reading instance time matrix from {adj_dir}...')

            count = 0
            for path_ in os.listdir(x_dir):
                if os.path.isfile(os.path.join(x_dir, path_)):
                    count += 1
                    self.envs.append(EnvRL(from_file=from_file, x_path=os.path.join(x_dir, path_),
                                           adj_path=os.path.join(adj_dir, path_), adaptive=adaptive))
                    if count >= n_envs:
                        break

            self.n_envs = count
        else:
            # print('Generating instances on the fly...')
            for i in range(n_envs):
                self.envs.append(EnvRL(n_nodes=n_nodes, w=w, maxT_pen=maxT_pen, tw_pen=tw_pen,
                                       seed=seed, verbose=verbose, adaptive=adaptive))
            self.n_envs = n_envs
        # print(f'Created {self.n_envs} environments.')

    def reset(self):
        for env in self.envs:
            env.reset()

    def get_features(self):
        x = np.zeros((self.n_envs, self.n_nodes, 3))
        idx = 0
        for env in self.envs:
            x[idx] = np.concatenate((self._normalize_features(env.x[:, 1:3]), env.x[:, -2, None]), axis=-1)
            idx += 1
        return x

    @staticmethod
    def _normalize_features(x):
        max_x = np.max(x, axis=0)
        min_x = np.min(x, axis=0)

        return (x - min_x) / (max_x - min_x)

    def step(self, next_nodes):
        tour_time = np.zeros((self.n_envs, 1))
        time_t = np.zeros((self.n_envs, 1))
        rwd_t = np.zeros((self.n_envs, 1))
        pen_t = np.zeros((self.n_envs, 1))
        feas = np.ones((self.n_envs, 1), dtype=bool)
        violation_t = np.zeros((self.n_envs, 1))

        idx = 0
        for env in self.envs:
            tour_time[idx], time_t[idx], rwd_t[idx], pen_t[idx], feas[idx], violation_t[idx], _ = env.step(
                next_nodes[idx][0])
            idx += 1
        return tour_time, time_t, rwd_t, pen_t, feas, violation_t

    def check_solution(self, sols):
        sols = sols.cpu().detach().numpy()
        if self.adaptive:
            pass
        else:
            rwds = np.zeros((self.n_envs, 1))
            pens = np.zeros((self.n_envs, 1))
            idx = 0
            for env in self.envs:
                _, rwds[idx], pens[idx], _ = env.check_solution(sols[idx])
                idx += 1
            return rwds, pens
