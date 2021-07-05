import math
import numpy as np
import time

from env import Env

try:
    from bayes_opt import BayesianOptimization
except:
    print(
        'Please make sure the package bayesianoptimization is installed via conda-forge or pip. \n(conda install -c conda-forge bayesian-optimization)')


def objective(x, env):
    '''


    Parameters
    ----------
    x : array
        Vector of the form [1, x1, x2, ..., x_n].
        The integers from 1 to n have to appear
        in the part [x1,..., x_n], so the number
        1 appears twice in total.
    env : environment
        Environment of the TSP-like problem.

    Returns
    -------
    obj : float
        Objective to be maximized.

    '''

    print('x', x)
    obj_cost, rewards, pen, feas = env.check_solution(x)

    MonteCarlo = 10000  # Number of Monte Carlo samples. Higher number means less noise.
    obj = 0  # Objective averaged over Monte Carlo samples, to be maximized with surrogate optimization
    for _ in range(MonteCarlo):
        obj_cost, rewards, pen, feas = env.check_solution(x)
        # print('Time: ', obj_cost)
        # print('Rewards: ', rewards)
        # print('Penalty: ', pen)
        # print('Feasible: ', feas)
        # print('Objective: ', rewards+pen)
        obj = obj + (rewards + pen)  # Maximize the rewards + penalties (penalties are negative)
    obj /= MonteCarlo

    return obj


def check_surrogate_solution(x):
    '''


    Parameters
    ----------
    x : array
        Vector of the form [1, x1, x2, ..., x_n].
        The integers from 1 to n have to appear
        in the part [x1,..., x_n], so the number
        1 appears twice in total.

    Returns
    -------
    obj : float
        Corresponding objective.

    '''
    n_nodes = 65
    env = Env(n_nodes, seed=6537855)  # Generate instance with n_nodes nodes
    obj = objective(x, env)
    print('Solution quality (higher is better): ', obj)
    return obj


if __name__ == '__main__':

    ##Test phase
    n_nodes = 65
    env = Env(n_nodes, seed=6537855)  # Generate instance with n_nodes nodes

    print('Generated instance (n=65). Compute the average over evaluating the same solution multiple times...')
    # sol = [1, 4, 3, 2, 5, 1]
    sol = np.arange(1, n_nodes + 1)
    np.random.shuffle(sol)
    sol = np.concatenate(([1], sol))  # Make sure solution starts at depot
    print('Solution: ', sol)
    MonteCarlo = 10000  # Number of Monte Carlo samples. Higher number means less noise.
    time1 = time.time()
    obj = 0  # Objective averaged over Monte Carlo samples, to be used for surrogate modelling
    for _ in range(MonteCarlo):
        obj_cost, rewards, pen, feas = env.check_solution(sol)
        # print('Time: ', obj_cost)
        # print('Rewards: ', rewards)
        # print('Penalty: ', pen)
        # print('Feasible: ', feas)
        # print('Objective: ', rewards+pen)
        obj = obj + (rewards + pen) / MonteCarlo
    time_elapsed1 = time.time() - time1
    print('Time elapsed: ', time_elapsed1)
    print('Average objective: ', obj)

    time2 = time.time()
    obj = 0  # Objective averaged over Monte Carlo samples, to be used for surrogate modelling
    for _ in range(MonteCarlo):
        obj_cost, rewards, pen, feas = env.check_solution(sol)
        # print('Time: ', obj_cost)
        # print('Rewards: ', rewards)
        # print('Penalty: ', pen)
        # print('Feasible: ', feas)
        # print('Objective: ', rewards+pen)
        obj = obj + (rewards + pen) / MonteCarlo
    time_elapsed2 = time.time() - time2
    print('Time elapsed: ', time_elapsed2)
    print('Average objective: ', obj)

    time3 = time.time()
    obj = 0  # Objective averaged over Monte Carlo samples, to be used for surrogate modelling
    for _ in range(MonteCarlo):
        obj_cost, rewards, pen, feas = env.check_solution(sol)
        # print('Time: ', obj_cost)
        # print('Rewards: ', rewards)
        # print('Penalty: ', pen)
        # print('Feasible: ', feas)
        # print('Objective: ', rewards+pen)
        obj = obj + (rewards + pen) / MonteCarlo
    time_elapsed3 = time.time() - time3
    print('Time elapsed: ', time_elapsed3)
    print('Average objective: ', obj)

    print('Evaluating the objective function takes about ', (time_elapsed1 + time_elapsed2 + time_elapsed3) / 3,
          ' seconds on this machine.')


    def x_to_route(x):
        # After rounding x, transform it to a route.
        nodes = np.arange(1, n_nodes + 1).tolist()
        xnew = []
        for xi in x:
            i = int(xi)
            xnew.append(nodes[i])
            nodes.remove(nodes[i])
        xnew.insert(0, 1)  # Start from starting depot
        print(xnew)

        return xnew


    def objBO(**x):
        vars = [f'v{i}' for i in range(n_nodes)]
        xvars = [x[v] for v in vars]

        # Bayesianoptimisation does not naturally support integer variables.
        # As such we round them.
        xrounded = np.floor(np.asarray(xvars))
        xnew = x_to_route(xrounded)

        r = objective(xnew, env)

        # Bayesianoptimization maximizes by default.
        # Include some random noise to avoid issues if all samples are the same.
        eps = 1e-6
        rnoise = r + np.random.standard_normal() * eps
        return rnoise


    varnames = {f'v{i}' for i in range(n_nodes)}
    pbounds = {f'v{i}': (0.0, max(1e-4, n_nodes - i - 1e-4))
               # keep upper bound above 0 to avoid numerical errors, but subtract a small number so the np.floor function does the right thing
               for i in range(n_nodes)}

    optimizer = BayesianOptimization(
        f=objBO,
        pbounds=pbounds,
        verbose=2
    )

    random_init_evals = 10
    max_evals = 100
    optimizer.maximize(
        init_points=random_init_evals,
        n_iter=max_evals - random_init_evals)

    print('Finished optimizing with surrogate model.')

    solX = []
    for i in range(n_nodes):
        solX.append(optimizer.max['params'][f'v{i}'])

    solY = optimizer.max['target']
    print('Solution: ', solX)
    print('Objective: ', solY)
    route_solX = np.floor(np.asarray(solX))
    print('Rounded solution: ', route_solX)
    route_solX = x_to_route(route_solX)
    print('Route: ', route_solX)

    print('Do one run using this solution.')
    obj_cost, rewards, pen, feas = env.check_solution(route_solX)
    print('Time: ', obj_cost)
    print('Rewards: ', rewards)
    print('Penalty: ', pen)
    print('Feasible: ', feas)
    print('Objective: ', rewards + pen)

    print('Average objective using the found solution:', objective(route_solX, env))
