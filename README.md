# AI for TSP Competititon

In a Travelling Salesman Problem (TSP), the goal is to find the tour with the smallest cost visiting all locations (
customers) in a network exactly once.

However, in practical applications, one rarely knows all the travel costs between locations precisely. Moreover, there
could be specific time windows at which customers need to be served, and specific customers can be more valuable than
others. Lastly, the salesman is often constrained by a maximum capacity or travel time, representing a limiting factor
in the number of nodes that can be visited.

In this competition, we consider a more realistic version of the classical TSP, i.e., the time-dependent orienteering
problem with stochastic weights and time windows (TD-OPSWTW). In this formulation, the stochastic travel times between
locations are only revealed as the salesman travels in the network. The salesman starts from a depot and must return to
the depot at the end of the tour. Moreover, each node (customer) in the network has its prize, representing how
important it is to visit a given customer on a tour. Each node has associated time windows. We consider that a salesman
may arrive earlier at a node without compromising its prize, but the salesman has to wait until the opening times to
serve the customer. Lastly, the salesman must not violate a total travel time budget while collecting prizes in the
network. The goal is to collect the most prizes in the network while respecting the time windows and the total travel
time of a tour allowed to the salesman.

## Dependencies

* Python=3.8 (should be OK with v >= 3.6)
* PyTorch=1.8 (track 2 only)
* Numpy=1.20
* bayesian-optimization=1.1.0 (track 1 only)
* Pandas=1.2.4
* Conda=4.8.4 (optional)

Please check ``environment.yml``

We consider two tracks in the competition: Surrogate-based Optimization (track 1) and Reinforcement Learning (track 2).

## Track 1: Surrogate-based Optimization

In the surrogate-based track, the goal is to find the tour s that gives the highest reward for one instance i:

`s* = argmax E[f(s,i)] `

We use the expected value because the simulator is stochastic: it can give different rewards even if the same route is
evaluated multiple times. The expected value for a route is approximated by evaluating the objective `f` for that route
ten thousand times and calculating the average reward. This computation takes multiple seconds on standard hardware.
Therefore, the problem can be seen as an expensive optimization problem. We goal is to solve this problem using a
surrogate-based optimization method such as Bayesian optimization.

In the end, given an instance, participants should submit the route that they believe gives the highest reward for that
instance.

### Objective

For calculating the reward of a route, we first need to create the instance. This can be done by creating an
environment.

```python
from env import Env

n_nodes = 5
env = Env(n_nodes, seed=12345)  # Generate instance with n_nodes nodes
```

This code also appears in `baseline_surrogate/demo_surrogate.py`. When the main competition starts, the exact number of
nodes and the exact random seed will be given, so the instance is known.

To evaluate a solution, we need to make sure the solution is in the correct format. A solution always has the
form `[1,s_1,...,s_n]`, with `n` the number of nodes. The numbers `s_1,...,s_n` need to contain all integers from 1 to
n. This means that the number 1 will appear twice in the solution. As this number indicates the starting node, it means
that the route consists of starting from the starting node, visiting any number of nodes, then returning to the starting
node at some point. Any nodes that appear in the solution after returning to the starting node are ignored in the
environment. A solution can be run in the environment with the following code:

```python
sol = [1, 2, 3, 1, 4, 5]
print('Solution: ', sol)
obj_cost, rewards, pen, feas = env.check_solution(sol)
print('Time: ', obj_cost)
print('Rewards: ', rewards)
print('Penalty: ', pen)
print('Feasible: ', feas)
print('Objective: ', rewards + pen)
```

The term `rewards+pen` is the objective that needs to be optimized, where `pen` is a negative reward that comes from
violating constraints in the problem. However, as mentioned earlier, we evaluate the objective ten thousand times and
take the average, as the objective can differ even for the same solution due to the random noise present in the
environment. The exact objective is given in the function `objective`, which requires a solution and an environment as
input and provides the averaged objective as output:

```python
def objective(x, env):
    obj_cost, rewards, pen, feas = env.check_solution(x)
    MonteCarlo = 10000  # Number of Monte Carlo samples. Higher number means less noise.
    obj = 0  # Objective averaged over Monte Carlo samples, to be maximized with surrogate optimization
    for _ in range(MonteCarlo):
        obj_cost, rewards, pen, feas = env.check_solution(x)
        obj = obj + (rewards + pen)  # Maximize the rewards + penalties (penalties are negative)
    obj /= MonteCarlo
    return obj
```

In the competition, we provide the instance, and participants are expected to submit the solution that they believe
gives the best (highest) objective for this instance. An example solution submission is given in the
file `check_solution_surrogate.py`. The file submitted on the website should simply be a `.out` file that contains the
numbers of the solution on separate lines, e.g.:

```
1
2
3
1
4
5
```

### Using continuous solvers: example with Bayesian optimization with Gaussian processes

Though the optimization problem is discrete, it is possible to apply continuous optimization algorithms (such as
Bayesian optimization with Gaussian processes) to this problem by rounding solutions. We provide code for this
in `baseline_surrogate/demo_surrogate.py`, using a similar transformation of the search space as in
['Black-box combinatorial optimization using models with integer-valued minima, Laurens Bliek, Sicco Verwer & Mathijs de Weerdt'](https://doi.org/10.1007/s10472-020-09712-4 "IDONE paper")
. In this approach, we let the first variable denote any of the n nodes, the second variable any of the nodes that are
remaining, and so on until only one node is remaining. A continuous solver then only needs to use the right upper and
lower bounds, and then after rounding we can transform these values into a valid solution with the following code:

```python
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
```

The objective for a continuous solver such as Bayesian optimization is then given by:

```python
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
```

For the lower and upper bounds we need to be careful with the rounding, so instead of having the lower bound be 0 and
the upper bound the remaining number of cities, we add and subtract a small number:

```python
pbounds = {f'v{i}': (0.0, max(1e-4, n_nodes - i - 1e-4))
           # keep upper bound above 0 to avoid numerical errors, but subtract a small number so the np.floor function does the right thing
           for i in range(n_nodes)}
```

Finally, we can call the Bayesian optimization method as follows after installing it
with `conda install -c conda-forge bayesian-optimization`:

```python
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
```

Other surrogate algorithms might be able to use the `objective` function with its discrete input directly, or require to
transform the objective in a different way than the function `objBO` shown above. Participants are free to use their own
objective function, even multi-fidelity ones or using different ways to incorporate constraints, but the one that will
be used for evaluating a solution is the function `objective`.

## Track 2: Reinforcement Learning

In the Reinforcement Learning (RL) track, we are interested in a **policy**.

**Policy**: A policy in the TD-OPSWTW selects the next node to be visited given a sequence of previously visited nodes.
Note that to cope with the stochastic travel times, a policy must be <em>adaptive</em>. Therefore, a policy needs to
consider the instance information to construct tours dynamically that respect the time windows of nodes and the total
tour time allowed for the instance. Note that unlike Track 1, we are interested in general policies applicable to any
instance of the TD-OPSWTW in the training distribution. The following figure shows an example of a next node visitation
decision that has to be made by a policy visiting ``n=6`` nodes.

![image info](./figures/policy_rl2.png)

In the figure, a policy has visited nodes 1 (depot) and 6 with travel time ``t_{1,6}`` revealed after visiting node 6.
At this current decision epoch, the policy has to choose the next node to visit. The prizes ``p_i`` and time window
bounds  ``[l_i, u_i]`` are known and given in the instance, as well as the maximum allowed tour time `T`. The decision
should consider the prizes of each node, the time windows, and the total remaining travel time when selecting the next
node (in this case, node 3).

To achieve a feasible solution, a policy needs to visit nodes respecting the upper bounds of the time windows, i.e., it
can violate the lower bounds and arrive early without penalties and the maximum tour times. When the policy decides to
arrive early at a node, the travel time gets shifted to the beginning of the time window. For example, if the travel
time between the depot (node 1) and node 6 is lower than ``l_6``, the salesman has to wait until
``l_6`` to depart from that node. This information becomes available as soon as the salesman arrives at node 6. Lastly,
a policy must always return to the depot, and this travel time is also included in the maximum allowed tour time.

### Environment

The RL environment is implemented on an instance by instance basis. That is, to initiate the environment, one has to
pass an instance file. If an instance file is not given, the environment will create an instance on the fly. This
behaviour can be helpful for training. One can call the RL environment as follows

 ```python
from env_rl import EnvRL

env = EnvRL(n_nodes=5, seed=1234)
print(env.get_current_node_features())
print(env.get_seed())
print(env.get_sim_name())
print(env.get_instance_name())
```

Note that the previous call initiates an environment generating an instance on the fly with 5 nodes. Each instance is
assigned an `instance_name` and a `name`. Both are used in the submission file that is used to score submissions. Every
time the environment is `reset()`, new travel times are drawn at random (new Monte Carlo sample). We call each
simulation, i.e., each randomness, a `tour000`. During the evaluation, you will be given instances, seeds, and a number
of tours to generate for each instance file.

Note that the RL environment's default behaviour is ``adaptive``. This means that it expects a node as input of each
``step()`` call. However, we allow for the environment to be called in the same manner as Track 1 by
setting ``adaptive=False`` and calling `check_solution()` as before.

To call the environment from an instance file:

 ```python
from env_rl import EnvRL

env = EnvRL(from_file=True, x_path='data/valid/instances/instance0001.csv',
            adj_path='data/valid/adjs/adj-instance0001.csv')
print(env.n_nodes)
# 20
```

Note that when the environment is initiated, the first simulation is already started, i.e., calling ``reset()`` will
create a second simulation,i.e., ``tour002``.

#### Taking a step in the environment

A tour in the environment always starts from the depot. Thus, the first call of the ``step()`` method does not need to
include the depot. By convention, node 1 is always the depot. Therefore, a tour is considered complete if node 1 is
passed to the ``step()`` method. After that, no other prizes, penalties, or travel times are incurred. Please see the
example below:

 ```python
from env_rl import EnvRL

env = EnvRL(5, seed=123456)
env.step(2)
env.step(4)
env.step(5)
env.step(1)
env.step(3)
print(env.tour)
# [1, 2, 4, 5, 1]
```

#### Travel times, prizes, penalties, and violations

Each call of the ``step()`` method returns several useful metrics for the task. In order: total travel time of a
sequence, travel time between the previous node and the current node, prize collected at the current node, penalty
incurred at the current node, boolean of the feasibility after visiting the current node, type of violation, and a
boolean of the completed tour.

 ```python
from env_rl import EnvRL

env = EnvRL(5, seed=123456)
total_time, time, prize, penalty, feasible, violation, done = env.step(2)
```

Note that each violated constraint (not respecting the time windows from above or not respecting the maximum tour time)
incur penalties. In other words, it is possible to submit only infeasible tours. However, these will be penalised in the
final scores. Each type of violation is also identified
as ``0: no violation, 1: time window violation, 2: maximum tour time violation``. The maximum tour time violation takes
precedence over the time window violation and incurs the highest penalties. The prizes are between `[0, 1]` and depend
on the maximum travel time to the depot, with nodes farther from the depot assigned higher prizes. Penalties are as
follows:

* Violating a time window: ``-1.0``
* Violating the maximum tour time: ``-1.0*n``

#### Instance features

To aid learning, one can make use of instance and (maximum) travel times between locations. We point out that these
travel times are **not** exactly the travel times experienced in the simulations. One can recover the instance features
containing: node coordinates in 2D (used to generate the simulations), time windows, prizes, and maximum tour time. See
the example below:

 ```python
from env_rl import EnvRL

env = EnvRL(5, seed=123456)
inst, max_travel_times = env.get_features()
print(f'instance')
print(inst)
# instance
# [[1 65.0 27.0 0 339 0.0 354]
#  [2 49.0 32.0 10 27 0.16 354]
#  [3 56.0 23.0 287 323 0.1 354]
#  [4 171.0 36.0 153 195 1.0 354]
#  [5 43.0 8.0 14 53 0.28 354]]
print('max travel times')
print(max_travel_times)
# max travel times
# [[0 17 10 106 29]
#  [17 0 11 122 25]
#  [10 11 0 116 20]
#  [106 122 116 0 131]
#  [29 25 20 131 0]]
```

In the above example the first row of ``inst`` represents:

* node:1,
* coordinates: 65.0, 27.0
* tw: [0, 339]
* prize: 0
* max tour time: 354

### Baseline

We provide a baseline to the RL competition based
on ['Neural Combinatorial Optimization with Reinforcement Learning, Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi, Samy Bengio'](https://arxiv.org/abs/1611.09940 "cool paper")
. Note that the above approach is **not** adaptive and will not perform well in the given task. This baseline is just a
reference as to how RL can be used. Moreover, it only uses the coordinates and prizes to make decisions on complete
tours.

#### Calling the baseline:

    .
    ├── baseline_rl              # Baseline based on Neural CO
        python train_baseline.py

If you would like a reference to an adaptive method,
consider  ['Attention, Learn to Solve Routing Problems!, Wouter Kool, Herke van Hoof, Max Welling'](https://arxiv.org/abs/1803.08475 " another cool paper")
.

## Data

### Training

Participants are free to use any form of training as long it includes a Reinforcement Learning method. At the end of the
competition, we will invite the winners to submit their codes for compliance with the competition rules. If it becomes
clear that the proposed method does not use any form of RL or the submitted results cannot be reproduced with the code
provided, these teams will be disqualified.

### Validation

We provide *1000* instances as validation set ``instance0000.csv``, ``adj-instance0000.csv``. Note that this validation
set will be used throughout the competition to evaluate your submissions on the website. For each validation instance,
participants have to generate **100** simulations and use their methods to generate tours. Each tour (Monte Carlo
simulation) is assigned a name following ``tour000``. This naming scheme is already present in the environment. The
instances have different nodes: 20, 50, 100, 200 (250 instances each).

#### Validation instances:

    .
    ├── data              
        ├── valid
            ├── instances
                instance0001.csv
                ...
            ├── adjs
                adj-instance0001.csv
                ...

Participants can use any instances for training, but validation will always be done on the same instance sizes. Based on
the validation instances, we will select the best performing teams. These teams will be given a final test dataset. This
final test dataset will be used to compute the final scores and define the competition winners. Note that this test
dataset will follow the same generation process as the validation dataset. The details about the test dataset will be
revealed to the qualifying participants.

## Submission

A submission file example can be found in the ``baseline_rl`` folder, named ``example_output_rl.json``. The submission
file is a .json file containing the instance name, followed by the number of nodes, seed, and the 100 simulated tours.
Thus ``tour001`` until ``tour100``.

 ```json
 {
  "instance0001": {
    "nodes": 5,
    "seed": 12345,
    "tours": {
      "tour001": [
        1,
        2,
        3,
        1,
        5,
        4
      ],
      "tour002": [
        1,
        5,
        4,
        2,
        1,
        3
      ]
    }
  },
  "instance0002": {
    "nodes": 4,
    "seed": 12345,
    "tours": {
      "tour001": [
        1,
        2,
        1,
        3,
        4
      ],
      "tour002": [
        1,
        3,
        1,
        2,
        4
      ]
    }
  }
}
...
```

For the validation data, all seeds are the same ``seed: 12345``. Each tour name must be followed by an array of integers
of size ``n+1``. For a solution to be considered valid, the depot ``node: 1`` must appear twice in every array.
Moreover, the depot **must** appear in the first position. If the solutions do not comply with these standards, the
submission will be invalid.

## Scoring Submissions

We will score submissions considering the sum of prizes and penalties. That is, for each instance and tour (Monte Carlo
simulation)
we will compute ``score = prizes+penalties``. The scores of all Monte Carlo simulations will be averaged, and the
submission with the highest average ``score`` will be ranked highest. Note that all participants can compute their
validation scores before submission. We encourage you to do so as this will minimise the submission errors. You can
score a solution calling

```
python check_solution_rl.py
```

# Acknowledgements

Special thanks to https://github.com/pemami4911/neural-combinatorial-rl-pytorch for the Neural Combinatorial
Optimization code implemented in Pytorch that has been used as part of this repo.
