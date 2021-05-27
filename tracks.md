---
description: Because two tracks is better than one!
---

# Tracks

The competition is split into two tracks that are scored separately. Prizes will be awarded to the first and second places in both tracks.

## Track 1: Online Supervised Learning \(surrogates\)

Problem: time-dependent orienteering problem with stochastic weights and time windows \(TD-OPSWTW\) \[1\]. Given one instance, previously tried routes, and the reward for those routes, the goal is to learn a model that can predict the reward for a new route. Then an optimizer finds the route that gives the best reward according to that model, and that route is then evaluated, giving a new data point. Then the model is updated, and this iterative procedure continues for a fixed number of steps.

[\[1\]](https://www.sciencedirect.com/science/article/pii/S037722171630368X) C Verbeeck, Pieter Vansteenwegen, and E-H Aghezzaf. Solving the stochastic time-dependent orienteering problem with time windows. European Journal of Operational Research, 255\(3\):699–718, 2016.

## Track 2: Reinforcement Learning

The problem is the same as in track 1, but for multiple instances. We consider an environment \(simulator\) that can generate multiple instances following the same distribution and expects as output \(partial\) solutions containing the order at which the nodes should be visited. The simulator returns general instance features and the time-dependent cost for traversing the last edge in a given solution. The goal is to minimize the cost of the total path over multiple samples of selected test instances.

