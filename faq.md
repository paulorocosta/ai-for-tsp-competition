---
description: Frequently asked questions
---

# FAQ

## 1. Missing time windows invalidate feasibility?

Yes, our interpretation is that if you violate one of the time windows from above \(you can still be early without any penalty\) we deem the solution 'infeasible'.  
This is also true for when you exceed `maxT`. Note that if you only exceed the time windows a couple of times there is still a chance you can get a 'good' tour out of it even though it's 'infeasible'.

## 2. How many instances are we expected to solve in Track 1 \(Surrogate\) ?

In the first track you are expected to solve a single instance of the problem. 

## 3. How many instances are we expected to solve in Track 2 \(Reinforcement\)?

In the second track you are expected to learn a policy and solve 1,000 instances sampled 100 times. 

## 4. Is there a specific hardware for the trained models?

We won't enforce any limits there. While using more hardware certainly gives an advantage, we believe there is more to be gained from developing smart and efficient solutions, especially considering the enormous size of the search space. Using a big cluster that might have downtime during the test phase is at your own risk, and might give someone with smaller but more reliable hardware an advantage. We do encourage all participants, when submitting their code, to give us an indication of what kind of hardware they used. This will not be used in determining the winner, but it will give us an indication if the winning team indeed mainly won due to their hardware capabilities or due to their inventive solution. Besides this, there will be possibilities for not only the winning teams but also the teams who develop inventive solutions to publish their results via the DSO workshop at IJCAI2021. This way, participating in the competition should be rewarding no matter the hardware capabilities.

## 4. Are the instances of the validation and test phases different?

Yes, the instances for both tracks between the validation and test phases will be different. This is to ensure that participants do not overfit to the validation instances. More details will come soon!

