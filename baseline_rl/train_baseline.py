import numpy as np
import os
import torch
import torch.optim as optim

from batch_env_rl import BatchEnvRL
from neuralconet import NeuralCombOptRL

use_cuda = True

model = NeuralCombOptRL(input_dim=3, embedding_dim=128, hidden_dim=128,
                        max_decoding_len=10, terminating_symbol='<0>', n_glimpses=2,
                        n_process_block_iters=3, tanh_exploration=10, use_tanh=True, is_train=True, use_cuda=use_cuda)

actor_optim = optim.Adam(model.actor_net.parameters(), lr=1e-4)

critic_exp_mvg_avg = torch.zeros(1)
beta = 0.9

if use_cuda:
    model = model.cuda()
    # critic_mse = critic_mse.cuda()
    critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

n_epochs = 100
n_sims = 100
n_sims_val = 10
batch_size = 32
step = 0
log_step = 1
for e in range(0, n_epochs):
    model.train()
    env = BatchEnvRL(n_envs=batch_size, n_nodes=10, adaptive=False)
    # Use beam search decoding for validation
    model.actor_net.decoder.decode_type = "stochastic"
    for b in range(0, n_sims):
        batch = env.get_features().copy()
        # batch in (batch_size, n_nodes, dim)
        batch = torch.from_numpy(batch)

        if use_cuda:
            batch = batch.cuda()
        # for NCO we need (batch_size, dim, n_nodes)
        batch = batch.transpose(-1, -2)

        _, probs, actions, action_idxs = model(batch)

        action_idxs = torch.stack(action_idxs).transpose(0, 1)
        nodes = action_idxs + 1
        ones = torch.ones(batch_size, 1)
        if use_cuda:
            ones = ones.cuda()

        nodes = torch.cat([ones, nodes], dim=-1)
        rwds, pens = env.check_solution(nodes)
        R = rwds + pens
        R = torch.from_numpy(R)
        if use_cuda:
            R = R.cuda()

        if b == 0:
            critic_exp_mvg_avg = R.mean()
        else:
            critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

        advantage = R - critic_exp_mvg_avg

        logprobs = 0
        nll = 0
        for prob in probs:
            # compute the sum of the log probs
            # for each tour in the batch
            logprob = torch.log(prob)
            nll += -logprob
            logprobs += logprob
        # guard against nan
        nll[(nll != nll).detach()] = 0.
        # clamp any -inf's to 0 to throw away this tour
        logprobs[(logprobs < -1000).detach()] = 0.
        logprobs = logprobs.unsqueeze(-1)

        # multiply each time step by the advanrate
        reinforce = - advantage * logprobs
        actor_loss = reinforce.mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

        env.reset()
        step += 1

        if step % log_step == 0:
            print(
                f'epoch: {e + 1}, batch: {b + 1}, avg reward: {R.mean().data:.2f}, critic: {critic_exp_mvg_avg.data:.2f}')

            model_dir = os.path.join('model', 'run01')
            if not os.path.exists(model_dir):
                print(f'Creating a new model directory: {model_dir}')
                os.makedirs(model_dir)

            checkpoint = {
                'actor': model.state_dict(),
                'optimizer': actor_optim.state_dict(),
                'epoch': e,
                'avg_rwd': R.mean().data,
                'step': step,
            }
            torch.save(checkpoint, os.path.join(model_dir, f'baseline-epoch{e}-step{step}.pt'))

        # Use beam search decoding for validation
    model.actor_net.decoder.decode_type = "greedy"

    print('validation')
    model.eval()
    env_val = BatchEnvRL(n_envs=1, n_nodes=10, adaptive=False, seed=1234)
    R_ = 0
    for b_val in (0, n_sims_val):

        batch = env.get_features().copy()
        # batch in (batch_size, n_nodes, dim)
        batch = torch.from_numpy(batch)

        if use_cuda:
            batch = batch.cuda()
        # for NCO we need (batch_size, dim, n_nodes)
        batch = batch.transpose(-1, -2)

        _, probs, actions, action_idxs = model(batch)

        action_idxs = torch.stack(action_idxs).transpose(0, 1)
        nodes = action_idxs + 1
        ones = torch.ones(batch_size, 1)
        if use_cuda:
            ones = ones.cuda()

        nodes = torch.cat([ones, nodes], dim=-1)
        rwds, pens = env.check_solution(nodes)
        R = rwds + pens
        R_ += R

    print(f'validation avg rwds {np.mean(R_)}')
