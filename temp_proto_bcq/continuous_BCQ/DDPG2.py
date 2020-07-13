import copy
import numpy as np  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):

    def __init__(self, state_dim, action_dim, latent_dim, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + action_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, 1)

        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, action, z)

        return u, mean, std

    def decode(self, state, action, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, action, z], 1)))
        a = F.relu(self.d2(a))
        return self.d3(a)


class DDPG2(object):

    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005):

        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = VAE(state_dim, action_dim, latent_dim, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.discount = discount
        self.tau = tau
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def grad_pen(self, state, action):
        """Define the gradient penalty regularizer"""
        eps_a = state.clone().detach().data.normal_(0, 10)
        eps_b = action.clone().detach().data.normal_(0, 10)
        input_a_i = state + eps_a
        input_b_i = action + eps_b
        input_a_i.requires_grad = True
        input_b_i.requires_grad = True
        # Create the operation of interest
        score, _, _ = self.vae(input_a_i, input_b_i)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(
            outputs=score,
            inputs=[input_a_i, input_b_i],
            only_inputs=True,
            grad_outputs=[torch.ones_like(score)],
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )
        assert len(list(grads)) == 2, "length must be exactly 2"
        # Return the gradient penalty (try to induce 1-Lipschitzness)
        grads = torch.cat(list(grads), dim=-1)
        grads_norm = grads.norm(2, dim=-1)
        _grad_pen = (grads_norm - 1.).pow(2)
        grad_pen = _grad_pen.mean()
        return grad_pen

    def train(self, replay_buffer, iterations, batch_size=100, train_vae=False):

        if train_vae:
            vae_pre = 2500
            for i in range(vae_pre):
                # Sample replay buffer
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
                # Variational Auto-Encoder loss
                recon, mean, std = self.vae(state, action)
                recon_loss = F.mse_loss(recon, reward)
                KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                vae_loss = recon_loss + 0.5 * KL_loss
                # GP
                vae_loss += 10 * self.grad_pen(state, action)
                # Optimize VAE
                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                self.vae_optimizer.step()
                print("reward vae step done, {}/{}".format(i, vae_pre))

        for it in range(iterations):

            # Sample replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # XXX: use the reconstructed reward instead of the real one
            recon, _, _ = self.vae(state, action)
            # print("GT: {} | VAE: {}".format(reward[0], recon[0]))
            reward.copy_(recon)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (not_done * self.discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
