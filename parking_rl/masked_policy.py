"""
PPO için aksiyon maskesi: gözlemin son 4 boyutu {0,1} (geçerli yönler).
Logitler maskelenir (geçersiz softmax olasılığı 0); value ağı maskesiz çekirdek gözlemi kullanır.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


class CoreGridExtractor(FlattenExtractor):
    """Box gözlemin son N_MASK boyutunu (aksiyon maskesi) özellik vektörüne almaz."""

    N_MASK = 4

    def __init__(self, observation_space: gym.Space) -> None:
        assert isinstance(observation_space, spaces.Box)
        sh = observation_space.shape
        assert len(sh) == 1 and sh[0] > self.N_MASK
        d = int(sh[0]) - self.N_MASK
        core = spaces.Box(
            low=observation_space.low[:d],
            high=observation_space.high[:d],
            dtype=np.float32,
        )
        super().__init__(core)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return super().forward(observations[..., : -self.N_MASK])


def _mask_logits(logits: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """mask: 1 geçerli, 0 geçersiz. Tüm satır maskelenmişse logitlere dokunma."""
    ok = mask > 0.5
    row_ok = ok.any(dim=-1, keepdim=True)
    neg = th.finfo(logits.dtype).min / 2
    masked = logits.masked_fill(~ok, neg)
    return th.where(row_ok.expand_as(logits), masked, logits)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Ayrık aksiyonlar için logits maskeleme (SB3 Categorical)."""

    def _core_and_mask(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor | None]:
        if isinstance(obs, dict):
            return obs, None
        m = obs[..., -CoreGridExtractor.N_MASK :]
        return obs, m

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        _, mask = self._core_and_mask(obs)
        if mask is None:
            return super().get_distribution(obs)
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)
        logits = _mask_logits(logits, mask)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        _, mask = self._core_and_mask(obs)
        if mask is None:
            return super().forward(obs, deterministic)
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        logits = self.action_net(latent_pi)
        logits = _mask_logits(logits, mask)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        _, mask = self._core_and_mask(obs)
        if mask is None:
            return super().evaluate_actions(obs, actions)
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        logits = self.action_net(latent_pi)
        logits = _mask_logits(logits, mask)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy


def masked_policy_kwargs() -> dict[str, Any]:
    return {"features_extractor_class": CoreGridExtractor}
