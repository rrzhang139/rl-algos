import numpy as np
import random


def rssm_1d_demo(
    episodes=120,          # how many episodes to run
    horizon=15,            # MPC planning horizon
    n_sequences=200,       # CEM samples / iteration
    plan_iters=3,          # CEM iterations
    elite_frac=0.1,        # top‑k fraction for elite set
    noise_std=0.1,         # env transition noise
):
    """Tiny PlaNet‑style demo in 1‑D using NumPy only."""

    # ---- simple 1‑D point‑mass environment ----
    # state = position x  ;  action a ∈ {‑1,0,1}
    # dynamics:  x' = x + a + Gaussian_noise
    # reward:   r = – x'^2     (maximize by driving x toward 0)

    action_space = np.array([-1.0, 0.0, 1.0])

    # ----- RSSM parameters we will learn (here: linear) -----
    A = 1.0     # x' ≈ A·x  + B·a
    B = 0.0

    replay = []          # (x, a, x') tuples
    returns_history = []  # episode returns for logging

    # -------- helper: environment step --------
    def env_step(x, a):
        x_next = x + a + np.random.randn() * noise_std
        reward = -x_next**2
        return x_next, reward

    # -------- helper: CEM planner in latent space (here latent==state) --------
    def plan_action(x):
        mu = np.zeros(horizon)
        sigma = np.ones(horizon)
        for _ in range(plan_iters):
            seqs = np.random.randn(n_sequences, horizon) * sigma + mu
            seqs = np.clip(np.round(seqs), -1, 1)          # discrete {‑1,0,1}
            returns = np.zeros(n_sequences)

            for i, seq in enumerate(seqs):
                xp, ret = x, 0.0
                for a in seq:
                    xp = A * xp + B * a                    # model rollout
                    ret += -xp**2                          # model reward
                returns[i] = ret

            elite_idx = returns.argsort()[-int(elite_frac * n_sequences):]
            elite = seqs[elite_idx]
            mu, sigma = elite.mean(axis=0), elite.std(axis=0) + 1e-3

        # MPC: execute first act
        return int(np.round(mu[0]))

    # ------------- main loop -------------
    for ep in range(1, episodes + 1):
        x = np.random.uniform(-5, 5)     # random start
        ep_return = 0.0

        for t in range(100):             # limit episode length
            # exploration: first 50 transitions are random
            a = plan_action(x) if len(
                replay) > 50 else random.choice(action_space)
            x_next, r = env_step(x, a)

            replay.append((x, a, x_next))
            ep_return += r
            x = x_next

        returns_history.append(ep_return)

        # ----- “learning” phase: linear least‑squares fit of dynamics -----
        if len(replay) > 50:
            X = np.array([s for (s, _, _) in replay])
            Acol = np.array([a for (_, a, _) in replay])
            Y = np.array([sp for (_, _, sp) in replay])
            feats = np.column_stack([X, Acol])
            # solve  Y ≈ feats @ W   ⇒  W = (featsᵀ feats)⁻¹ featsᵀ Y
            W, *_ = np.linalg.lstsq(feats, Y, rcond=None)
            A, B = W

        # -------- logging --------
        if ep % 20 == 0:
            recent = np.mean(returns_history[-20:])
            print(
                f"Episode {ep:03d} | avg return (last 20): {recent:6.2f}   |  A≈{A:5.3f}, B≈{B:5.3f}")

    return returns_history


# -------- Run the demo --------
rssm_1d_demo(episodes=120)
