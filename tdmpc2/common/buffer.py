import numpy as np
import torch


class ReplayBuffer:
    """Replay buffer that stores entire episodes.

    Parameters
    ----------
    obs_dim : int
        Dimension of observation vectors.
    act_dim : int
        Dimension of action vectors.
    size : int
        Maximum number of transitions to keep in memory.
    horizon : int
        Length of the training snippets sampled from the buffer.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int, horizon: int = 1):
        self.obs_buf = np.zeros((size + 1, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=bool)
        self.ep_idx_buf = np.zeros(size, dtype=np.int32)
        self.ptr = 0
        self.max_size = size
        self.size = 0
        self.horizon = horizon
        self.curr_ep = 0

    def store(self, obs, act, rew, next_obs, done):
        """Insert a transition at the current pointer."""
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ep_idx_buf[self.ptr] = self.curr_ep
        # next observation is stored at the next index so we can
        # easily retrieve (K+1) observation sequences later.
        next_ptr = (self.ptr + 1) % (self.max_size + 1)
        self.obs_buf[next_ptr] = next_obs

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if done:
            self.curr_ep += 1

    def _valid_start(self, idx: int) -> bool:
        """Check whether a ``horizon`` window starting at ``idx`` is valid."""
        if idx + self.horizon >= self.size:
            return False
        ep = self.ep_idx_buf[idx]
        return (self.ep_idx_buf[idx : idx + self.horizon] == ep).all()

    def sample_batch(self, batch_size: int):
        """Sample a batch of sequential snippets.

        Returns a dict with:
        ``obs``  -- (batch, horizon+1, obs_dim)
        ``act``  -- (batch, horizon, act_dim)
        ``rew``  -- (batch, horizon)
        ``done`` -- (batch, horizon)
        """
        assert self.size > self.horizon, "Not enough data to sample"

        idxs = []
        max_start = self.size - self.horizon - 1
        while len(idxs) < batch_size:
            # Takes a bit longer, but will be used on updates which can be configured as hyperparameter (steps_per_update)
            idx = np.random.randint(0, max_start + 1)
            if self._valid_start(idx):
                idxs.append(idx)

        idxs = np.array(idxs)
        obs = np.stack([
            self.obs_buf[i : i + self.horizon + 1] for i in idxs
        ])
        act = np.stack([
            self.act_buf[i : i + self.horizon] for i in idxs
        ])
        rew = np.stack([
            self.rew_buf[i : i + self.horizon] for i in idxs
        ])
        done = np.stack([
            self.done_buf[i : i + self.horizon] for i in idxs
        ])
        # next_obs = np.stack([
        #     self.obs_buf[i + 1 : i + self.horizon + 2] for i in idxs
        # ])

        batch = dict(
            obs=torch.tensor(obs, dtype=torch.float32),
            act=torch.tensor(act, dtype=torch.float32),
            rew=torch.tensor(rew, dtype=torch.float32),
            done=torch.tensor(done, dtype=torch.float32),
            # next_obs=torch.tensor(next_obs, dtype=torch.float32),
        )
        return batch
