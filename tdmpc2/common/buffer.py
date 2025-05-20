import numpy as np
import torch

class ReplayBuffer:
    """Simple replay buffer for storing experience tuples."""

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        """Create buffers with fixed size.

        Parameters
        ----------
        obs_dim : int
            Dimension of observation space.
        act_dim : int
            Dimension of action space.
        size : int
            Maximum number of transitions to store.
        """

        # Pre-allocate numpy arrays to hold data.  Each array has shape
        #   [size, dim].  This avoids constantly resizing lists.
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.ptr, self.max_size = 0, size
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        """Add one transition to the buffer."""
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        # Circular buffer: overwrite old data when capacity is reached.
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int):
        """Sample a random batch of transitions for training."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )
        # Convert numpy arrays to PyTorch tensors on the fly.
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in batch.items()}
