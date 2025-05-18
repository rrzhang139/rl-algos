import numpy as np
import matplotlib.pyplot as plt

# ----- Simple 5x5 Gridworld ----- #


class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        # start at (0,0)
        self.pos = (0, 0)
        return self._state()

    def _state(self):
        # flatten position to state index
        return self.pos[0] * self.size + self.pos[1]

    def step(self, action):
        # 0: up, 1: right, 2: down, 3: left
        r, c = self.pos
        if action == 0 and r > 0:
            r -= 1
        elif action == 1 and c < self.size - 1:
            c += 1
        elif action == 2 and r < self.size - 1:
            r += 1
        elif action == 3 and c > 0:
            c -= 1
        self.pos = (r, c)
        reward = -1
        done = self.pos == self.goal
        if done:
            reward = 0
        return self._state(), reward, done

# ----- Dyna‑Q ----- #


def dyna_q(episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=20):
    env = GridWorld()
    n_states = env.size * env.size
    n_actions = 4
    Q = np.zeros((n_states, n_actions))

    # Model: dictionary mapping (s,a) -> (r, s')
    model_r = {}
    model_s = {}

    steps_per_episode = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        steps = 0

        while not done:
            # ε‑greedy action
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = np.argmax(Q[s])

            s_next, r, done = env.step(a)

            # Real Q‑learning update
            td_target = r + gamma * (0 if done else np.max(Q[s_next]))
            Q[s, a] += alpha * (td_target - Q[s, a])

            # Update model
            model_r[(s, a)] = r
            model_s[(s, a)] = s_next

            # Planning with simulated transitions
            for _ in range(planning_steps):
                # randomly pick previously seen state‑action
                (sp, ap) = list(model_r.keys())[
                    np.random.randint(len(model_r))]
                rp = model_r[(sp, ap)]
                sp_next = model_s[(sp, ap)]
                td_target_p = rp + gamma * np.max(Q[sp_next])
                Q[sp, ap] += alpha * (td_target_p - Q[sp, ap])

            s = s_next
            steps += 1

        steps_per_episode.append(steps)

        # simple log every 50 episodes
        if (ep + 1) % 50 == 0:
            avg_steps = np.mean(steps_per_episode[-50:])
            print(
                f"Episode {ep+1:4d}: avg steps (last 50 eps) = {avg_steps:.2f}")

    return steps_per_episode


steps_data = dyna_q()

# Plot learning curve
plt.figure()
plt.plot(np.convolve(steps_data, np.ones(25)/25, mode='valid'))
plt.xlabel('Episode')
plt.ylabel('Steps to reach goal (25‑ep MA)')
plt.title('Dyna‑Q Learning Curve')
plt.show()
