# Mini TD-MPC2

This directory contains a toy implementation of the TD-MPC2 algorithm. It is
simplified for educational purposes and trains on the `PickCube-v1` task from
the [ManiSkill](https://github.com/haosulab/ManiSkill) benchmark.

Run training with:

```bash
python -m tdmpc2.train
```

The implementation defines minimal models and a small training loop so that the
core ideas are easy to follow.

Note that ManiSkill must be installed for the environment to work:

```bash
pip install maniskill
```
