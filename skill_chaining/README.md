# Skill Chaining Example

This directory contains a tiny demonstration of policy sequencing on the `PegInsertionSide-v1` environment from [ManiSkill](https://github.com/haosulab/ManiSkill). The idea follows "Policy Sequencing" which trains a policy for each subtask and then fine-tunes the subsequent policy to start from the terminal states of the previous one.

The provided script trains two simple policies sequentially. The second policy loads the weights of the first and continues training, illustrating how skills can be chained together.

Run training with:

```bash
python -m skill_chaining.train_skill_chaining
```

Make sure ManiSkill is installed:

```bash
pip install maniskill
```
