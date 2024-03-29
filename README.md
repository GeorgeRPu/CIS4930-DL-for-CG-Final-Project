# Classifying Images via Reinforcement Learning

Project part 3 for CIS4930 Deep Learning for Computer Graphics (2020 Fall).

## Dependencies

Dependencies can be installed using

```bash
pip install -r requirements.txt
```

preferrably in a new virtualenv or conda environment.
Note that you must use pip as the PyTorch package is listed as pytorch in conda.

The file `dqn.py` contains an implementation of the DQN algorithm based partially on the [PyTorch DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
The file `environments.py` contains a classification task environment, which subclasses `gym.Env`, where the agent earns reward for correctly labeling an image from a dataset.

## Running Instructions

[Hydra](https://hydra.cc/) is used to manage hyperparameters and manage experiment artifacts (figures, logs, saved models).
Hyperparameters are defined in `config.yaml` and can be overridden at runtime using a key=value syntax, e.g

```bash
python train.py episodes=10000
```

sets the episodes hyperparameter to 10000.
More information on individual hyperparameters is included in `config.yaml` in the form of comments.

To run the DQN algorithm on CartPole-v0, use

```bash
python train.py
```

Override `render=True` to view the agent train on CartPole-v0.
To run the DQN algorithm on the classification environment, use

```bash
python train.py env=MNIST
```

To test a trained model on an environment, use

```bash
python test.py <path to params>
```

It has several command line arguments. Use the flag `--help` to get more
information. A supervised learning baseline can be run using

```bash
python baseline.py
```
