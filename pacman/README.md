# PacmanDQN
Deep Reinforcement Learning in Pac-man

## Example usage

Run a model on `smallGrid` layout for 10000 episodes, of which 9000 episodes
are used for training.

```
$ python3 pacman.py -p PacmanDQN -n 10000 -x 9000 -l smallGrid
```

OR

You can directly run qrsh jobs in format jobs_*.qrsh

### Layouts
Different layouts can be found and created in the `layouts` directory


## Requirements

- `python==3.5.1`
- `tensorflow==0.8rc`

## Acknoledgemenets

DQN Framework by  (made for ATARI / Arcade Learning Environment)
* [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/mrkulk/deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow))

Pac-man implementation by UC Berkeley:
* [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html) ([http://ai.berkeley.edu/project_overview.html](http://ai.berkeley.edu/project_overview.html))

Deep Reinforcement Learning in pac-man
* [Deep RL pacman] (https://github.com/tychovdo/PacmanDQN)
