# Pacman Game Q-learning Agent

## What is it about?

Script contains a reinforcement learning algorithm (Q-learning) that controls Pacman player in the classic game, by deciding its next move and helping it win. Learner wins 9/10 games on average, after being trained on 2000 episodes.

Script uses as starting point [UC Berkeley CS188 Intro to AI](http://ai.berkeley.edu/reinforcement.html).
**Note: this is not a solution for the coursework in CS188.**

Code is written in Python 2.7

## How does it work?

1. Download the reinforcement package of Pacman game from [here](http://ai.berkeley.edu/projects/release/reinforcement/v1/001/reinforcement.zip), or alternatively [here](http://ai.berkeley.edu/reinforcement.html).

2. Unzip the package and place mlLearningAgents.py inside the directory.

3. Run the following command line in the directory (performs 2000 runs of training and 10 runs of playing in a small grid):
  
          python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
