## MCTS Tic-Tac-Toe AI
#### Sean Osier


#### Summary
In Early 2016, wanting to learn more about Monte Carlo Tree Search (MCTS), a major part of the original [AlphaGo algorithm / paper](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) that had just came out, I ported a [Python 2 implementation I found online](http://mcts.ai/code/python.html) over to Python 3 (`mcts.py`).

I also created my own implementation (`mcts_gt.py`) to try to improve upon some of the draw backs I observed in the original.

For both of these cases, I used Tic-Tac-Toe as my simple, test case game.

Finally, I also explored (`ML_Tic-Tac-Toe_AI_Experiments.ipynb`) building a Tic-Tac-Toe AI using a machine learning model trained to pick moves, basically the same idea as the "policy network" from the AlphaGo paper.

#### How to Run

**NOTE: Requires Python 3**

Running `python mcts.py` and `python mcts_gt.py` will both play a sample game between two different MCTS AIs. Otherwise you can just import the desired classes and functions into your own code as desired.

`jupyter notebook ML_Tic-Tac-Toe_AI_Experiments.ipynb` will allow you explore some of the machine learning work as well as the AI classes for that setup. (Note, this code is still somewhat rough and not entirely cleaned up. Use with caution.)
