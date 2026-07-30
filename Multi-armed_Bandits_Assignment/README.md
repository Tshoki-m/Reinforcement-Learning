 PLAN

 RL_Assignment/

├── bandit.py

├── epsilon_greedy.py

├── optimistic_greedy.py

├── ucb.py

├── experiments.py (plots included)

├── requirements.txt

└── README.md
 
 
 Member 1 : 
 
 ε-Greedy

epsilon_greedy.py Needs to contain:

class EpsilonGreedyAgent:
select_action()
update()

Responsible for:
Implementing the ε-greedy algorithm.
Using ε = 0.1 for the required plot.
Running experiments with several ε values (e.g., 0, 0.01, 0.1, 0.2) for the comparison plot.
Producing the reward data for their algorithm.

Deliverables:
ε-greedy implementation.
Reward data for all ε values.
Brief explanation of how ε affects exploration.



Member 2 

Greedy with Optimistic Initialization

optimistic_greedy.py needs to contain

class OptimisticGreedyAgent:
select_action()
update()

Responsible for:
Implementing the optimistic greedy algorithm.
Using Q
1
=5 for the required plot.
Running experiments with different initial values (e.g., 0, 2, 5, 10).
Producing the reward data.

Deliverables:

Optimistic greedy implementation.
Reward data for different Q
1
.
Brief explanation of optimistic initialization.


Member 3 

ucb.py needs to contain
class UCBAgent:
select_action()
update()


Responsible for:
Implementing the UCB algorithm.
Using c=2 for the required plot.
Running experiments with different c values (e.g., 0.5, 1, 2, 5).

Deliverables:
UCB implementation.
Reward data for different c.



Shared Components:

Multi-Armed Bandit Environment:
One person can write it , but everyone uses the exact same file:

bandit.py
contains:
Generate the 10 arms.
Generate the true means.
pull(action) method.


experiments.py
Final graphs.


Merge all code into one project.
