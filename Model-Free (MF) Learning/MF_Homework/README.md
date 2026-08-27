
              
  HOMEWORK
                   
       
   MEMBER 1 -  Algorithm
   -  SARSA(λ)
   -  E(s,a)
   -  ε-greedy

Understand the provided TD(\(\lambda\))
Modify it to SARSA(\(\lambda\))
Implement \(E(s,a)\)
Implement ε-greedy
Test all 3 λ values
Make sure the algorithm produces correct returns
Document the algorithm
Extra: Member 1 should also help debug the other members' code.


   MEMBER 2- Heatmaps + Animation
   -200 frames
   -3 heatmaps
   -video
Use the finalized SARSA implementation
Perform the single run
Run all three λ values
Calculate:
$$ V(s)=\max_a Q(s,a) $$
Generate 200 frames
Put the three heatmaps side-by-side
Label episodes and λ values
Convert frames into animation/video
Check that the animation is correct
This is quite a bit of work because there are 200 images to generate and process.

   
   
   MEMBER 3 - Statistics
   - 100 runs
   - mean return
   - variance
Use the finalized SARSA implementation
Run:
$$ 100\text{ runs}\times3\lambda $$
Record the return for all 200 episodes
Calculate mean:
$$ \mu $$
Calculate standard deviation/variance
Generate the combined plot
Add error bars or shaded variance
Save the numerical results
Check that the results are reproducible




FINAL REVIEW
    - ALL 3 MEMBERS
              






This is computationally heavier than it initially looks.
