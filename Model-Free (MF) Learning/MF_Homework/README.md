
              
              HOMEWORK
                   │
       ┌───────────┼───────────┐
       
       ↓           ↓           ↓
       
   MEMBER 1     MEMBER 2     MEMBER 3
   Algorithm    Animation    Statistics
   
       │           │           │
       
   SARSA(λ)      200 frames   100 runs
   E(s,a)        3 heatmaps   mean return
   ε-greedy      video        variance
   
       │           │           │
       
       └───────────┼───────────┘
       
                   ↓
                   
             FINAL REVIEW
              ALL 3 MEMBERS
              


Member 1 — Algorithm

Understand the provided TD(\(\lambda\))
Modify it to SARSA(\(\lambda\))
Implement \(E(s,a)\)
Implement ε-greedy
Test all 3 λ values
Make sure the algorithm produces correct returns
Document the algorithm
Extra: Member 1 should also help debug the other members' code.

Member 2 — Heatmaps + Animation

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

Member 3 — 100-run experiment

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

This is computationally heavier than it initially looks.
