# ACT: Action Chunking with Transformers

Original research: https://tonyzhaozh.github.io/aloha/, https://arxiv.org/pdf/2304.13705

In Experiment 1 (Reactive Clicks), we made a simple CNN that learnt how to move the mouse and click targets that randomly appeared. 

Now in Experiment 2, I'd like to adapt the ALOHA ACT VLA from robotics into computer use. ACT takes in cameras + proprioception + action sequences. The camera becomes our computer screen, proprioception & action becomes our mouse and keyboard.

Proprioception: we should research and test out different variants
  1. Simplest: sparse vector where each keyboard key is an up or down (0/1)
  2. Sparse embeddings with pooling
  3. Key-state tokeniser: Don't aggregate at all. Feed each currently-pressed key as a separate token into the transformer encoder alongside the vision tokens and mouse state. Basically treat keys like additional observation tokens.

Vision backbone: we should also test out different variants
  1. ResNet18 (~22M): this is the one used in the original ACT paper
  2. SigLIP2 (base 86M)
  3. DINOv2 (ViT-S/14 21M)
  4. DINOv2 (ViT-B/14 86M)

Task environments (pick one):
  - Drag and drop (with labyrinth?): tests mouse trajectories and smooth motion
  - Simple form filling: mouse, keyboard
  - MiniWoB++ (https://miniwob.farama.org): choose subset, also adapt observation & action space to our model output + handle reward

Previous context: should play around with variations
  - No context (baseline)
  - Can be more input tokens
  - Observations is expensive but informationally rich
  - Action space is cheap but only what I did, not what happened
```
Run your SigLIP2 backbone at full resolution on the current frame (256 spatial tokens). For the previous N frames, don't keep the full spatial tokens — just keep the [CLS] token or a global average pool of the spatial features (1 token per past frame). So your cross-attention sees 256 + N tokens instead of 256 × (N+1). Five frames of history adds 5 tokens — negligible cost. You lose spatial detail about where things changed in past frames, but you retain a semantic summary of what the screen looked like.
```
  - Other options: differential images (subtract), recurrent memory

Reminder that we need to aim for a minimum of 30hz frequency on our M1 8GB RAM (mps/mlx capable), so we need to be wary and apply any optimisations as needed.