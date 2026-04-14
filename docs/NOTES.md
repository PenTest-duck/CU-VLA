=== HUMAN NOTES - IF YOU ARE AN AI AGENT, READ ONLY, DO NOT WRITE ===

Representing mouse actions/movements in action space
  - We can make the assumption that AT MOST 1 of left click (L), right click (R), and scroll (D/U) will be taken simultaneously
    - SD/SU is one 'tick' of the scroll wheel per frame (e.g. 30 hz), or should it be velocity?
  - Mouse actions may be paired with mouse movement simultaneously
  - Minimally, mouse actions can be 0, L, R, SD, SU; classification problem
  - Minimally, mouse movement is a 2D velocity (Dx, Dy); either regression or classification (binning)

Representing keyboard actions in action space
  - One-hot encoding of all keys (i.e. allow simultaneous key presses for all keys)
  - 26 alphabet + 10 numbers + 11 symbols + 4 arrows + Space/Return/Delete/LShift/Rshift/Tab/CapsLock/Escape/Ctrl/LCmd/RCmd/LOpt/ROpt
  - Optional: Fn + 12 function keys
  - Excluded: power

Inputs:
  - MacBook Pro M1 has 1440x900 resolution with 2560x1600 pixels
    - Probably could be lower .. figure out minimal resolution specs for a fully functional computer
  - WxHx3 dimension per frame
  - Excluded: audio (speakers & mics)

System 0/1/2 thinking (inspired by Figure AI's Helix 02):
  - System 0: lowest-level primitive
  - System 1: motor control actions/trajectories, e.g. "click x", "type x", "highlight x", "scroll to x"
  - System 2: reasoning, task planning
  - But also this whole breakdown itself of the system levels encodes an assumption

How to have continuous memory window over past states?
  - Maybe only store System 1/2?
  - How to compress? Esp. if we are doing high-frequency

Take inspiration from other uses like gaming, e.g. JARVIS-VLA

Frequency domain

FAST tokeniser

Typing test (TypeRacer/MonkeyType/10FastFingers style?)
  - High-frequency key press primitives
  - Tests simultaneous key presses (e.g. RShift+A)
  - Should we differentiate LShift vs RShift, to mimic human typing?
  - Order of pressing: RShift->Rshift+A (2 frames) or Rshift+A/A+RShift (single frame?)

Don't clip cursor delta_x, delta_y at 50px/frame?
Normalise cursor movements instead of pixels?
  - Log-space deltas vs discrete exponential binning (like FDM1)

osu!
