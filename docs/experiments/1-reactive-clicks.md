# Experiment 1: Reactive Clicks

I'd like to validate the initial premise of the project by devising a very small and rudimentary experiment in a click reaction time test.

There would be a small screen, where a red circle will appear after a random interval of time, at a random spot on the screen. If you click anywhere on the red circle, it disappears. The reaction time is measured as the time between when the red circle appeared to when it got clicked. The test progresses indefinitely as a new red circle will appear after the previous one got clicked. We track the individual and average reaction times.

So there would need to be some sort of way to have image input, real-time image input of the screen, and then the outputs would be the mouse movements and then mouse click. It'll be click and also release. For training, we can use a scripted dataset where you have random intervals of time. It also clicks at random places within the red circle to introduce some variability. It would generally tend to be the fastest way, like the straight line from the existing mouse position to the target mouse position, but operating at some form of reasonable human level, like fast human-level movement. We want to mimic that: what would a human expert or fast human be able to achieve?

Results:
```
============================================================                                       
  CNN (TinyCNN)                                                                                    
============================================================                                       
  Episodes:          200                                                                           
  Hit rate:          193/200 (96.5%)                                                               
                                                                                                    
  Total steps:       mean=8.8  p50=8  p95=17                                                       
  Onset latency:     mean=0.0 steps (0ms)                                                          
  Movement time:     mean=8.8 steps (294ms)                                                        
  Total RT (steps):  mean=8.8 (295ms)                                                              
                                                                                                    
  Loop latency:      p50=1.21ms  p95=1.76ms  p99=2.67ms                                            
  Effective Hz (p95): 569 Hz   
```