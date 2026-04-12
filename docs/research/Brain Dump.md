Using VLAs for computer use 
If you apply first principles thinking, essentially using a computer is just a series of actions that are taken with the keyboard and mouse and the input that you get is through the screen. There's sound but we can ignore sound for now so the question is why can't we have a vision-language-action (VLA) model that is able to run at a super high frequency that matches or exceeds human level? That could be 30hz to 60hz to 120hz. I feel like if the input is the computer's screen and then the output, instead of current computer use models which go like "You output a series of actions like click or type or drag or double click", you break it down even further into first principles.

Instead of "click" it's:
- left click mouse down
- left click mouse up


You also drag instead of scroll and then you give the number of pixels as a tool call. Instead of that, half, perhaps because it's running at a control frequency of, say, 60 Hz, you can go like "scroll as a single action", like a single mouse wheel turn. When the mouse wheel clicks a little bit as it turns, every Hz you do that.

Instead of "drag", where you put in a starting pixel position and an ending pixel position, it's sort of like how VLA's are used for robots, like joint positions and joint targets. Instead of thinking in terms of, say, pixels, maybe think in terms of how a virtual being might operate a mechanical physical mouse. Except this mouse will have some benefits: you can treat the track pad as an infinite dimension. You can also treat it if your human fingers have to kind of scroll a bit and then you have to re-scroll and then re-scroll if you're scrolling down a lot. The scroll wheel you can turn as fast as you can, as much as you can, et cetera.

Likewise for the keyboard, instead of "type" and then you put in the whole string, you break it down into first principles of:
- key press down
- key press up


Essentially that's what, at the basic level, a keyboard does: key press down, key press up. That can also encode actions like long press and hold. You can also encode like pressing multiple keys etc. 

Now world models is something else to consider. There's, I think, a little bit of research already on computer use of world models. That might be like a next level up, combining VLAs and world models, predicting the next UI state or the next computer state. The encoding, the representation of the next computer state, is what world models are based on so perhaps that could be the next.

I think, in terms of the high-frequency VLA models, it would be really good if I can maybe get some sort of small experiment going, like a very toned-down experiment of a high-frequency VLA being used for computer use. Maybe it's like a really simple application or not even an actual computer, literally just a simple UI that has a single button, for example, really kindergarten. Maybe that, and then demonstrate that it can operate at high frequency or to demonstrate the high frequency, maybe like one of those reaction tests. Make one of those reaction tests and then have this VLA computer using a VLA model be able to perform in it.

It'd be really good if it can run on consumer hardware and I'm not even talking about professional consumer hardware like an NVIDIA RTX 5090. I only have a MacBook Pro with 8 GB of RAM M1, which is like not amazing. If I can maybe start out with a really, really small, really kindergarten-proof-of-concept to validate that it works and then build up from there. This is essentially what Tesla's digital optimus project is all about and what Elon Musk is trying to do in collaboration with XAI and Tesla, or that macrohard project. Obviously this is gonna be hella complex so I can start out small and test it out as a little bit of a side research side project. Maybe I could test it out like that. 
