# Aug 28, 2022

### The Problem
Previously, the problem was when the the `disc_loss.backward()` function was called, after the first step, 
an error was thrown in the generator's module, specifically in the `Sampler`. When I switched the Pancho's 
Generator, another type of error was thrown. 
This time, it turns out that his generator spits out a shape of (batch, 18, 256, 256). The expected shape is (batch, 18, 1, 256, 256). The channel
dimension was dropped.

### Conditioning Stack
* AlConditioningStack outputs the correct dimensions per the paper.
* ConditioningStack doesn't.
* Therefore, we'll use AlConditioningStack.

### Latent Conditioning Stack
* alLCS seems to take the correct input (8x8x8 draws from a normal distn) 
and generate the correct output (1x768x8x8).
* LatentConditioningStack seems to ignore input shape, 
but it still generates the correct output (1x768x8x8). **Should investigate this.**


### Sampler/Output
* Revised AlGenerator to move the end code of forward into an `AlSampler` class.
* will need to investigate the AlSampler class next. 
* Sampler is a class and seems to generate the correct sizes.



