# Aug 28, 2022

### The Problem
Previously, the problem was when the the `disc_loss.backward()` function was called, after the first step, 
an error was thrown in the generator's module, specifically in the `Sampler`. When I switched the Pancho's 
Generator, another type of error was thrown. 
This time, it turns out that his generator spits out a shape of (batch, 18, 256, 256). The expected shape is (batch, 18, 1, 256, 256). The channel
dimension was dropped.

### Conditioning Stack
* The paper specifies the CS to output
  * [ (batch_sz, 48, 64, 64)
        * (batch_sz, 96, 32, 32),
        * (batch_sz, 192, 16, 16),
        * (batch_sz, 394, 8, 8) ]
* AlConditioningStack outputs the correct dimensions per the paper.
* ConditioningStack also does assuming it is initialized correctly. 

### Latent Conditioning Stack
* alLCS seems to take the correct input (8x8x8 draws from a normal distn) 
and generate the correct output (1x768x8x8).
* LatentConditioningStack seems to ignore input shape, 
but it still generates the correct output (1x768x8x8). **Should investigate this.**


### Sampler/Output
* Revised AlGenerator to move the end code of forward into an `AlSampler` class.
* will need to investigate the AlSampler class next.
* AlSampler takes as input:
  * the reversed output of the conditioning stack 
      * [ (batch_sz, 384, 8, 8) 
      * (batch_sz, 192, 16, 16), 
      * (batch_sz, 96, 32, 32), 
      * (batch_sz, 48, 64, 64) ]
  * and the output of the latent conditioning class, a tensor of shape (1x768x768x768)

* Sampler is now class and seems to generate the correct sizes. 



# Aug 31, 2022
## The Problem
We've more or less narrowed down the problem to the Sampler. AL's code doesn't run because its
sampler outputs now outputs batchx22x1x256x256 but was originally expected to ouput
batchx22x256x256.
RVC code base crashes in the sampler when doing backward() for the 2nd step.

### Next step:
study both samplers and compare and contrast.


