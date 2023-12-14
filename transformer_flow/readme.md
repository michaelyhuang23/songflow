### Dec 13 8:55 pm
So, I think we can conclude that traditional flow fails
There are two solutions moving forward. First I realize that it might not be a good idea to do autoregressive flow over time since the frequency data at each time step are now minimally related. It might be a better idea to do autoregressive flow over frequency. This is a easy fix. 

Second, facebook has a VQ-VAE encoder for audio that is accessible through the huggingface. It seems that the VQ-VAE deals with audio data directly instead of spectrogram. I should explore it. 

Third, we should use a larger dataset which I'm generating right now. 

Actually, approach 2's tokenizer discretizes audio into discrete categorical tokens that is good for LLM. However, I have realized that the vocoder model like WaveFlow are actually capable of generating audio from low quality spectrogram.