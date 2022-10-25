Repo to store various fast attention mechanisms: Hybrid and Performers. The performers code comes from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/enformer_pytorch.py

The Stochastic Positional Encoding comes from https://github.com/aliutkus/spe/tree/main/src/pytorch/spe. 

Adding in the FFT from https://proceedings.neurips.cc//paper/2021/file/c0f168ce8900fa56e57789e2a2f2c9d0-Paper.pdf.

Added in 2d Toeplitz matrix masking from our work https://proceedings.mlr.press/v162/choromanski22a/choromanski22a.pdf
For Graph related experiments using graph diffusion kernels (GKAT) in our work see https://github.com/arijitthegame/GKAT-Experiments/tree/main/arijit_refactor

Note that: it is straightforward to add in the Toeplitz masking into performer_vit as well. See https://github.com/arijitthegame/Fast-Attention/blob/main/performer_lucidrains.py#L454-L473

This is in active development and code is used in https://github.com/arijitthegame/enformer_performer/blob/main/enformer_refactored.py.

Future Work: Add the HRF variant (Angular Hybrid) defined in https://openreview.net/pdf?id=EMigfE6ZeS. The current code is too slow and not properly optimized for GPU/TPU. 

########################
Based on current results and results from https://arxiv.org/abs/2107.07999, should default to RPE. Currently out of scope : develop TPU friendly RPE aka optimize the current implementation, the main bottleneck being fft not optimized for TPUs.
