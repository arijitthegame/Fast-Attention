Repo to store various fast attention mechanisms: Hybrid and Performers. The performers code comes from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/enformer_pytorch.py

The Stochastic Positional Encoding comes from https://github.com/aliutkus/spe/tree/main/src/pytorch/spe. 

Adding in the FFT from https://proceedings.neurips.cc//paper/2021/file/c0f168ce8900fa56e57789e2a2f2c9d0-Paper.pdf.

#TODO: Add the HRF variant (Angular Hybrid) defined in https://openreview.net/pdf?id=EMigfE6ZeS. The current code is too slow and not properly optimized for GPU/TPU. 

For maximum confusion, there will be pytorch as well as tensorflow/jax code thrown in here to be used for various projects with different collaborators. 
