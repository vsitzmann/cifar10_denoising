# cifar10_denoising
A tiny example application of pytorch_prototyping, with an example training script. The application is denoising of CIFAR10-images using a residual U-Net architecture. While the structure of this project is slightly overkill for the problem it's trying to solve, it is intended to serve as starter code for research projects. 

The training script and "DenoisingUnet" class handle:
1. checkpointing
2. Logging with tensorboardx
3. Writing evaluation results to disk
