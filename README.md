# transferGAN
This is a GAN-system for picture to picture translation, where pixel values in images correspond particle momentum before entering material and after exiting the material.

The data is sparse with only few pixels activated in an 256x256 grid both in before and after images and this will be reflected in the default options, but the codebase should be flexible enough for applying this for other purposes. Another design goal is a modular structure which allows extending this work with newer designs for networks, loss-functions, visualization, and logging.

# Code ownership
This work is done for [Muon-Solutions Oy](http://muon-solutions.com/). The company owns all rights to the code and has agreed to publish it under GPL-3 license.

# Thanks
As the codebase grows it will be easy to recognize familirities with the official pytorch implementation of [pix2pix and cycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which is at minimum an inspiration for this code.
After some work I realized that [BicycleGAN](https://github.com/junyanz/BicycleGAN) would have been the best starting point, which has the multimodality implemented.
The HDF5 loader is based on this posting in [Pytorch forum](https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU)
