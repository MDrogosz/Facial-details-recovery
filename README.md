# Quality recovery

**Introduction**
This repo contains my GAN implementation, where generator is predestined to recover details from face photos that lost quality due to compression.

**Data**
Training images come from <a href="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">CelebA</a> dataset. Input images undergo exertions of resizing to desired format , center cropping and normalization. In both generator and discriminator starting weights are generated from Gaussian function, with given mean and deviation.

**Model Description**
Discriminator model is straightforward CNN downscaling model taking input of shape 3x128x128, and returning 1x2 vector with argmax representing class with highest probability. It is meant to predict whether given image is original, or the one created by generator. Generator at the other hand consists of both downscaling, and then sequentially upscaling layers, additionally having two skip connection to preserve details.

Example of result achieved after 4 epochs of training: 
![image](https://user-images.githubusercontent.com/62211774/86174307-b39dce80-bb21-11ea-8026-5300bf355527.png)


Despite the model being quite simple in terms of task complexity, it still manages to perform quite well. For better results one could consider additional layer and extending dimension size of existing ones, but this solution could be computionally heavy and require additional memory.
