# buaa-ai-gan
北航人工智能大作业-GAN实现人脸性别转换

# 阅读论文
- 《Generative Adversarial Nets》 
  
了解GAN基本架构，实现最基本的GAN生成手写数字，实现：[GAN-MNIST](.GAN-MNIST/)

- 《Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks》
   
了解DCGAN的基本架构，实现DCGAN生成人脸，实现：[DCGAN](.DCGAN/)
 
- 《A Style-Based Generator Architecture for Generative Adversarial Networks》


- 《Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation》

# 实现思路

1. 完成一个基础的MNIST对抗网络模型，使其可以通过一组随机向量来生成一张随机的手写数字图像。 可借助基础的DCGAN结构，训练出一组用于MNIST数据集的生成器、判别器模型。

目的：掌握GAN的基础玩法。

2. 搭建出一个`图片->Z向量`的模型。 使其可以输入一张确定的图片，根据该图片来产出一个Z向量，用该Z向量传入根据1中完成的生成模型，使其可以生成出和刚刚输入的确定图片一样的风格。

目的：找出`隐藏因子`得到出一个可自行控制的生成模型。

3. 输入一张图片，使其可以生成该图片相关风格的数据。

目的：尝试生成出自己喜欢的数字，为后续StyleGAN的人像编辑做准备。

4. 用PSP找出照片的`隐藏因子`，用StyleGAN生成性别转换后的图片

目的：实现课程目标

5. 调整`隐藏因子`，尝试特征融合

目的：学习并尝试StyleGAN的更多能力