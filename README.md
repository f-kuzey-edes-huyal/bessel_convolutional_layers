# Bessel Convolutional Layers
  
This repository contains the Pytorch version of the Bessel Convolutional Neural Nets. [Delchevalerie et al.](https://proceedings.neurips.cc/paper/2021/hash/f18224a1adfb7b3dbff668c9b655a35a-Abstract.html) published it in 2021. The Tensorflow version was included with their paper.

Convolutional neural networks of this type employ Bessel coefficients to represent images, with the goal of achieving an orientation-independent design. This is important when analyzing satellite images or clinical data. 

When utilizing this architecture, **keep in mind these three points**: 

1. You must insert ***retain_graph=True*** inside ***loss.backward()***  since this design has both real and complex weights (***loss.backward(retain_graph=True)***).
2. According to the paper, ***Tanh*** activation performs better with this architecture.  I had issues with ***ReLU** activation during classification, so I had to switch it out for a ***Tanh*** one.
3. The maximum number of Bessel coeffients is indicated by ***j_max*** and ***m_max***. The number of coeffients in the accurate mathematical representation of the image will be unlimited. However, we must choose a limited number for them. A higher value for ***m_max*** and ***j_max*** indicates more parameters.

 
