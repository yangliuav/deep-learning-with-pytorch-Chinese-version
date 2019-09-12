# 深度学习 (PyTorch)

本教程是翻译自 Udacity 的深度学习教程(https://www.udacity.com/course/deep-learning-nanodegree--nd101)。这篇教程包含了大多数深度学习的热门课题。同时，在大多数的情况下，这个项目也能带领你实现，诸如CNN, RNN 和 GAN 网络，当然这也包含了权重初始化和batch正则化等等问题。

## 准备工作（在 google colab或者 microsoft notebook 运行的你的文件，免去没钱买显卡之苦）
### Google colab
1. 找到你先运行的文件，例如，XX.
2, 将`https://github.com/`替换成 `https://colab.research.google.com/github/`，或者使用插件[Chrome extension](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo)来完成替换。
3，登录自己的google账号。
4，点击`COPY TO DRIVE`，重命名文件。
<img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/copy_to_drive.png">
5，这时就可以在colab运行你的程序了。

### Microsoft notebook
1，登录 `https://notebooks.azure.com`，登录你的微软账号。
2，新建项目，Project，

<img src="https://docs.microsoft.com/en-us/azure/notebooks/media/quickstarts/upload-github-repo-command.png">

3，填入Github地址，导入整个项目。

<img src="https://docs.microsoft.com/en-us/azure/notebooks/media/quickstarts/upload-github-repo-popup.png">

4，点入相应的文件，运行run即可

### 国内用户
如果您的区域，以上的两个网站都不能用，我建议你试一下， [Jupyter Binder](https://mybinder.org/v2/gh/GokuMohandas/practicalAI/master) 和 [Kesci](https://www.kesci.com/home/column/5c20e4c5916b6200104eea63)。

## 内容简介

### 简介

### 神经网络

* [神经网络简介](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-neural-networks): Learn how to implement gradient descent and apply it to predicting patterns in student admissions data.
* [使用Numpy进行数据分析](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-analysis-network): [Andrew Trask](http://iamtrask.github.io/) leads you through building a sentiment analysis model, predicting if some text is positive or negative.
* [PyTorch简介](https://github.com/yangliuav/deep-learning-with-pytorch-Chinese-version/tree/master/intro-to-pytorch): Learn how to build neural networks in PyTorch and use pre-trained networks for state-of-the-art image classifiers.

### 卷积神经网络

* [Convolutional Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/convolutional-neural-networks): Visualize the output of layers that make up a CNN. Learn how to define and train a CNN for classifying [MNIST data](https://en.wikipedia.org/wiki/MNIST_database), a handwritten digit database that is notorious in the fields of machine and deep learning. Also, define and train a CNN for classifying images in the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
* [Transfer Learning](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/transfer-learning). In practice, most people don't train their own networks on huge datasets; they use **pre-trained** networks such as VGGnet. Here you'll use VGGnet to help classify images of flowers without training an end-to-end network from scratch.
* [Weight Initialization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/weight-initialization): Explore how initializing network weights affects performance.
* [Autoencoders](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/autoencoder): Build models for image compression and de-noising, using feedforward and convolutional networks in PyTorch.
* [Style Transfer](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/style-transfer): Extract style and content features from images, using a pre-trained network. Implement style transfer according to the paper, [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Gatys et. al. Define appropriate losses for iteratively creating a target, style-transferred image of your own design!

### Recurrent Neural Networks

* [Intro to Recurrent Networks (Time series & Character-level RNN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text; learn how to implement these in PyTorch for a variety of tasks.
* [Embeddings (Word2Vec)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/word2vec-embeddings): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.
* [Sentiment Analysis RNN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn): Implement a recurrent neural network that can predict if the text of a moview review is positive or negative.
* [Attention](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/attention): Implement attention and apply it to annotation vectors.

### Generative Adversarial Networks

* [Generative Adversarial Network on MNIST](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist): Train a simple generative adversarial network on the MNIST dataset.
* [Batch Normalization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/batch-norm): Learn how to improve training rates and network stability with batch normalizations.
* [Deep Convolutional GAN (DCGAN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/dcgan-svhn): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.
* [CycleGAN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/cycle-gan): Implement a CycleGAN that is designed to learn from unpaired and unlabeled data; use trained generators to transform images from summer to winter and vice versa.

### Deploying a Model (with AWS SageMaker)

* [All exercise and project notebooks](https://github.com/udacity/sagemaker-deployment) for the lessons on model deployment can be found in the linked, Github repo. Learn to deploy pre-trained models using AWS SageMaker.

### Projects

* [Predicting Bike-Sharing Patterns](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing): Implement a neural network in NumPy to predict bike rentals.
* [Dog Breed Classifier](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification): Build a convolutional neural network with PyTorch to classify any image (even an image of a face) as a specific dog breed.
* [TV Script Generation](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-tv-script-generation): Train a recurrent neural network to generate scripts in the style of dialogue from Seinfeld.
* [Face Generation](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-face-generation): Use a DCGAN on the CelebA dataset to generate images of new and realistic human faces.

### Elective Material

* [Intro to TensorFlow](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/tensorflow/intro-to-tensorflow): Starting building neural networks with TensorFlow.
* [Keras](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/keras): Learn to build neural networks and convolutional neural networks with Keras.

---

## Todo list

