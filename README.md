# Is-That-G.W.Bush

This simple [Torch7](http://torch.ch/) deep neural network, is designed to determine if an image is a picture of George W. Bush's face.

## Data

I am using a subset of the "Labeled Faces in the Wild" dataset. Found here: http://vis-www.cs.umass.edu/lfw/

The two subsets of the data I downloaded were "people with names starting with A" and "George_W_Bush".

## Data Preprocessing

In the `image_pre_proccess.lua` file, I prepare each input image so it is compatable with the network. This mainly consists of resizing the image to the correct number of pixels.

## The Neural Network

The topology of the network is based on [this well known convolutional neural network](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

To train the network, I use the first letter in the filename of each image. If it is a "G" I expect the output [1, 0], and if its an "A" I expect the output [0, 1].
In other words, this network is meant to classify images as "GWB" or "Not GWB".

