# Is-That-G.W.Bush

This simple Torch7 neural network, is designed to determine if an image is a picture of George W. Bush's face.

# Data

I am using a subset of the "Labeled Faces in the Wild" dataset. Found here: http://vis-www.cs.umass.edu/lfw/

The two subsets of the data I downloaded were "people with names starting with A" and "George_W_Bush".

# The Neural Network

The topology of the network is not finalized at this point.
To train the network, I use the first letter in the filename of each image. If it is a "G" I expect the output [1, 0], and if its an "A" I expect the output [0, 1].
In other words, this network is meant to classify images as "GWB" or "Not GWB"

