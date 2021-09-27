# Image2Text  
## Image Annotation 
### shellhacks 2021 project
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
## What it does ?  
I take an image as an input from the user and it understands the image very well, learn from the image and from previous learning too. finally it produces textual understanding of the image as captions.


 ## How we built it ? 
--> we have used 15000 images and respective captions to create this project
--> First we started with cleaning captions and images
--> Then we removed unnecessary words numbers from the captions to make them precise
--> We Filtered Words according to a certain threshold frequency, that is no of times a word appears in a sentence.
--> finally we got 1845 keywords from 40000 words from the dataset
-->The we used image preprocessing technique to convert image to feature vector
--> We used tokenization and word embedding techniques to process the captions
-->Then we created a Description for training data by creating Dictionaries to Map each Image to its corresponding captions
--> we created new data to make our model more accurate using data generator functions
-->Then we extracted the image feature from the image dataset and created our deep CNN Resnet model and trained it using google cloud machine learning APIs and clusters
--> Then we deployed our model using some backend scripts and HTML using Twilio APIs.

## Tech:
-->Python 
-->HTML 
-->Flask
-->Tensor_flow 
-->keras 
-->open cv
-->Google Cloud 
-->Twilo API 

## Resuts: 

![](Accuracy.JPG)
![](Accuracy.JPG)



## License: 
MIT
**Free Software, Hell Yeah!**
### Contact: 
Email: vineeth.artifintell@gmail.com 
website : vineethramesh.me 


