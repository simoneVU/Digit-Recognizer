# Digit-Recognizer
Digit recognizer from Scratch with Numpy and Pandas

### Data Preprocessing
The MNIST dataset is used for training and testing the Neural Network. More information regarding the dataset can be found [here](http://yann.lecun.com/exdb/mnist/).
The data preprocessing step consists of two steps: loading and reading the encoded binary files and generating the Neural Network input. The binary files are read with the 
`gzip` python library and decoded based on the guide on the [MNIST website](http://yann.lecun.com/exdb/mnist/). Subsequently, the pixels data for each image are extracted as saved in a numpy array transposed to have 
each one image pixels data per column (as shown in the image below).

![Digit_Recognizer_overview drawio](https://user-images.githubusercontent.com/60779914/158414671-697d35ff-4e3e-4fce-914f-610df7b7460d.png)
