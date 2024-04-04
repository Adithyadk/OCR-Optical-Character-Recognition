# OCR (Optical Character Recognition) System

This project implements an Optical Character Recognition (OCR) system using Convolutional Neural Networks (CNNs) with TensorFlow/Keras. The OCR model is trained to recognize alphabets from images. The convolutional architecture is designed to efficiently learn and classify alphabetic characters with high accuracy.

## Introduction

OCR systems have a wide range of applications, including document digitization, text extraction from images, and handwriting recognition. This project focuses specifically on recognizing alphabets, providing a foundation that can be extended to more complex character recognition tasks.

## Convolutional Neural Networks (CNN) — Architecture Explained
![Alt text](https://miro.medium.com/v2/resize:fit:750/format:webp/0*vb72NzJrSMxQZ7j9)


### Introduction
A convolutional neural network (CNN), is a network architecture for deep learning which learns directly from data. CNNs are particularly useful for finding patterns in images to recognize objects. They can also be quite effective for classifying non-image data such as audio, time series, and signal data.



![image](https://github.com/Adithyadk/OCR-Optical-Character-Recognition/assets/114129263/5216ec35-2fe4-4528-9e4a-c2a3fb0f1913)

### Kernel or Filter or Feature Detectors
In a convolutional neural network, the kernel is nothing but a filter that is used to extract the features from the images.

## Formula = [i-k]+1

i -> Size of input , K-> Size of kernel

![image](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*n7GQU2sA0Y8P72Yf.png)

### Stride
Stride is a parameter of the neural network’s filter that modifies the amount of movement over the image or video. we had stride 1 so it will take one by one. If we give stride 2 then it will take value by skipping the next 2 pixels.

## Formula =[i-k/s]+1

i -> Size of input , K-> Size of kernel, S-> Stride

![image](https://miro.medium.com/v2/resize:fit:828/format:webp/1*E52voKt51l-PRD-fKtWMSg.png)

### Padding
Padding is a term relevant to convolutional neural networks as it refers to the number of pixels added to an image when it is being processed by the kernel of a CNN. For example, if the padding in a CNN is set to zero, then every pixel value that is added will be of value zero. When we use the filter or Kernel to scan the image, the size of the image will go smaller. We have to avoid that because we wanna preserve the original size of the image to extract some low-level features. Therefore, we will add some extra pixels outside the image. Kindly use this link to learn more about padding.(https://medium.com/@draj0718/zero-padding-in-convolutional-neural-networks-bf1410438e99)

## Formula =[i-k+2p/s]+1

i -> Size of input , K-> Size of kernel, S-> Stride, p->Padding

![image](https://miro.medium.com/v2/resize:fit:828/format:webp/0*voP_09Y6cK97GsHq.png)

### Pooling
Pooling in convolutional neural networks is a technique for generalizing features extracted by convolutional filters and helping the network recognize features independent of their location in the image.

![image](https://github.com/Adithyadk/OCR-Optical-Character-Recognition/assets/114129263/2c14d060-9ab5-49d2-87be-0b6e7abb4663)

### Flatten
Flattening is used to convert all the resultant 2-Dimensional arrays from pooled feature maps into a single long continuous linear vector. The flattened matrix is fed as input to the fully connected layer to classify the image.

![image](https://miro.medium.com/v2/resize:fit:828/format:webp/1*xz9SQ3YzE2LRaR-_qfbZ6w.png)

Layers used to build CNN
Convolutional neural networks are distinguished from other neural networks by their superior performance with image, speech, or audio signal inputs. They have three main types of layers, which are:

* Convolutional layer
* Pooling layer
* Fully-connected (FC) layer

### Convolutional layer

This layer is the first layer that is used to extract the various features from the input images. In this layer, We use a filter or Kernel method to extract features from the input image.

![image](https://miro.medium.com/v2/resize:fit:396/format:webp/1*l6bbg1myg35Efz6Cd5tvxA.png)


### Pooling layer

The primary aim of this layer is to decrease the size of the convolved feature map to reduce computational costs. This is performed by decreasing the connections between layers and independently operating on each feature map. Depending upon the method used, there are several types of Pooling operations. We have Max pooling and average pooling.

![image](https://github.com/Adithyadk/OCR-Optical-Character-Recognition/assets/114129263/0221760a-404a-47d3-86c2-ab4db0e60ff5)


### Fully-connected layer

The Fully Connected (FC) layer consists of the weights and biases along with the neurons and is used to connect the neurons between two different layers. These layers are usually placed before the output layer and form the last few layers of a CNN Architecture.

### Dropout

Another typical characteristic of CNNs is a Dropout layer. The Dropout layer is a mask that nullifies the contribution of some neurons towards the next layer and leaves unmodified all others.

![image](https://miro.medium.com/v2/resize:fit:828/format:webp/1*58cnSVOrz6g_wa5M-HPpuQ.jpeg)

### Activation Function
An Activation Function decides whether a neuron should be activated or not. This means that it will decide whether the neuron’s input to the network is important or not in the process of prediction. There are several commonly used activation functions such as the ReLU, Softmax, tanH, and the Sigmoid functions. Each of these functions has a specific usage.

**Sigmoid**  — For a binary classification in the CNN model

**tanH** - The tanh function is very similar to the sigmoid function. The only difference is that it is symmetric around the origin. The range of values, in this case, is from -1 to 1.

**Softmax**- It is used in multinomial logistic regression and is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes.

**RelU**- the main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time.



**Alphabetic OCR with TensorFlow**

This project implements an Optical Character Recognition (OCR) model using TensorFlow and Keras that recognizes single alphabets (a-z).

**Requirements:**

* TensorFlow ([https://www.tensorflow.org/](https://www.tensorflow.org/))
* Keras ([https://keras.io/](https://keras.io/))
* NumPy ([https://numpy.org/](https://numpy.org/))

**Data Preparation:**

The project expects two folders named `Train` and `Test` inside the main project directory. These folders should contain images of handwritten alphabets categorized according to their corresponding letters. 

**Model Architecture:**

The model utilizes a Convolutional Neural Network (CNN) architecture with convolutional layers for feature extraction and fully-connected layers for classification.

* **Convolutional Layers:** These layers extract features from the input images. They use filters to identify patterns in the data.
* **Max Pooling Layers:** Reduce the dimensionality of the data extracted by convolutional layers.
* **Batch Normalization Layers:** Normalize the data across batches to improve training stability.
* **Flatten Layer:** Reshapes the data from a multi-dimensional tensor to a single-dimensional vector before feeding it into the fully-connected layers.
* **Dense Layers:** Fully-connected layers that perform the final classification of the extracted features.

**Training:**

The script trains the model using the `image_dataset_from_directory` function from TensorFlow to automatically load and preprocess the images from the `Train` and `Test` folders.

**Prediction:**

The script demonstrates how to use the trained model for predicting the alphabet from a single image. It loads the image, preprocesses it, and feeds it to the model for prediction.

**Running the Script:**

1. Ensure you have TensorFlow, Keras, and NumPy installed.
2. Download your training and testing datasets and place them in the `Train` and `Test` folders respectively. 
3. Run the script `ocr.py` (assuming you named it that).

**Saving and Loading the Model:**

The script saves the trained model as `New_Trained_Model`. You can load this model in future predictions using the `load_model` function.

**Further Development:**

This is a basic example of an OCR model for alphabets. Here are some ideas for further development:

* Extend the model to recognize uppercase and lowercase letters.
* Implement techniques for handling noise and distortions in the images.
* Increase the model complexity for improved accuracy. 



## Key Features

- **CNN Architecture**: Utilizes a Convolutional Neural Network architecture for effective feature extraction and classification of alphabetic characters.
  
- **Training and Validation**: Trains the OCR model using labeled training data and validates its performance using separate validation data.

- **Prediction**: Provides functionality to make predictions on new images, enabling users to input an image containing an alphabet and receive the corresponding predicted character.

## Contribution

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### I hope this README provides a comprehensive overview of the project!

