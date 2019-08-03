# **Traffic Sign Recognition Classifier** 

---
Overview
---
In this project, deep neural networks and convolutional neural networks were used to classify traffic signs. I have trained and validated the model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I then tried out my model on images of German traffic signs which I downloaded from the web.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./examples/data_exploration.png "Data Exploration"
[image2]: ./examples/training_data.png "Histogram of Training Data"
[image3]: ./examples/test_data.png "Histogram of Test Data"
[image4]: ./examples/val_data.png "Histogram of validation data"
[image5]: ./examples/lenet.PNG "LeNet Architecture"
[image6]: ./examples/capture.PNG "dimensionality"
[image7]: ./examples/test_images.PNG "dimensionality"


---

### Data Set Summary & Exploration
I used the pandas library to calculate summary statistics of the traffic signs data set:
```
# Import pandas library
import pandas as pd

# Number of training examples
train_num = len(X_train)

# Number of validation examples
validation_num = len(X_valid)

# Number of testing examples.
test_num = len(X_test)

# The shape of an traffic sign image
image_shape = X_train[0].shape

# The number of unique classes/labels there are in the dataset.
classes_num = len(pd.Series(y_train).unique())

print("Size of the training set =", train_num)
print("Size of the validation set =", validation_num)
print("Size of the testing set =", test_num)
print("The shape of a traffic sign image =", image_shape)
print("The number of unique classes/labels in the data set is =", classes_num)
```

#### Exploratory visualization of the dataset.
The pickled data is a dictionary with 4 key/value pairs:
- ‘features’ is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- ‘labels’ is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- ‘sizes’ is a list containing tuples, (width, height) representing the original width and height the image.
- ‘coords’ is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. (THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES).

I used the pandas library to calculate summary statistics of the traffic signs data set:
```
Size of the training set = 34799
Size of the validation set = 4410
Size of the testing set = 12630
The shape of a traffic sign image = (32, 32, 3)
The number of unique classes/labels in the data set is = 43
```

Histogram of images of Traffic Sign Dataset (training data, test data and validation data) are as shown below. This indicates the dataset contains 43 unique classes of traffic sign signals.

![alt text][image2]
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture
##### Data preprocessing
**Shuffling**: Initial step followed in data pre-processing is to shuffle the data. It is very important to shuffle the training data otherwise ordering of data might have huge effect on how the network trends (Neural Network training).
**Normalization**: The image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel — pixel_mean)/ (max_pixel-min_pixel)` is used to normalize the data. After normalization pixel values will be in the range -1 to +1.
Other method which can be used in normalizing the data is (pixel — 128)/ 128.

##### Model Architecture
LeNet Architecture is used to build the Traffic Sign Classifier model. It includes two Convolutional Layers and three Fully-Connected layers.
![alt text][image5]

**Dimensionality**: The number of neurons of each layer in our CNN can be calculated by using below formula,
```
output_height = [(input_height - filter_height + 2 * padding) / vertical_stride] + 1
output_width = [(input_width - filter_width + 2 * padding) / vertical_stride] + 1
output_depth = number of filters
```
##### LeNet Architecture:
```
Layer 1: Convolutional Layer. 
Input = 32x32x3. Output = 28x28x6.
5X5 filter is used with an input depth of 3 and output depth of 6. 
output_height = [(32–5+2*0)/1]+1 = 28.
output_width = [(32–5+2*0)/1]+1 = 28.
output_depth = filter_depth = 6
 
Activation: ReLU activation function is used to activate the output of Convolutional Layer.

Pooling: Input = 28x28x6. Output = 14x14x6.
Pool the output using 2x2 kernel with a 2x2 stride.
output_height = 28/2 = 14.
output_width = 28/2 = 14.
output_depth = 6

Layer 2: Convolutional Layer.
Input = 14x14x6. Output = 10x10x16.
output_height = [(14-5+2*0)/1]+1 = 10.
output_width = [(14-5+2*0)/1]+1 = 10.
output_depth = filter_depth = 16
Activation: ReLU activation function is used to activate the output of Convolutional Layer.

Pooling: Input = 10x10x16. Output = 5x5x16.
Pool the output using 2x2 kernel with a 2x2 stride 
output_height = 10/2 = 5.
output_width = 10/2 = 5.
output_depth = 16

Flatten: Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. 
Input = 5x5x16. Output=5x5x16=400.

Layer 3: Fully Connected. Input = 400. Output = 120.

Activation: ReLU activation function is used to activate the output of Fully Connected Layer.

Layer 4: Fully Connected. Input = 120. Output = 84.

Activation: ReLU activation function is used to activate the output of Fully Connected Layer.

Layer 5: Fully Connected. Input = 84. Output = 43.
Set the output with a width equal to number of classes in our label set. These outrputs are called as logits.

Output: Return the result of the 3rd fully connected layer.

```
#### Model Training
**Hyper-Parameters**:
- EPOCH variable is used to tell the TensorFlow how many times to run our training data through the network. More number of EPOCHS results in better model training but it takes a longer time to train the network.
- BATCH_SIZE variable is used to tell the TensorFlow how many training images to run through the network at a time. If the BATCH_SIZE is larger, the model gets trained faster but our processor may have a memory limit on how large a batch it can run. EPOCH and BATCH SIZE values affect the training speed and model accuracy.
- Learning rate tells the TensorFlow how quickly to update the network weights.

Chosen: EPOCH=50, BATCH_SIZE = 128, learning_rate = 0.001

**Training Pipeline:**
- Pass the input data to the LeNet function to calculate our logits.
- ‘tf.nn.softmax_cross_entropy_with_logits’ function is used to compare the logits to the ground truth training labels and calculate the cross entropy. Cross Entropy is just a measure of how different the logits are from the ground truth training labels.
- tf.reduce_mean’ function averages the cross entropy from all of the training images.
- Adam Optimizer uses the Adam algorithm to minimize the loss function using learning rate.
- Run the minimize function on the optimizer which uses back-propagation to update the network and minimize our training loss.
```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)
```
#### Model Evaluation
Evaluate how well the model loss and accuracy of the model for a given dataset. A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. High accuracy on the training set but low accuracy on the validation set implies overfitting.
```
# Measure whether given prediction is correct by comparing logit prediction to the one-hot encoded ground truth label.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
# Calculte the model's overall accuracy by averaging the individual prediction accuracies.
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    # Break the training data into batches and train the model on each batch.
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE] # batch the dataset
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y}) # run through evaluation pipeline
        total_accuracy += (accuracy * len(batch_x)) # Calculate the avarage accuracy of each batch to calculate the total accuracy of the model
        total_loss += (loss * len(batch_x))
    return total_accuracy/num_examples
```
- Run the training data through the training pipeline to train the model.
- Before each epoch shuffles the training set to ensure that our training is not biased by the order of the images.
- Break the training data into batches and train the model on each batch.
- At the end of each epoch, we evaluate the model on our validation data.
- Once we have completely trained the model, save it. We can load it later or modify it or evaluate our model on the test dataset.
```
# Create TensorFlow session and initialize variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    # Train over 'N' number of epochs
    for i in range(EPOCHS):
        # Before each epoch shuffle the training set.
        X_train, y_train = shuffle(X_train, y_train)
        # Break the training data into batches and train the model on each batch.
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        # At the end of each epoch, we evaluate the model on our validation data
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    # Save the model after the completion of training.
    saver.save(sess, './lenet')
    print("Model saved")
```

As we train the model, we see that validation accuracy starts off really high and stays there. This is the result of the powerful Convolutional Network Architecture LeNet and because of the choice hyperparameters.
```
Training...

EPOCH 1 ...
Validation Accuracy = 0.701

EPOCH 2 ...
Validation Accuracy = 0.814

EPOCH 3 ...
Validation Accuracy = 0.850

EPOCH 4 ...
Validation Accuracy = 0.874

EPOCH 5 ...
Validation Accuracy = 0.888

EPOCH 6 ...
Validation Accuracy = 0.896

......


EPOCH 48 ...
Validation Accuracy = 0.922

EPOCH 49 ...
Validation Accuracy = 0.934

EPOCH 50 ...
Validation Accuracy = 0.937

Model saved
```
Evaluate the performance of the model on the test set. This has to be done only once after the completion of training. Otherwise, we would be using the test dataset to choose the best model and then the test dataset would not provide a good estimate of how well the model would do in the real world.
```
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```
Test Accuracy = 0.925 is obtained when the Model is tested across the Test dataset.

#### Test a Model on New Images
To give more insight into how the model is working, five pictures of German traffic signs were downloaded from the web and the model is used to predict the traffic sign type.

Here are the results of the prediction:
![alt text][image7]
```
11 -- Right-of-way at the next intersection
12 -- Priority road
1 -- Speed limit (30km/h)
25 -- Road work
38 -- Keep right
```
The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.
```
True Label is -->    11:Right-of-way at the next intersection

   11: Right-of-way at the next intersection 100.000%
   30: Beware of ice/snow             0.000%
   21: Double curve                   0.000%
    7: Speed limit (100km/h)          0.000%
   31: Wild animals crossing          0.000%
##############################################
##############################################
True Label is -->    12:Priority road                 

   12: Priority road                  100.000%
   10: No passing for vehicles over 3.5 metric tons 0.000%
   42: End of no passing by vehicles over 3.5 metric tons 0.000%
    5: Speed limit (80km/h)           0.000%
   11: Right-of-way at the next intersection 0.000%
##############################################
##############################################
True Label is -->     1:Speed limit (30km/h)          

    2: Speed limit (50km/h)           100.000%
    1: Speed limit (30km/h)           0.000%
    3: Speed limit (60km/h)           0.000%
    0: Speed limit (20km/h)           0.000%
    4: Speed limit (70km/h)           0.000%
##############################################
##############################################
True Label is -->    25:Road work                     

   25: Road work                      100.000%
   20: Dangerous curve to the right   0.000%
   24: Road narrows on the right      0.000%
   11: Right-of-way at the next intersection 0.000%
    3: Speed limit (60km/h)           0.000%
##############################################
##############################################
True Label is -->    38:Keep right                    

   38: Keep right                     100.000%
   34: Turn left ahead                0.000%
   36: Go straight or right           0.000%
   32: End of all speed and passing limits 0.000%
   20: Dangerous curve to the right   0.000%
```