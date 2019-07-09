library(keras)
library(tidyverse)

# https://tensorflow.rstudio.com/keras/
# https://keras.rstudio.com/index.html

# mnist tutorial

# load data

# MNIST consists of 28 x 28 grayscale images of handwritten digits.  the output is the predicted digit
mnist <- dataset_mnist()
mnist %>% glimpse()


##################


# extract train x and y data

# The x data is a 3-d array (images,width,height) of grayscale values . 
# To prepare the data for training we convert the 3-d arrays into matrices by reshaping width and height 
# into a single dimension (28x28 images are flattened into length 784 vectors). 
# Then, we convert the grayscale values from integers ranging between 0 to 255 into floating point values 
# ranging between 0 and 1:

x_train <- mnist$train$x
x_train %>% glimpse()
x_train %>% class()

y_train <- mnist$train$y
y_train %>% glimpse()
y_train %>% class()

# extract test x and y data
x_test <- mnist$test$x
y_test <- mnist$test$y


###############


# reshape x train/test data

# Note that we use the array_reshape() function rather than the dim<-() function to reshape the array. 
# This is so that the data is re-interpreted using row-major semantics (as opposed to R’s default column-major semantics), 
# which is in turn compatible with the way that the numerical libraries called by Keras interpret array dimensions.

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_train %>% glimpse()
x_train %>% class()

x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# rescale x train/test data
x_train <- x_train / 255
x_test <- x_test / 255

# inspect x_test
x_test %>% glimpse()
x_test %>% class()
df <- x_test %>% data.frame()
df %>% glimpse()
df %>% select(1:5) %>% slice(1:10)
df %>% mutate(row_number = row_number()) %>% filter(X260 == 0) %>% nrow()
df %>% mutate(row_number = row_number()) %>% filter(X260 != 0) %>% nrow()
df %>% select(1, 260:265) %>% slice(1:10)


################


# one-hot encode y data to binary class matrices

# The y data is an integer vector with values ranging from 0 to 9. 
# To prepare this data for training we one-hot encode the vectors into 
# binary class matrices using the Keras to_categorical() function:
y_train <- to_categorical(y_train, 10)
y_train %>% glimpse()
y_train %>% class()

y_test <- to_categorical(y_test, 10)


#####################


# defining the model

# The core data structure of Keras is a model, a way to organize layers. 
# The simplest type of model is the Sequential model, a linear stack of layers.

# The input_shape argument to the first layer specifies the shape of the 
# input data (a length 784 numeric vector representing a grayscale image). 
# The final layer outputs a length 10 numeric vector (probabilities for each digit) 
# using a softmax activation function.The input_shape argument to the first layer 
# specifies the shape of the input data (a length 784 numeric vector representing a grayscale image). 
# The final layer outputs a length 10 numeric vector (probabilities for each digit) using a softmax activation function.

model <- keras_model_sequential() 
model %>% 
        layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
        layer_dropout(rate = 0.4) %>% 
        layer_dense(units = 128, activation = 'relu') %>%
        layer_dropout(rate = 0.3) %>%
        layer_dense(units = 10, activation = 'softmax')

summary(model)

model
model %>% class()


# Next, compile the model with appropriate loss function, optimizer, and metrics:
model %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizer_rmsprop(),
        metrics = c('accuracy')
)

model
model %>% class()


#############################


# train the model

# Use the fit() function to train the model for 30 epochs using batches of 128 images:
history <- model %>% fit(
        x_train, y_train, 
        epochs = 30, batch_size = 128, 
        validation_split = 0.2
)

history %>% class()

# inspect model performance on validation set
plot(history)


#############################


# evaluate model performance on test set
model %>% evaluate(x_test, y_test)


############################


# make predictions on test set
predictions <- model %>% predict_classes(x_test)
predictions %>% class()
predictions %>% glimpse()
predictions %>% enframe()
predictions %>% data.frame() %>% head()

# inspect x_test
x_test %>% glimpse()
x_test %>% class()
df <- x_test %>% data.frame()
df %>% glimpse()
df %>% select(1:5) %>% slice(1:10)
df %>% mutate(row_number = row_number()) %>% filter(X260 == 0) %>% nrow()
df %>% mutate(row_number = row_number()) %>% filter(X260 != 0) %>% nrow()
df %>% select(1, 260:265) %>% slice(1:10)









