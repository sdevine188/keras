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


##################################################################################################
####################################################################################################
####################################################################################################


# regression
# https://keras.rstudio.com/articles/tutorial_basic_regression.html

# get data
boston_housing <- dataset_boston_housing()
boston_housing %>% glimpse()

# split initial data to get vectors of train/test data/labels
c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

# inspect
# note the train_data contains 13 variabels for each of 404 records, so its 5252 long
paste0("Training entries: ", length(train_data), ", labels: ", length(train_labels))

train_data[1, ] # Display sample features, notice the different scales
train_labels[1:10] # note the labels are house prices in thousands

# add column names
column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')
train_df <- as_tibble(train_data)
colnames(train_df) <- column_names
train_df


########################


# normalize data
# Test data is *not* used when calculating the mean and std.

# Normalize training data
train_data <- scale(train_data) 

# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(train_data, "scaled:center") 
col_stddevs_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

train_data[1, ] # First training sample, normalized


##################


# create the model
build_model <- function() {
        
        model <- keras_model_sequential() %>%
                layer_dense(units = 64, activation = "relu",
                            input_shape = dim(train_data)[2]) %>%
                layer_dense(units = 64, activation = "relu") %>%
                layer_dense(units = 1)
        
        model %>% compile(
                loss = "mse",
                optimizer = optimizer_rmsprop(),
                metrics = list("mean_absolute_error")
        )
        
        model
}

model <- build_model()
model %>% summary()


#################


# train the model
# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
        on_epoch_end = function(epoch, logs) {
                if (epoch %% 80 == 0) cat("\n")
                cat(".")
        }
)    

epochs <- 500

# Fit the model and store training stats
history <- model %>% fit(
        train_data,
        train_labels,
        epochs = epochs,
        validation_split = 0.2,
        verbose = 0,
        callbacks = list(print_dot_callback)
)

# inspect
history

library(ggplot2)
plot(history, metrics = "mean_absolute_error", smooth = FALSE) +
        coord_cartesian(ylim = c(0, 5))

###############


# update the fit to stop running new epochs once it stops improving 
# The patience parameter is the amount of epochs to check for improvement.
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

model <- build_model()
history <- model %>% fit(
        train_data,
        train_labels,
        epochs = epochs,
        validation_split = 0.2,
        verbose = 0,
        callbacks = list(early_stop, print_dot_callback)
)

plot(history, metrics = "mean_absolute_error", smooth = FALSE) +
        coord_cartesian(xlim = c(0, 150), ylim = c(0, 5))


###############


# evaluate model on test set
c(loss, mae) %<-% (model %>% evaluate(test_data, test_labels, verbose = 0))
paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))

# predict on test set
test_predictions <- model %>% predict(test_data)
test_predictions[ , 1]
