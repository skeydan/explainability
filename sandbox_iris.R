library(dplyr)
library(lime)
library(keras)
library(MASS)


iris_test <- iris[1, 1:4]
iris_train <- iris[-1, 1:4]
iris_lab <- iris[[5]][-1]

# lda
model <- lda(iris_train, iris_lab)
model %>% predict(iris_test)
explainer <- lime(iris_train, model)

explanation <- lime::explain(iris_test, explainer, n_labels = 1, n_features = 2,
                       #feature_select = "lasso_path")
                       feature_select = "highest_weights")
                       #  feature_select = "forward_selection",
                       #  feature_select = "tree",)
explanation
plot_features(explanation)


# mlp
iris_train <- as.matrix(iris_train)
iris_cat <- to_categorical(as.numeric(iris_lab)-1, num_classes = 3)


model <- keras_model_sequential()
model %>% layer_dense(units= 3, input_shape = 4) %>%
  layer_activation("softmax")
model %>% compile(optimizer = "adam", loss = "categorical_crossentropy")
model %>% summary()
hist <- model %>% fit(x = iris_train, y = iris_cat, epochs = 500, batch_size = 10)
plot(hist)

# http://www.business-science.io/business/2017/11/28/customer_churn_analysis_keras.html

model_type.keras.models.Sequential <- function(x, ...) {
  return("classification")
}

predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(Yes = pred, No = 1 - pred))
}

predict_model(x = model, newdata = iris_test, type = 'raw') %>%
  tibble::as_tibble()

explainer <- lime(
  x              = data.frame(iris_train), 
  model          = model, 
  bin_continuous = TRUE)

explanation <- lime::explain(
  iris_test, 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 2,
  feature_select = "lasso_path",
#  feature_select = "highest_weights",
#  feature_select = "forward_selection",
#  feature_select = "tree",
  kernel_width = 0.5)

explanation
plot_features(explanation)
plot_explanations(explanation)
