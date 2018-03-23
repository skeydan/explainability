library(keras)
library(purrr)
library(stringr)

width_height <- 150

img_path <- "data/n02121620_cat/n02121620_1001.JPEG"

model_name <- "inceptionv3"
#model_name <- "xception"

dir_name <- paste0("activations_", model_name, "_", 
                   img_path %>% basename() %>% stringr::str_replace(".JPEG", ""))


if (!dir.exists(dir_name)) dir.create(dir_name)


img <- image_load(img_path, target_size = c(width_height, width_height))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, width_height, width_height, 3))

plot(as.raster(img_tensor[1,,,]/255))

# preprocess
img_tensor <- switch(model_name,
                     inceptionv3 = inception_v3_preprocess_input(img_tensor),
                     xception = xception_preprocess_input(img_tensor))

model <- switch(model_name,
                inceptionv3 = application_inception_v3(weights = "imagenet"),
                xception = application_xception(weights = "imagenet"))

model

model %>% predict(img_tensor) %>% imagenet_decode_predictions()

##
# Create a model that will return the layer outputs from the specified layers, given the input
max_plotting_range <- 1:200
conv_layers_to_plot <- model$layers[max_plotting_range] %>% 
  Filter(function(layer) class(layer)[1] == "keras.layers.convolutional.Conv2D", .)
layer_outputs <- lapply(conv_layers_to_plot, function(layer) layer$output)
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

activations <- activation_model %>% predict(img_tensor)
purrr::map(activations, dim)

plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1,
        col = terrain.colors(12))
}

image_size <- 58
images_per_row <- 16
for (i in seq_along(activations)) {
  layer_activation <- activations[[i]]
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  png(paste0(dir_name, "/conv_", i, ".png"),
      width = image_size * images_per_row,
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  par(op)
  dev.off()
}

