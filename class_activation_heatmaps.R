# A class activation heatmap is a 2D grid of scores associated with a specific output class,
# computed for every location in any input image, indicating how important each location is 
# with respect to the class under consideration.

# Grad- CAM : Visual Explanations from Deep Networks via Gradient-based Localization.

# It consists of taking the output feature map of a convolution layer, given an input
# image, and weighing every channel in that feature map by the gradient of the class
# with respect to the channel.
# Intuitively, one way to understand this trick is that you’re
# weighting a spatial map of how intensely the input image activates different channels
# by how important each channel is with regard to the class

library(keras)
library(purrr)
library(stringr)
library(magick)
library(viridis)

k_set_learning_phase(0)

width_height <- 150

img_path <- "data/n02121620_cat/n02121620_1001.JPEG"

model_name <- "inceptionv3"
#model_name <- "xception"

dir_name <- paste0("CAM_", model_name, "_", 
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

preds <- model %>% predict(img_tensor)
preds %>% imagenet_decode_predictions()
class_pred <- which.max(preds[1,])

# final output tensor, dim batch_size * 1000: get position of most probable class
max_class_output <- model$output[, class_pred]

# get gradients of last conv layer w.r.t. that class probability
last_conv_layer <- model %>% get_layer( switch(model_name,
                                               inceptionv3 = "conv2d_94",
                                               xception = "tbd"))
last_conv_layer$output$shape
grads <- k_gradients(max_class_output, last_conv_layer$output)[[1]]
grads

# mean gradient for each filter in this feature map
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
pooled_grads

iterate <- k_function(list(model$input),
                      list(pooled_grads, last_conv_layer$output[1,,,]))

c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img_tensor))

dim(pooled_grads_value)
dim(conv_layer_output_value)

# multiply all outputs from last conv layer with avg gradient w.r.t. that filter
# this weights activation by how much that filter matters to classification
for (i in 1:(dim(conv_layer_output_value)[[3]])) {
  conv_layer_output_value[,,i] <-
    conv_layer_output_value[,,i] * pooled_grads_value[[i]]
}

# heatmap contains mean gradient-weighted activation for each location (averaged over filters)
heatmap <- apply(conv_layer_output_value, c(1,2), mean)


heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)

# 
write_heatmap <- function(heatmap, filename, width = width_height, height = width_height,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}
write_heatmap(heatmap, paste0(dir_name, "/", last_conv_layer$name, ".png"))


# overlay image and heatmap
image <- image_read(img_path)
info <- image_info(image)
geometry <- sprintf("%dx%d!", info$width, info$height)

pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, paste0(dir_name, "/", last_conv_layer$name, ".png"),
              width = 14, height = 14, bg = NA, col = pal_col)

image_read(paste0(dir_name, "/", last_conv_layer$name, ".png")) %>%
image_resize(geometry, filter = "quadratic") %>%
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()

