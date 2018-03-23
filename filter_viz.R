library(keras)
library(grid)
library(gridExtra)

k_set_learning_phase(0)

dir_name <- "filters_inceptionv3"
#dir_name <- "filters_xception"

model_name <- "inceptionv3"
#model_name <- "xception"

width_height <- 150

num_steps <- 50 # times to perform gradient ascent

if (!dir.exists(dir_name)) dir.create(dir_name)

model <- switch(model_name,
                inceptionv3 = application_inception_v3(weights = "imagenet", include_top = FALSE),
                xception = application_xception(weights = "imagenet", include_top = FALSE))

model

# mean 0.5, sd 0.1
deprocess_image <- function(x) {
  dms <- dim(x)
  x <- x - mean(x)
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1
  x <- x + 0.5
  x <- pmax(0, pmin(x, 1))
  array(x, dim = dms)
}

generate_pattern <- function(layer_name, filter_index, size = 150) {
  layer_output <- get_layer(model, layer_name)$output
  layer_output
  
  # loss is average activation at specified filter
  loss <- k_mean(layer_output[, , , filter_index])
  
  grads <- k_gradients(loss, model$input)[[1]]
  # normalize the gradient tensor by dividing it by its L2 norm
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  iterate <- k_function(list(model$input), list(loss, grads))
  
  input_img_data <-
    array(runif(width_height * width_height * 3), dim = c(1, width_height, width_height, 3)) * 20 + 128
  
  step <- 1
  for (i in 1:num_steps) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    # do gradient ascent
    input_img_data <- input_img_data + (grads_value * step)
  }
  
  img <- input_img_data[1, , , ]
  deprocess_image(img)
  
}

# quick test
#grid.raster(generate_pattern("conv2d_3", 1))
#grid.raster(generate_pattern("block14_sepconv1", 1))

### inception v3
for (layer_name in paste0("conv2d_", seq(1,91, by = 10))) { 
  stopifnot(model_name == "inceptionv3")

### xception
#for (layer_name in paste0("block", 2:14, "_sepconv1")) { 
 # stopifnot(model_name == "xception")
  size <- 140
  png(paste0(dir_name, "/", layer_name, ".png"),
      width = 8 * size, height = 8 * size)
  grobs <- list()
  
  # just first 8 feature maps per layer
  for (i in 0:3) {
    for (j in 0:3) {
      pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
      grob <- rasterGrob(pattern,width = unit(0.9, "npc"),
                         height = unit(0.9, "npc"))
      grobs[[length(grobs)+1]] <- grob
    }
  }
  grid.arrange(grobs = grobs, ncol = 4)
  dev.off()
}

                         
                         
