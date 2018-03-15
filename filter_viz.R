library(keras)
library(grid)
library(gridExtra)

k_set_learning_phase(0)

model <- application_inception_v3(weights = "imagenet",
                                  include_top = FALSE)

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
  grads
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  iterate <- k_function(list(model$input), list(loss, grads))
  c(loss_value, grads_value) %<-%
    iterate(list(array(0, dim = c(1, 150, 150, 3))))
  
  input_img_data <-
    array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128
  step <- 1
  for (i in 1:40) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    # do gradient ascent
    input_img_data <- input_img_data + (grads_value * step)
  }
  
  img <- input_img_data[1, , , ]
  deprocess_image(img)
  
}

grid.raster(generate_pattern("conv2d_3", 1))

dir.create("filters_inceptionv3")

for (layer_name in paste0("conv2d_", seq(1,91, by = 10))) {
  size <- 140
  png(paste0("filters_inceptionv3/", layer_name, ".png"),
      width = 8 * size, height = 8 * size)
  grobs <- list()
  for (i in 0:3) {
    for (j in 0:3) {
      pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
      grob <- rasterGrob(pattern,width = unit(0.9, "npc"),
                         height = unit(0.9, "npc"))
      grobs[[length(grobs)+1]] <- grob
    }
  }
  grid.arrange(grobs = grobs, ncol = 8)
  dev.off()
}

                         
                         
