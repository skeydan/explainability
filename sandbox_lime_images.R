library(lime)
library(keras)
library(magick)
#library(dplyr)

model <- application_vgg16(
  weights = "imagenet",
  include_top = TRUE
)

img_path <- file.path("/home/key/pics/claude.jpeg")
img <- image_read(img_path)
img
as.raster(img) %>% dim()
#plot(as.raster(img))

image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224,224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}
#explainer <- lime(img_path, model, image_prep)
#explainer

res <- predict(model, image_prep(img_path))
res %>% which.max()
res %>% max()
imagenet_decode_predictions(res)

##
model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
model_labels
explainer <- lime(img_path, as_classifier(model, model_labels), image_prep)

##

#plot_superpixels(img_path)
#plot_superpixels(img_path, n_superpixels = 200, weight = 40)

##
explanation <- lime::explain(img_path, explainer, n_labels = 2, n_features = 20)
plot_image_explanation(explanation)
plot_image_explanation(explanation, display = 'block', threshold = 0.01)
plot_image_explanation(explanation, threshold = 0, show_negative = TRUE, fill_alpha = 0.6)


