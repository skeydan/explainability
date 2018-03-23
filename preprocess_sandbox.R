library(keras)
library(reticulate)

img <- array(c(rep(0,4), c(0, 127, 128, 255), rep(255,4)), dim = c(2,2,3))
img
typeof(img)

#img <- image_to_array(img)
#typeof(img)

# subtract imagenet channel means
# [123.68, 116.779, 103.939] 
# RGB -> BGR
imagenet_preprocess_input(img)

# inception
# def preprocess_input(x):
#    x /= 255.
#    x -= 0.5
#    x *= 2.
# return x
inception_v3_preprocess_input(img)
inception_resnet_v2_preprocess_input(img)
xception_preprocess_input(img)


####


k <- import("keras")
k
k_img <- keras_array(img)
k$applications$imagenet_utils$preprocess_input(k_img)

