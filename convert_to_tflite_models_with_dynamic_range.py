import tensorflow as tf
import keras_cv
from tensorflow import keras

# load the pipeline, get models

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
text_encoder_model = model.text_encoder
decoder_model = model.decoder
diffusion_model = model.diffusion_model
image_encoder_model = model.image_encoder

# convert models to tflite
converter1 = tf.lite.TFLiteConverter.from_keras_model(text_encoder_model)
converter1.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_text_encoder = converter1.convert()

with open('/tmp/sd_text_encoder_dynamic.tflite', 'wb') as f:
    f.write(tflite_text_encoder)

converter2 = tf.lite.TFLiteConverter.from_keras_model(diffusion_model)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_diffusion_model = converter2.convert()

with open('/tmp/sd_diffusion_model_dynamic.tflite', 'wb') as f:
    f.write(tflite_diffusion_model)

converter3 = tf.lite.TFLiteConverter.from_keras_model(decoder_model)
converter3.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_decoder = converter3.convert()

with open('/tmp/sd_decoder_dynamic.tflite', 'wb') as f:
    f.write(tflite_decoder)

converter4 = tf.lite.TFLiteConverter.from_keras_model(image_encoder_model)
converter4.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_image_encoder = converter4.convert()

with open('/tmp/sd_image_encoder_dynamic.tflite', 'wb') as f:
    f.write(tflite_image_encoder)
