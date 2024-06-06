# Converting Keras CV Stable Diffusion models to tflite

## environments
TensorFlow: 2.12.1
keras: 2.12.0
keras-cv: 0.4.2

Note that different version of TensorFlow and Keras could generate
incompatible tflite files.

## Conversion scripts

### fp32 
Tested with TensorFlow master branch (supposedly, using tf-nightly should be fine)

0. update group normalization, if you run into related problems, https://github.com/keras-team/keras-cv/pull/1035
1. [text encoder and decoder](convert_text_encoder_and_decoder_to_tflite_models.ipynb): Converting the text_encoder and decoder is trivial.
2. [diffusion model](convert_keras_diffusion_model_into_two_tflite_models.ipynb): Converting the diffusion model needs extra effort. The model weights of the diffusion models is about 3.4 GiB, which is much larger than file size limit (2 GiB) of [flatbuffer](https://google.github.io/flatbuffers/), the format TFLite used for its models. Surely, it's possible to modify flatbuffer to use 64-bit offset so that we can overcome the 2 GiB limit, but that will result in imcompatible files.

### full-integer quantization
It's also possible to generate Quantized int8 models with Post-Training Quantization (PTQ). I don't know how to generate representative datasets, but as a proof of concept, I wrote a [script](convert_keras_diffusion_model_into_two_tflite_models_qint8.ipynb) that uses only one sample input as the dataset :-)

### dynamic range quantization
Conversion [script](https://github.com/freedomtan/keras_cv_stable_diffusion_to_tflite/blob/main/convert_to_tflite_models_with_dynamic_range.py)

## Testing/Verifying converted tflite models
Borrowing some code from Keras CV implementation, we can do end-to-end test of converted TFLite models
* fp32 modeles: [notebook](text_to_image_using_converted_tflite_models.ipynb)
* dynamic range quantized modeles: [notebook](text_to_image_using_converted_tflite_models_dynamic.ipynb)

### test image encoder model with inpainting 
* dynamice range quantized models: [notebook](inpainting_using_converted_tflite_models_dynamic_2_in_1.ipynb)
The ['man-on-skateboard-cropped.png'](man-on-skateboard-cropped.png) is from this [tutorial](https://www.assemblyai.com/blog/stable-diffusion-in-keras-a-simple-tutorial/)

## Converted models
I put tflite models I converted to [HuggingFace](https://huggingface.co/freedomtw/stable_diffusion_tflite)

