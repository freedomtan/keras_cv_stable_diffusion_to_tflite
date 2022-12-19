# Scripts for converting Keras CV Stable Diffusion models to tflite

Tested with TensorFlow master branch (supposedly, using tf-nightly should be fine)

0. update group normalization, if you run into related problems, https://github.com/keras-team/keras-cv/pull/1035
1. [text encoder and decoder](convert_text_encoder_and_decoder_to_tflite_models.ipynb): Converting the text_encoder and decoder is trivial.
2. [diffusion model](convert_keras_diffusion_model_into_two_tflite_models.ipynb): Converting the diffusion model needs extra effort. The model weights of the diffusion models is about 3.4 GiB, which is much larger than file size limit (2 GiB) of [flatbuffer](https://google.github.io/flatbuffers/), the format TFLite used for its models. Surely, it's possible to modify flatbuffer to use 64-bit offset so that we can overcome the 2 GiB limit, but that will result in imcompatible files.
