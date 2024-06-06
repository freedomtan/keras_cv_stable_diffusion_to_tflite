To run Stable Diffusion end-to-end on Android, we need some glue code

* [bpe.cc](bpe.cc): quick-and-dirty implementation of a BPE encoder
* [merges.txt](merges.txt), [vocab.txt](vocab.txt): dictionary files converted from CLIP's [vocab_16e6](https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz) file
* [scheduling_util.cc](scheduling_util.cc): some functions for scheduling/sampling
* [test_stable_diffusion.cc](test_stable_diffusion.cc): run the 3 models with glue code

# How to build 
* set ANDROID_NDK_HOME environment variable, preferrably android ndk r21 or earlier because of TF 2.12
* `bazel build --config android_arm64 :test_stable_diffusion`  to build binary for android_arm64
  (`bazel build --config android_arm64 :test_stable_diffusion` to build binary for host machine)
  
# How run it
With `bazel-bin/test_stable_diffusion`, we can get a raw rgb file named`decoded.raw`. With tools such as ImageMagick (`convert -depth 8 -size 512x512+0 rgb:decoded.raw decoded.png`), we can got .png file.
## where to get the 3 tflite models
You can get them 
* by converting from Keras CV models with the [script](https://github.com/freedomtan/keras_cv_stable_diffusion_to_tflite/blob/main/convert_to_tflite_models_with_dynamic_range.py), or 
* from [HuggingFace](https://huggingface.co/freedomtw/stable_diffusion_tflite/tree/main)

# TODO
Currently, it takes only one arguement, the prompt, there are many parameters we can change,e.g., number of steps, batch size, noise, etc.
