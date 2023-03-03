To run Stable Diffusion end-to-end on Android, we need some glue code

* [bpe.cc](bpe.cc): quick-and-dirty implementation of a BPE encoder
* [merges.txt](merges.txt), [vocab.txt](vocab.txt): dictionary files converted from CLIP's [vocab_16e6](https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz) file
* [scheduling_util.cc](scheduling_util.cc): some functions for scheduling/sampling
* [test_stable_diffusion.cc](test_stable_diffusion.cc): some functions for scheduling/sampling

# How to build 
* set ANDROID_NDK_HOME environment variable
* `bazel build -c android_arm64 :test_stable_diffusion`  to build binary for android_arm64
  (`bazel build -c android_arm64 :test_stable_diffusion` to build binary for host machine)
  
# How run it
With `bazel-bin/test_stable_diffusion`, we can get a raw rgb file named`decoded.raw`. With tools such as ImageMagick (`convert -depth 8 -size 512x512+0 rgb:decoded.raw decoded.png`), we can got .png file.
