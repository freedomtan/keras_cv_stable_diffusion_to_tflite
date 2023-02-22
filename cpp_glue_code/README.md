To run Stable Diffusion end-to-end on Android, we need some glue code

* [bpe.cc](bpe.cc): quick-and-dirty implementation of a BPE encoder
* [merges.txt](merges.txt), [vocab.txt](vocab.txt): dictionary files converted from CLIP's [vocab_16e6](https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz) file
