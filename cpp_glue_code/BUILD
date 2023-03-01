load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts")
load("@org_tensorflow//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//:build_def.bzl", "android_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bpe",
    srcs = [
        "bpe.cc",
    ],
    hdrs = ["bpe.h"],
)

cc_binary(
    name = "test_text_encoder",
    srcs = ["test_stable_diffusion.cc"],
    deps = [
      ":bpe",
      "@org_tensorflow//tensorflow/lite:framework",
      "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    
)
