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
    name = "libsd",
    srcs = [
        "bpe.cc", "scheduling_util.cc"
    ],
    hdrs = ["bpe.h", "scheduling_util.h"],
)

cc_binary(
    name = "test_stable_diffusion",
    srcs = ["test_stable_diffusion.cc"],
    deps = [
      ":libsd",
      "@org_tensorflow//tensorflow/lite:framework",
      "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    
)
