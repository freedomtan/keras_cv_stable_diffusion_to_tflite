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
        "bpe.cc",
        "inpainting_util.cc",
        "scheduling_util.cc",
    ],
    hdrs = [
        "bpe.h",
        "inpainting_util.h",
        "scheduling_util.h",
        "tflite_util.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "test_stable_diffusion",
    srcs = ["test_stable_diffusion.cc"],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":libsd",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "test_stable_diffusion_2_in_1",
    srcs = ["test_stable_diffusion_2_in_1.cc"],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":libsd",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "test_stable_diffusion_inpainting",
    srcs = ["test_stable_diffusion_inpainting.cc"],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":libsd",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "test_stable_diffusion_inpainting_2_in_1",
    srcs = ["test_stable_diffusion_inpainting_2_in_1.cc"],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":libsd",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
