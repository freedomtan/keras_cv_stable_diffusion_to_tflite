#ifndef _STABLE_DIFFUSION_INPAINTING_UTIL_H_
#define _STABLE_DIFFUSION_INPAINTING_UTIL_H_

#include <fstream>
#include <iostream>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

std::vector<float> read_raw_image(std::string filename);
std::vector<float> read_raw_mask(std::string filename);
std::vector<float> maxpool2d(std::vector<float> in, int width, int height, int ksize, int strides, std::string padding);

#endif
