#include "inpainting_util.h"

std::vector<float> read_raw_image(std::string filename) {
  std::ifstream raw_image_stream(filename, std::ios::in |std::ios::binary);

  std::vector<uint8_t> contents((std::istreambuf_iterator<char>(raw_image_stream)), std::istreambuf_iterator<char>());
  std::vector<float> normalized_image;

  for (int i=0; i < contents.size(); i++) {
    normalized_image.push_back(contents[i] / 255.0 * 2.0 - 1.0);
  }
  return normalized_image;
}

std::vector<float> read_raw_mask(std::string filename) {
  std::ifstream raw_mask_stream(filename, std::ifstream::binary);
  std::vector<float> mask;
  uint8_t p;

  while (raw_mask_stream >> p) {
    mask.push_back(p * 1.0);
    mask.push_back(p * 1.0);
    mask.push_back(p * 1.0);
    mask.push_back(p * 1.0);
  }
  return mask;
}

std::vector<float> maxpool2d(std::vector<float> in, int width, int height,
                             int ksize, int strides, std::string padding) {
  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);

  int base_index = 0;

  // two inputs: input and new_sizes
  interpreter->AddTensors(1, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0});
  interpreter->SetOutputs({1});

  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input",
                                            {1, width, height, 4}, quant);
  interpreter->SetTensorParametersReadWrite(
      1, kTfLiteFloat32, "output", {1, width / strides, height / strides, 4},
      quant);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* max_pool2d_op =
      resolver.FindOp(tflite::BuiltinOperator_MAX_POOL_2D, 1);

  auto* params =
      reinterpret_cast<TfLitePoolParams*>(malloc(sizeof(TfLitePoolParams)));
  params->padding = kTfLitePaddingSame;
  params->stride_width = strides;
  params->stride_height = strides;
  params->filter_width = ksize;
  params->filter_height = ksize;
  interpreter->AddNodeWithParameters({0}, {1}, nullptr, 0, params,
                                     max_pool2d_op, nullptr);

  interpreter->AllocateTensors();

  // fill input tensor
  std::copy(in.begin(), in.end(), interpreter->typed_input_tensor<float>(0));

  interpreter->Invoke();

  const std::vector<int> outputs = interpreter->outputs();
  auto output = interpreter->typed_tensor<float>(1);
  std::vector<float> o(output,
                       output + interpreter->tensor(outputs[0])->bytes / 4);
  return o;
}
