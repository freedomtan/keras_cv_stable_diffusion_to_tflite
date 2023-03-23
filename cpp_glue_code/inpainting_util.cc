#include "inpainting_util.h"

std::vector<float> read_raw_image(std::string filename)
{
  std::ifstream raw_image_stream(filename, std::ifstream::binary);
  std::vector<float> normalized_image;

  int8_t p;
  auto index = 0;

  while (raw_image_stream >> p) {
    normalized_image.push_back(p / 255.0 * 2.0 - 1.0);
    index++;
  }
  return normalized_image;
}

std::vector<float> maxpool2d(std::vector<float> in, int ksize, int strides, std::string padding) {
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
                                            {1, 512, 512, 1}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteFloat32, "input",
                                            {1, 64, 64, 1}, quant);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* max_pool2d_op =
      resolver.FindOp(tflite::BuiltinOperator_MAX_POOL_2D, 1);

  auto* params = reinterpret_cast<TfLitePoolParams*>(malloc(sizeof(TfLitePoolParams)));
  params->padding = kTfLitePaddingSame;
  params->stride_width = strides;
  params->stride_height = strides;
  params->filter_width = ksize;
  params->filter_height = ksize;
  interpreter->AddNodeWithParameters({0}, {1}, nullptr, 0, params, max_pool2d_op,
                                     nullptr);

  interpreter->AllocateTensors();
  // auto input = interpreter->typed_tensor<float>(0);
  // memcpy(input, in, 512*512*1*4);
  std::copy(in.begin(), in.end(),
            interpreter->typed_input_tensor<int>(0));

  interpreter->Invoke();

  const std::vector<int> outputs = interpreter->outputs();
  auto output = interpreter->typed_tensor<float>(1);
  std::vector<float> o(output,
                       output + interpreter->tensor(outputs[0])->bytes / 4);
  return o;
}
