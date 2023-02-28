#include <iostream>
#include <memory>

#include "bpe.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"

int main(int argc, char *argv[]) {
  bpe bpe_encoder;

  string prompt = "a photo of an astronaut riding a horse on Mars";
  if (argc == 2) prompt = argv[1];
  auto encoded = bpe_encoder.encode(prompt);
  auto pos_ids = bpe_encoder.position_ids();

  auto model =
      tflite::FlatBufferModel::BuildFromFile("/tmp/sd/sd_text_encoder.tflite");
  if (model == nullptr) {
    // Return error.
    cout << "failed to load model "
         << "/tmp/sd/sd_text_encoder.tflite\n";
  } else {
    cout << "here\n";
  }

  // Create an Interpreter with an InterpreterBuilder.
  std::unique_ptr<tflite::Interpreter> interpreter;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    // Return failure.
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    // Return failure.
  }

  auto input_size = interpreter->inputs().size();
  cout << "input_size: " << input_size << "\n";
}
