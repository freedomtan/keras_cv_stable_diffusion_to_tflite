#include <iostream>
#include <memory>

#include "bpe.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

float *
run_text_encoder(string prompt) {
  bpe bpe_encoder;

  auto encoded = bpe_encoder.encode(prompt);
  auto pos_ids = bpe_encoder.position_ids();

  auto model = tflite::FlatBufferModel::BuildFromFile("/tmp/sd_tflite/sd_text_encoder_fixed_batch.tflite");
  if (model == nullptr) {
    cout << "failed to load model "
         << "tflite/sd_text_encoder.tflite\n";
    return NULL;
  }

  std::unique_ptr<tflite::Interpreter> interpreter;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    cout << "failed to build interpreter\n";
    return NULL;
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    cout << "failed allocate tensors\n";
    return NULL;
  }

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  std::copy(pos_ids.begin(), pos_ids.end(),
            interpreter->typed_input_tensor<int>(0));
  std::copy(encoded.begin(), encoded.end(),
            interpreter->typed_input_tensor<int>(1));

  if (interpreter->Invoke() != kTfLiteOk) {
    cout << "Failed to invoke tflite!\n";
    exit(-1);
  }
  auto output = interpreter->typed_tensor<float>(outputs[0]);

  if (output != NULL) {
    for (int i = 0; i < 16; i++) {
      std::cout << output[768+i] << "\t";
    }
    std::cout << "\n";
  } else {
    cout << "how come\n";
  }

  return output;
}

int main(int argc, char *argv[]) {
  string prompt = "a photo of an astronaut riding a horse on Mars";
  if (argc == 2) prompt = argv[1];

  auto encoded_text = run_text_encoder(prompt);
}
