#ifndef _STABLE_DIFFUSION_TFLITE_UTIL_H_
#define _STABLE_DIFFUSION_TFLITE_UTIL_H_

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

#if __DEBUG__
using namespace std::literals;
extern std::chrono::time_point<std::chrono::system_clock> start_time;
#endif
class tflite_runner {
 public:
  tflite_runner(std::string model_name) {
    model = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
    if (model == nullptr) {
      std::cout << "failed to load model: " << model_name << "\n";
      return;
    }
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) !=
        kTfLiteOk) {
      std::cout << "failed to build interpreter\n";
      return;
    }
  }
  std::unique_ptr<tflite::Interpreter> &get_interpreter() {
    return interpreter;
  }

 private:
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
};

class diffusion_runner : tflite_runner {
 public:
  diffusion_runner()
      : diffusion_runner("sd_tflite/sd_diffusion_model_dynamic.tflite") {}

  diffusion_runner(std::string model_name) : tflite_runner(model_name) {
    interpreter = move(tflite_runner::get_interpreter());

    if (interpreter->AllocateTensors() != kTfLiteOk) {
      std::cout << "failed allocate tensors\n";
    }
  }

  vector<float> diffusion_run(vector<float> latent, vector<float> t_emb,
                              vector<float> context) {
#if __DEBUG__
    auto now = std::chrono::system_clock::now();
    std::cout << (now - start_time) / 1ms / 1000.0 << ": " << __LINE__ << ": "
              << __FUNCTION__ << "\n";
#endif
    // latent, t_emb, context
    std::copy(latent.begin(), latent.end(),
              interpreter->typed_input_tensor<float>(0));
    std::copy(t_emb.begin(), t_emb.end(),
              interpreter->typed_input_tensor<float>(1));
    std::copy(context.begin(), context.end(),
              interpreter->typed_input_tensor<float>(2));

    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);

    if (interpreter->Invoke() != kTfLiteOk) {
      cout << "Failed to invoke tflite!\n";
      exit(-1);
    }

    const std::vector<int> outputs = interpreter->outputs();
    auto output = interpreter->typed_tensor<float>(outputs[0]);
    std::vector<float> o(output,
                         output + interpreter->tensor(outputs[0])->bytes / 4);
#if __DEBUG__
    now = std::chrono::system_clock::now();
    std::cout << (now - start_time) / 1ms / 1000.0 << ": " << __LINE__ << ": "
              << __FUNCTION__ << "\n";
#endif
    return o;
  }

  void release_interpreter() {
   auto i = interpreter.release();
   delete i;
  }
 private:
  std::unique_ptr<tflite::Interpreter> interpreter;
};

class diffusion_runner2 : tflite_runner {
 public:
  diffusion_runner2()
      : diffusion_runner2("sd_tflite/sd_diffusion_model_dynamic.tflite") {}

  diffusion_runner2(std::string model_name) : tflite_runner(model_name) {
    interpreter = move(tflite_runner::get_interpreter());

    interpreter->ResizeInputTensor(0, vector<int>({2, 64, 64, 4}));
    interpreter->ResizeInputTensor(1, vector<int>({2, 320}));
    interpreter->ResizeInputTensor(2, vector<int>({2, 77, 768}));
    if (interpreter->AllocateTensors() != kTfLiteOk) {
      std::cout << "failed allocate tensors\n";
    }
  }

  vector<float> diffusion_run(vector<float> latent, vector<float> t_emb,
                              vector<float> u_context, vector<float> context) {
#if __DEBUG__
    auto now = std::chrono::system_clock::now();
    std::cout << (now - start_time) / 1ms / 1000.0 << ": " << __LINE__ << ": "
              << __FUNCTION__ << "\n";
#endif
    // latent, t_emb, context
    std::copy(latent.begin(), latent.end(),
              interpreter->typed_input_tensor<float>(0));
    std::copy(latent.begin(), latent.end(),
              interpreter->typed_input_tensor<float>(0) + latent.size());
    std::copy(t_emb.begin(), t_emb.end(),
              interpreter->typed_input_tensor<float>(1));
    std::copy(t_emb.begin(), t_emb.end(),
              interpreter->typed_input_tensor<float>(1) + t_emb.size());
    std::copy(u_context.begin(), u_context.end(),
              interpreter->typed_input_tensor<float>(2));
    std::copy(context.begin(), context.end(),
              interpreter->typed_input_tensor<float>(2) + context.size());

    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);

#if __DEBUG__
    now = std::chrono::system_clock::now();
    std::cout << (now - start_time) / 1ms / 1000.0 << ": " << __LINE__ << ": "
              << __FUNCTION__ << "\n";
#endif
    if (interpreter->Invoke() != kTfLiteOk) {
      cout << "Failed to invoke tflite!\n";
      exit(-1);
    }
#if __DEBUG__
    now = std::chrono::system_clock::now();
    std::cout << (now - start_time) / 1ms / 1000.0 << ": " << __LINE__ << ": "
              << __FUNCTION__ << "\n";
#endif

    const std::vector<int> outputs = interpreter->outputs();
    auto output = interpreter->typed_output_tensor<float>(0);
    std::vector<float> o(output,
                         output + interpreter->tensor(outputs[0])->bytes / 4);
#if __DEBUG__
    now = std::chrono::system_clock::now();
    std::cout << (now - start_time) / 1ms / 1000.0 << ": " << __LINE__ << ": "
              << __FUNCTION__ << "\n";
#endif
    return o;
  }

 private:
  std::unique_ptr<tflite::Interpreter> interpreter;
};

#endif
