#include <getopt.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <valarray>

#include "bpe.h"
#include "scheduling_util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tflite_util.h"

#if __DEBUG__
using namespace std::literals;
std::chrono::time_point<std::chrono::system_clock> start_time;
#endif

vector<float> run_text_encoder(vector<int> encoded, vector<int> pos_ids) {
  vector<float> empty;
#if __DEBUG__
  std::cout << __LINE__ << ": " << __FUNCTION__ << "\n";
#endif
  auto model = tflite::FlatBufferModel::BuildFromFile(
      "sd_tflite/sd_text_encoder_dynamic.tflite");
  if (model == nullptr) {
    cout << "failed to load model "
         << "tflite/sd_text_encoder.tflite\n";
    return empty;
  }

  std::unique_ptr<tflite::Interpreter> interpreter;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    cout << "failed to build interpreter\n";
    return empty;
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    cout << "failed allocate tensors\n";
    return empty;
  }

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  std::copy(encoded.begin(), encoded.end(),
            interpreter->typed_input_tensor<int>(0));
  std::copy(pos_ids.begin(), pos_ids.end(),
            interpreter->typed_input_tensor<int>(1));

  if (interpreter->Invoke() != kTfLiteOk) {
    cout << "Failed to invoke tflite!\n";
    exit(-1);
  }
  auto output = interpreter->typed_tensor<float>(outputs[0]);
  std::vector<float> o(output,
                       output + interpreter->tensor(outputs[0])->bytes / 4);
  return o;
}

std::vector<float> get_normal(unsigned numbers, unsigned seed = 5,
                              float mean = 0.0, float stddev = 1.0) {
  std::default_random_engine generator(seed);
  std::normal_distribution<float> distribution(mean, stddev);

  std::vector<float> d;
  for (unsigned i = 0; i < numbers; i++) d.push_back(distribution(generator));

  return d;
}

vector<float> run_decoder(vector<float> latent) {
#if __DEBUG__
  std::cout << __LINE__ << ": " << __FUNCTION__ << "\n";
#endif
  vector<float> empty;
  auto model = tflite::FlatBufferModel::BuildFromFile(
      "sd_tflite/sd_decoder_dynamic.tflite");

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    cout << "failed allocate tensors\n";
    return empty;
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    cout << "failed allocate tensors\n";
    return empty;
  }

  std::copy(latent.begin(), latent.end(),
            interpreter->typed_input_tensor<float>(0));

  if (interpreter->Invoke() != kTfLiteOk) {
    cout << "Failed to invoke tflite!\n";
    exit(-1);
  }

  const std::vector<int> outputs = interpreter->outputs();
  auto output = interpreter->typed_output_tensor<float>(0);
  std::vector<float> o(output,
                       output + interpreter->tensor(outputs[0])->bytes / 4);
  return o;
}

void dump_vector(vector<float> v) {
  for (int i = 0; i < 16; i++) {
    cout << v[i] << "\t";
  }
  cout << "\n";
}

int main(int argc, char *argv[]) {
  unsigned seed = 5;
  int num_steps = 25;
  int ch;
  /* options descriptor */
  static struct option longopts[] = {
      {"seed", required_argument, NULL, 's'},
      {"num_steps", required_argument, NULL, 'n'},
      {NULL, 0, NULL, 0}};

  while ((ch = getopt_long(argc, argv, "s:n:", longopts, NULL)) != -1) {
    switch (ch) {
      case 's':
        seed = atoi(optarg);
        break;
      case 'n':
        num_steps = atoi(optarg);
        break;
      default:
        // usage();
        break;
    }
  }
  argc -= optind;
  argv += optind;

  string prompt = "a photo of an astronaut riding a horse on Mars";
  if (argc == 1) prompt = argv[0];

  bpe bpe_encoder;

  auto encoded = bpe_encoder.encode(prompt);
  auto pos_ids = bpe_encoder.position_ids();

  auto encoded_text = run_text_encoder(encoded, pos_ids);
  auto unconditional_text =
      run_text_encoder(bpe_encoder.unconditioned_tokens(), pos_ids);

  float unconditional_guidance_scale = 7.5;
  auto noise = get_normal(64 * 64 * 4, seed);
  auto latent = noise;
  auto timesteps = get_timesteps(1, 1000, 1000 / num_steps);
  auto alphas_tuple = get_initial_alphas(timesteps);
  auto alphas = get<0>(alphas_tuple);
  auto alphas_prev = get<1>(alphas_tuple);

  auto diffusion = diffusion_runner();

#if __DEBUG__
  start_time = std::chrono::system_clock::now();
#endif
  for (int i = timesteps.size() - 1; i >= 0; i--) {
    cout << "step " << timesteps.size() - 1 - i << "\n";
#if __DEBUG__
  auto now = std::chrono::system_clock::now();
  std::cout << (now - start_time) / 1ms / 1000.0 << ": " << __LINE__ << ": "
            << __FUNCTION__ << "\n";
#endif
    auto latent_prev = latent;
    auto t_emb = get_timestep_embedding(timesteps[i]);
#if __DEBUG__
  now = std::chrono::system_clock::now();
  std::cout << (now - start_time) / 1ms / 1000.0 << ": " << __LINE__ << ": "
            << __FUNCTION__ << "\n";
#endif
    auto unconditional_latent = diffusion.diffusion_run(latent, t_emb, unconditional_text);
    latent = diffusion.diffusion_run(latent, t_emb, encoded_text);
#if __DEBUG__
  now = std::chrono::system_clock::now();
  std::cout << (now - start_time) / 1ms / 1000.0 << ": " << __LINE__ << ": "
            << __FUNCTION__ << "\n";
#endif

    std::valarray<float> l(latent.data(), latent.size());
    std::valarray<float> l_prev(latent_prev.data(), latent_prev.size());
    std::valarray<float> u(unconditional_latent.data(),
                           unconditional_latent.size());
    l = u + unconditional_guidance_scale * (l - u);
    auto a_t = alphas[i];
    auto a_prev = alphas_prev[i];

    auto prev_x0 = (l_prev - sqrtf(1.0 - a_t) * l) / sqrtf(a_t);
    l = (l * sqrtf(1.0 - a_prev) + sqrtf(a_prev) * prev_x0);
    latent.assign(std::begin(l), std::end(l));
  }
  auto decoded = run_decoder(latent);
  std::valarray<float> d(decoded.data(), decoded.size());
  d = (d + 1) / 2 * 255;
  vector<uint8_t> decoded_uint8;
  for (auto e : d) {
    if (e > 255.0) e = 255;
    if (e < 0.0) e = 0;
    decoded_uint8.push_back((uint8_t)e);
  }

  std::ofstream rgb_file("decoded.raw", std::ios::out | std::ofstream::binary);
  std::copy(decoded_uint8.begin(), decoded_uint8.end(),
            std::ostreambuf_iterator<char>(rgb_file));
}
