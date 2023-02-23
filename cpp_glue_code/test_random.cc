#include <iostream>
#include <random>
#include <vector>

std::vector<float> get_normal(unsigned numbers, unsigned seed=5, float mean=0.0, float stddev=1.0)
{
  std::default_random_engine generator(seed);
  std::normal_distribution<float> distribution (mean, stddev);

  std::vector<float> d;
  for (int i=0; i < numbers; i++)
    d.push_back(distribution(generator));
  
  return d;
}

int main()
{
  auto dis = get_normal(64*64*4);
  for (int i=0; i < 10; i++)
    std::cout << dis[i] << "\n";
  return 0;
}
