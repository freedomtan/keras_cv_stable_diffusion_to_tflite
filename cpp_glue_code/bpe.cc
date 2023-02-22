#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

using namespace std;

string vocab_file = "vocab.txt";
string merges_file = "merges.txt";

std::map<std::string, int> vocab;
std::vector<tuple<string, string>> merges;

vector<string> get_tokens(string line) {
  stringstream c(line);

  vector<string> tokens;
  string intermediate;

  while (getline(c, intermediate, ' ')) {
    tokens.push_back(intermediate);
  }
  return tokens;
}

void dump_string_vector(vector<string> to_dump) {
  for (int i = 0; i < to_dump.size(); i++) {
    if (i != (to_dump.size() - 1))
      cout << to_dump[i] << ", ";
    else
      cout << to_dump[i] << "\n";
  }
}

set<tuple<string, string>> get_pairs(vector<string> token) {
  set<tuple<string, string>> pairs;

  for (int i = 0; i < token.size() - 1; i++) {
    pairs.insert(pairs.end(), make_tuple(token[i], token[i + 1]));
  }
  return pairs;
}

string concat_vector_to_string(vector<string> to_concat) {
  string word_to_return;

  for (auto i = to_concat.begin(); i < to_concat.end(); i++) {
    word_to_return.append(*i);
  }
  return word_to_return;
}

string bpe_encode(string token) {
  vector<string> word;
  for (auto i : token) {
    word.push_back(string(1, i));
  }
  word[token.size() - 1] = word[token.size() - 1] + "</w>";
  set<tuple<string, string>> pairs = get_pairs(word);
  if (pairs.size() == 0) {
    return token + "</w>";
  }

  while (true) {
    map<tuple<string, string>, int> can_merges;

    for (tuple<string, string> pair : pairs) {
      for (int i = 0; i < merges.size(); i++) {
        if (merges[i] == pair) {
          can_merges[pair] = i;
        }
      }
    }
    if (can_merges.size() == 0) break;

    auto min = min_element(can_merges.begin(), can_merges.end(),
                           [](const auto& lhs, const auto& rhs) {
                             return lhs.second < rhs.second;
                           });
    auto first = get<0>(min->first);
    auto second = get<1>(min->first);

    vector<string> new_word;
    int i = 0;
    while (i < word.size()) {
      vector<string> remaining(word.begin() + i, word.end());

      auto found_i = find(word.begin() + i, word.end(), first);
      auto found = -1;
      if (found_i != word.end()) found = distance(word.begin(), found_i);

      if (found != -1) {
        vector<string> prefix(word.begin() + i, word.begin() + found);
        new_word.insert(new_word.end(), prefix.begin(), prefix.end());

        if ((i < (word.size() - 1)) && word[found + 1] == second) {
          new_word.push_back(word[found] + word[found + 1]);
          i = found + 2;
        } else {
          new_word.push_back(word[found]);
          i = found + 1;
        }
      } else {
        new_word.insert(new_word.end(), remaining.begin(), remaining.end());
        break;
      }
    }

    word = new_word;
    pairs = get_pairs(new_word);
  }

  string word_to_return;
  if (word.size() == 1) {
    word_to_return = word[0];
  } else {
    for (auto i = word.begin(); i < word.end() - 1; i++) {
      word_to_return = word_to_return + (*i + " ");
    }
    word_to_return = word_to_return + word[word.size()-1];
  }

  cout << "word_to_return: " << word_to_return << "\n";
  return word_to_return;
}

#define START_OF_TEXT 49406
#define END_OF_TEXT 49407

vector<int> encode(string line) {
  // clean up whitespace, replacing all consective ws with single " "
  regex ws("\\s+");
  line = regex_replace(line, ws, " ");

  // to lower
  transform(line.begin(), line.end(), line.begin(), ::tolower);
  auto tokens = get_tokens(line);

  vector<int> codes;
  codes.push_back(START_OF_TEXT);
  for (auto t : tokens) {
    auto returned = get_tokens(bpe_encode(t));
    for (auto i=0; i < returned.size(); i++) {
      codes.push_back(vocab[returned[i]]);
    }
  }
  codes.push_back(END_OF_TEXT);

  return codes;
}

#ifdef __TEST_BPE__
int main(int argc, char *argv[]) {
  std::ifstream merges_stream(merges_file, std::ifstream::binary);
  std::ifstream vocab_stream(vocab_file, std::ifstream::binary);

  string k, v;
  auto index = 0;

  while (vocab_stream >> v >> index) {
    vocab[v] = index;
  }

  while (merges_stream >> k >> v) {
    merges.push_back(make_tuple(k, v));
  }

  cout << "vocab size: " << vocab.size() << "\n";
  cout << "merges size: " << merges.size() << "\n";

  string prompt = "a photo of an astronaut riding a horse on Mars";
  if (argc == 2) 
    prompt = argv[1];

  auto encoded = encode(prompt);
  cout << "[";
  for (auto i = encoded.begin(); i < encoded.end(); i++) {
     if (i != (encoded.end() - 1))
       cout << *i << ", ";
     else
       cout << *i;
  }
  cout << "]" << "\n";
}
#endif
