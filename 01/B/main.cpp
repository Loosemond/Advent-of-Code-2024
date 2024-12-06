#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

int main() {
  std::ifstream file("input.txt");
  if (!file.is_open()) {
    std::cerr << "Error: Could not open the file.\n";
    return 1;
  }

  // Read lines from the file
  std::vector<std::string> secret;
  std::string line;
  while (std::getline(file, line)) {
    secret.push_back(line);
  }
  file.close();

  size_t num_lines = secret.size();
  size_t num_lines_sq{};
  int numbers_to_skip{0};

  // Allocate device memory for line start positions
  std::vector<size_t> line_offsets;
  size_t total_chars = 0;
  for (const auto &s : secret) {
    line_offsets.push_back(total_chars);
    total_chars += s.size() + 1; // +1 for null terminator
  }

  num_lines_sq = std::pow(2, (int)std::log2(num_lines) + 1);
  numbers_to_skip = num_lines_sq - num_lines;
  printf("Original:%zu, Sq:%zu\n", num_lines, num_lines_sq);
  sycl::queue q;
  char *device_buffer = sycl::malloc_shared<char>(total_chars, q);
  size_t *device_offsets = sycl::malloc_shared<size_t>(num_lines, q);
  int *left_side = sycl::malloc_shared<int>(num_lines_sq, q);
  int *right_side = sycl::malloc_shared<int>(num_lines_sq, q);
  int *results = sycl::malloc_shared<int>(num_lines_sq, q);

  // Copy data to device memory
  size_t offset = 0;
  for (size_t i = 0; i < num_lines; ++i) {
    std::memcpy(device_buffer + offset, secret[i].c_str(),
                secret[i].size() + 1);
    offset += secret[i].size() + 1;
    device_offsets[i] = line_offsets[i];
  }
  // for (size_t i = 0; i < num_lines_sq; ++i) {
  //   left_side[i] = -1;
  //   right_side[i] = -1;
  // }
  // Parallel read input to 2 arrays
  q.parallel_for(num_lines, [=](sycl::id<1> idx) {
     //  printf("Started read\n");
     size_t start = device_offsets[idx];
     char *line_start = &device_buffer[start];
     int found_first = false;
     int last = -1;
     int n_found{0};
     while (*line_start != ' ') {

       if ((*line_start >= '0' && *line_start <= '9')) {
         if (n_found > 0) {
           left_side[idx] = left_side[idx] * 10;
         }
         left_side[idx] = static_cast<int>(*line_start - '0') + left_side[idx];
         n_found++;
         //  printf("Left: %i, N found %i, string: %c\n", left_side[idx],
         //  n_found,
         //         *line_start);
       }
       ++line_start;
     }
     line_start += 2;
     n_found = 0;
     while (*line_start) {
       if ((*line_start >= '0' && *line_start <= '9')) {
         if (n_found > 0) {
           right_side[idx] = right_side[idx] * 10;
         }
         right_side[idx] =
             static_cast<int>(*line_start - '0') + right_side[idx];
         n_found++;
         //  printf("Right: %i\n", right_side[idx]);
       }
       ++line_start;
     }
   }).wait();

  printf("Finished read\n");

  q.parallel_for(num_lines, [=](sycl::id<1> idx) {
     for (int i{0}; i < num_lines_sq; i++) {
       if (left_side[idx] == right_side[i]) {
         results[idx] += left_side[idx];
         //  printf("results[%i] = %i\n", (int)idx, results[idx]);
       }
     }
   }).wait();

  printf("Finished left\n");

  // Print the results
  int sum{0};
  int r{0};
  for (size_t i = 0; i < num_lines; ++i) {
    // std::cout << "Left: " << left_side[i] << " ";
    // std::cout << "Right: " << right_side[i] << "\n";
    // printf("%i\n", results[i]);
    sum += results[i];
  }

  printf("Result: %i\n", sum);

  sycl::free(device_buffer, q);
  sycl::free(device_offsets, q);
  sycl::free(results, q);

  return 0;
}