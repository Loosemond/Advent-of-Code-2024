#include <cstring>
#include <fstream>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

int main() {
  std::ifstream file("test_input.txt");
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

  // Allocate device memory for line start positions
  std::vector<size_t> line_offsets;
  size_t total_chars = 0;
  for (const auto &s : secret) {
    line_offsets.push_back(total_chars);
    total_chars += s.size() + 1; // +1 for null terminator
  }

  sycl::queue q;
  char *device_buffer = sycl::malloc_shared<char>(total_chars, q);
  size_t *device_offsets = sycl::malloc_shared<size_t>(num_lines, q);
  int *results = sycl::malloc_shared<int>(num_lines, q);

  // Copy data to device memory
  size_t offset = 0;
  for (size_t i = 0; i < num_lines; ++i) {
    std::memcpy(device_buffer + offset, secret[i].c_str(),
                secret[i].size() + 1);
    offset += secret[i].size() + 1;
    device_offsets[i] = line_offsets[i];
  }
  for (size_t i = 0; i < num_lines; ++i) {
    std::cout << "Line " << i + 1 << " : " << device_buffer[device_offsets[i]]
              << "\n";
    // sum += results[i];
  }
  // Parallel read input to 2 arrays

  // Parallel sort of 2 arrays at the same time ?
  q.parallel_for(num_lines, [=](sycl::id<1> idx) {
     size_t start = device_offsets[idx];
     char *line_start = &device_buffer[start];
     int found_first = false;
     int last = -1;
     char a;
     int l;
     int n{static_cast<int>(num_lines)};
     while (*line_start) {
       if ((*line_start >= '0' && *line_start <= '9')) {
         printf("id %i - %c\n", static_cast<int>(idx), *line_start);
         for (int k = 2; k <= n; k *= 2) {
           for (int j = k / 2; j > 0; j /= 2) {
             for (int i = 0; i < n; i++) {
               l = i ^ j;
               if (l > i) {
                 if (((i & k) == 0) && (device_buffer[device_offsets[i]] >
                                        device_buffer[device_offsets[l]]) ||
                     ((i & k) != 0) && (device_buffer[device_offsets[i]] <
                                        device_buffer[device_offsets[l]])) {
                   a = device_offsets[i];
                   device_offsets[i] = device_offsets[l];
                   device_offsets[l] = a;
                 }
               }
             }
           }
         }
       }
       //  *line_start = ' '; // Convert to uppercasedevice_offsets
       ++line_start;
     }
   }).wait();

  // Print the results
  std::uint64_t sum{};
  for (size_t i = 0; i < num_lines; ++i) {
    std::cout << "Line " << i + 1 << " : " << device_buffer[device_offsets[i]]
              << "\n";
    // sum += results[i];
  }
  printf("Total: %lu\n", sum);

  sycl::free(device_buffer, q);
  sycl::free(device_offsets, q);
  sycl::free(results, q);

  return 0;
}
