## Install DPC++
https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp

## Configure DPC++
https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2024-1/use-the-setvars-and-oneapi-vars-scripts-with-linux.html

```bash
source ~/intel/oneapi/setvars.sh
```


To develop and run applications for specific GPUs, you must first install the corresponding drivers or plug-ins:

- To use an Intel GPU, [install](https://dgpu-docs.intel.com/installation-guides/index.html) the latest Intel GPU drivers.
- To use an AMD GPU, [install](https://developer.codeplay.com/products/oneapi/amd/guides/) the oneAPI for AMD GPUs plugin from Codeplay.
- To use an NVIDIA GPU, [install](https://developer.codeplay.com/products/oneapi/nvidia/guides/) the oneAPI for NVIDIA GPUs plugin from Codeplay.


## Nvidia
You need to install cuda and the onepi lib to support nvidia

### Build

```bash
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda main.cpp -o aoc
```

### Run
```bash
ONEAPI_DEVICE_SELECTOR="ext_oneapi_cuda:*" SYCL_PI_TRACE=1 ./aoc
```

## Compiling for multiple targets
```bash 
icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda,spir64 \
        -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1030 \
        -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_80 \
        -o sycl-app sycl-app.cpp
```

