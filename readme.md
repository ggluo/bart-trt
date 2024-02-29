# Bart With TensorRT

This repository shows how to inference deep learning models with BART using TensorRT as backend.

## Prerequisites

- BART `TensorRT` branch.
- Python with necessary libraries for `gaussian_blur.py` script.
- TensorRT installed for optimized inference.

## Usage


1. **create a computational graph**.
   ```
   python gaussian_blur.py data/grd blurred_brain data/model.onnx
   ```

2. **bart trt**: Performs graph inference using TensorRT.
   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT/lib
   bart trt data/model.onnx data/first_engine.trt data/grd data/out
   ```

4. **bart toimg**: Converts the output data file to an image file.
   ```
   bart toimg data/out out_blurred_brain.png
   ```

## Notes

- Ensure that input files are correctly specified and available in the specified paths.
- Make [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar) is correctly installed.

