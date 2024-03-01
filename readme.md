# Run TF/PyTorch/JAX models for MRI reconstruction with BART and TensorRT

This repository shows how to use bart to inference various deep learning models that are implemented in different frameworks (PyTorch, TensorFlow, JAX) using TensorRT as backend. Models are converted to the ONNX format and then converting them to TensorRT engines using the bart trt command and then engines will be executed with bart trt too.

## Prerequisites

- BART `TensorRT` branch.
- Python 3.x
- PyTorch (for PyTorch models)
- TensorFlow (for TensorFlow models)
- JAX (for JAX models)
- TensorRT
- onnx

## Usage

To run inference using all models in the repository, execute the provided shell script `run_all.sh`. This script runs inference for PyTorch, TensorFlow, and JAX models, and converts the output to images using the `bart toimg` command.

Before run this script,

- Make sure BART is compile correctly
- Make [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar) is correctly installed.
- Make sure env varibles are set up correctly

   ```shell 
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT/lib
   ```
- Ensure that input files are correctly specified and available in the specified paths.

### Script Details

The shell script `run_all.sh` follows these steps:

1. Sets the shell to exit immediately if any command fails (`set -e`).
2. Executes the `bart toimg` command to convert a dataset (`data/grd`) to an image (`data/brain.png`).
3. Sets environment variables for TensorRT and TensorFlow.
4. Runs inference for PyTorch model:
   - Converts the PyTorch model to ONNX format (`gaussian_blur_pytorch.py`).
   - Converts the ONNX model to TensorRT engine and execute it (`bart trt`).
   - Converts the output to images (`bart toimg`).
5. Runs inference for TensorFlow model:
   - Converts the TensorFlow model to ONNX format (`gaussian_blur_tf.py`).
   - Converts the ONNX model to TensorRT engine and execute it (`bart trt`).
   - Converts the output to images (`bart toimg`).
6. Runs inference for JAX model:
   - Converts the JAX model to ONNX format (`gaussian_blur_jax.py`).
   - Converts the ONNX model to TensorRT engine and execute it (`bart trt`).
   - Converts the output to images (`bart toimg`).


