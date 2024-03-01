set -e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT/lib
bart toimg data/grd data/brain.png

export NV_TENSORRT_LOG_LEVEL=WARNING
export TF_CPP_MIN_LOG_LEVEL=3

if [ 1 == 1 ] ; then
    echo "Running TRT Inference for pytorch model"
    python gaussian_blur_pytorch.py data/grd data/pytorth_blur data/pytorth_blur_model.onnx
    bart trt data/pytorth_blur_model.onnx data/grd data/trt_blur_pytorch
    bart toimg data/trt_blur_pytorch data/trt_blur_pytorch
fi


if [ 1 == 1 ] ; then
    echo "Running TRT Inference for tensorflow model"
    python gaussian_blur_tf.py data/grd data/tf_blur data/tf_blur_model.onnx
    bart trt data/tf_blur_model.onnx data/grd data/trt_blur_tf
    bart toimg data/trt_blur_tf data/trt_blur_tf
fi


if [ 1 == 1 ] ; then
    echo "Running TRT Inference for jax model"
    python gaussian_blur_jax.py data/grd data/jax_blur data/jax_blur_model.onnx
    bart trt data/jax_blur_model.onnx data/grd data/trt_blur_jax
    bart toimg data/trt_blur_jax data/trt_blur_jax
fi
