bart toimg data/grd brain.png
python gaussian_blur.py data/grd blurred_brain data/model.onnx
bart trt data/model.onnx data/first_engine.trt data/grd data/out
bart toimg data/out out_blurred_brain.png
