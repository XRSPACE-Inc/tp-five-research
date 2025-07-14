printf "run in root!\n"
python -m export.export_to_unity_ict --idname myslate_11_css_iphone --checkpoint dataset/myslate_11_css_iphone/log/ckpt/chkpnt150000.pth -med -ss || { echo 'failed' ; exit 1; }
printf "======================= export export/source_dict.pth =======================\n"
python -m export.export --ict || { echo 'failed' ; exit 1; }
printf "======================= export to jit done =======================\n"

python -m export.jit2onnx export/outputs/exported0.pt export/outputs/exported0_testing_param.npz || { echo 'failed' ; exit 1; }
printf "======================= export to onnx done =======================\n"

onnxsim export/outputs/exported0.onnx export/outputs/exported.onnx || { echo 'failed' ; exit 1; }
printf "======================= onnx simplifier done =======================\n"

# python -m onnxruntime.tools.check_onnx_model_mobile_usability 'export/outputs/exported.onnx'
python -m onnxruntime.tools.check_onnx_model_mobile_usability 'export/outputs/exported.onnx' --log_level debug || { echo 'failed' ; exit 1; }
printf "======================= done check onnx model =======================\n"

python -m onnxruntime.tools.convert_onnx_models_to_ort export/outputs/exported.onnx --optimization_style Runtime --output_dir 'export/outputs/' || { echo 'failed' ; exit 1; }
printf "======================= done convert onnx =======================\n"
