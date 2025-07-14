import argparse
import torch
import onnx
import numpy as np
import torch.onnx
from pathlib import Path

# use conda torch2
# next use conda tf2 to run: onnx2tf -i .\saved_model\temp\myslate_11_css_iphone_mediapipe_51_ss.onnx -ois 1,51
parser = argparse.ArgumentParser()
parser.add_argument('input', help='input model path')
parser.add_argument('npz', help='input npz path (contains expected input & output)')
args = parser.parse_args()

input_file = Path(args.input)
model = torch.jit.load(str(input_file), map_location='cpu')

d = np.load(args.npz, allow_pickle=True)

x = torch.Tensor(d['input'])

if d['output'].dtype == np.object_:
    output_dict = d['output'][()]
    output = []
    for k in output_dict:
        output.append(torch.Tensor(output_dict[k]))
else:
    output = torch.Tensor(d['output'])

# Input to the model
# x = torch.randn(1, 3, 224, 224, requires_grad=True)
torch_out = model(x)

output_path = Path(f'./export/outputs/{input_file.stem}.onnx')
output_path.parent.mkdir(exist_ok=True, parents=True)
output_path = str(output_path)

output_names = []
# dynamic_axes = None
dynamic_axes = {'input': {0: 'batch_size'}}
if isinstance(output, list):
    for i in range(len(output)):
        output_names.append(f'output_{i}')
        dynamic_axes[f'output_{i}'] = {0: 'batch_size'}
else:
    output_names.append('output')
    dynamic_axes[f'output'] = {0: 'batch_size'}


# Export the model
torch.onnx.export(
    model,                          # model being run
    x,                              # model input (or a tuple for multiple inputs)
    output_path,                    # where to save the model (can be a file or file-like object)
    export_params=True,             # store the trained parameter weights inside the model file
    opset_version=13,               # the ONNX version to export the model to
    do_constant_folding=True,       # whether to execute constant folding for optimization
    input_names=['input'],          # the model's input names
    output_names=output_names,      # the model's output names
    # dynamic_axes=dynamic_axes,      # variable length axes
)

print(f'exported {output_path}')

onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)


import onnxruntime
ort_session = onnxruntime.InferenceSession(output_path, providers=["CPUExecutionProvider"])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

if isinstance(output, list):
    print('has multiple output:')
    for i in range(len(output)):
        np.testing.assert_allclose(to_numpy(torch_out[i]), ort_outs[i], rtol=1e-03, atol=1e-05)
        print(f'output_{i}: {ort_outs[i].shape}, {ort_outs[i].dtype}')
else:
    print(ort_outs[0].shape, ort_outs[0].dtype)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

try:
    from onnxconverter_common import float16  # noqa
    model_fp16 = float16.convert_float_to_float16(onnx_model)
    onnx.save(model_fp16, f"./outputs/{input_file.stem}_fp16.onnx")
except Exception:
    print('onnxconverter_common not installed not outputing fp16 onnx file')
