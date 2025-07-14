import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from io import BytesIO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ict', action='store_true', help='is base model ict facekit?')
    return parser.parse_args()


# def quatProduct_batch(q1, q2):
#     r1 = q1[:, 0]  # [B]
#     r2 = q2[:, 0]
#     v1 = torch.stack((q1[:, 1], q1[:, 2], q1[:, 3]), dim=-1)  # [B,3]
#     v2 = torch.stack((q2[:, 1], q2[:, 2], q2[:, 3]), dim=-1)

#     r = r1 * r2 - torch.sum(v1*v2, dim=1)  # [B]
#     v = r1.unsqueeze(1) * v2 + r2.unsqueeze(1) * v1 + torch.cross(v1, v2)  # [B,3]
#     q = torch.stack((r, v[:, 0], v[:, 1], v[:, 2]), dim=1)

#     return q


def quatProduct_batch(q1, r2, v2):
    r1 = q1[:, 0:1]
    v1 = q1[:, 1:]

    # reduceSum is not supported in coreML
    # r = r1 * r2 - torch.sum(v1*v2, dim=1, keepdim=True)

    v3 = v1 * v2
    v3 = v3[:, :1] + v3[:, 1:2] + v3[:, 2:]
    r = r1 * r2 - v3

    v = r1 * v2 + r2 * v1 + torch.cross(v1, v2)
    q = torch.cat((r, v), dim=1)

    return q


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: V,d
        N_v, input_dim = input.shape
        input_ori = input.reshape(N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(N_v, -1)

        return output


class Model(torch.nn.Module):
    def __init__(self, state_dict, lite_model, vertex_size, bs_size) -> None:
        super().__init__()
        for k in state_dict:
            if '.' in k:
                continue
            self.register_buffer(k, torch.zeros_like(state_dict[k]))
        self.deformNet = MLP(
            input_dim=bs_size + 51,
            output_dim=10,
            hidden_dim=128 if lite_model else 256,
            hidden_layers=6
        )
        self.vertex_size = vertex_size
        self.bs_size = bs_size
        self.register_buffer('cond', torch.ones([self.vertex_size, self.bs_size]))

    def forward(self, condition: torch.Tensor):
        # onnx::Tile is not supported in coreml
        # condition = condition.repeat(13453, 1)
        # cond = torch.ones([self.vertex_size, self.bs_size])
        cond = self.cond.clone() * condition

        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, cond), dim=-1)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        deforms = torch.tanh(deforms)
        uv_vertices_deforms = deforms[..., :3]
        # rot_delta_0 = deforms[..., 3:7]
        # # onnx::Exp is not supported in coreml
        # rot_delta_r = torch.exp(rot_delta_0[..., 0:1])
        # rot_delta_v = rot_delta_0[..., 1:]
        # rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        r = torch.pow(2, deforms[..., 3:4] * 1.44269504)
        scale_coef = torch.pow(2, deforms[..., 7:] * 1.44269504)
        uv_vertices = self.default_shape.clone()

        verts_final = uv_vertices + uv_vertices_deforms

        _xyz = verts_final

        rotation = quatProduct_batch(self._rotation_base, r, deforms[..., 4:7])
        # rotation = rot_delta
        scale = self._scaling_base * scale_coef
        # normals = torch.zeros_like(_xyz)
        # f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1)
        # f_rest = self._features_rest.transpose(1, 2).flatten(start_dim=1)
        # opacities = self._opacity

        # _xyz: torch.Size([13453, 3])
        # normals: torch.Size([13453, 3])
        # f_dc: torch.Size([13453, 3])
        # f_rest: torch.Size([13453, 45])
        # opacities: torch.Size([13453, 1])
        # scale: torch.Size([13453, 3])
        # rotation: torch.Size([13453, 4])
        # return torch.cat((_xyz, normals, f_dc, f_rest, opacities, scale, rotation), 1)
        return torch.cat((_xyz, rotation, scale), -1)


def export_color(path, _features_dc, opacities):
    SH_C0 = 0.28209479177387814
    color_rgb = 0.5 + SH_C0 * _features_dc[:, 0]
    # color_rgb = color_rgb.clamp(min=0)
    color_a = 1 / (1 + torch.exp(-opacities))
    color = torch.cat((color_rgb, color_a), -1).detach().cpu().numpy()
    color = (color * 255).clip(0, 255).astype(np.uint8)
    buffer = BytesIO()
    buffer.write(color.tobytes())
    with open(path, "wb") as f:
        f.write(buffer.getvalue())


def main():
    args = parse_args()
    state_dict = torch.load('export/source_dict.pth')['state']

    # _features_rest = state_dict['_features_rest']
    _features_dc = state_dict['_features_dc']
    opacities = state_dict['_opacity']

    trim_dict = {}
    for k in state_dict:
        if k == '_features_rest' or k == '_features_dc' or k == '_opacity':
            continue

        val = state_dict[k]

        if k == 'uv_vertices_shape_embeded':
            val = val.reshape([-1, 51])
        elif k == 'default_shape':
            val = val.reshape([-1, 3])

        trim_dict[k] = val

    model = Model(
        trim_dict,
        False,
        14480 if args.ict else 13453,
        53 if args.ict else 51,
    )
    trim_dict['cond'] = torch.ones([model.vertex_size, model.bs_size])

    model.load_state_dict(trim_dict)

    input_arr = np.loadtxt('export/input.txt')
    x = torch.Tensor(input_arr)[None, ...]
    output = model(x)

    file_name = 'exported0'

    # print(output.shape)
    traced_model = torch.jit.trace(model, x)

    output_path = Path('export/outputs/')
    output_path.mkdir(exist_ok=True)

    export_color(output_path / f'{file_name}_color.bytes', _features_dc, opacities)

    jit_path = output_path / f'{file_name}.pt'
    torch.jit.save(traced_model, jit_path)
    print(f'{jit_path}:', os.path.getsize(jit_path) / 1e6, 'MB')

    traced_model = torch.jit.load(jit_path, map_location='cpu')
    out = traced_model(x)
    print('========== unfused compare to jit ==========')

    if isinstance(output, tuple):
        diff = 0
        for i in range(len(output)):
            print(f'[{i}] torch shape: {output[i].shape}, jit shape{out[i].shape}')
            diff += (output[i] - out[i]).sum()
        print('has multiple output diff sum:', diff)
        isallclose = False
        output_dict = {}
        for i in range(len(output)):
            isallclose |= torch.isclose(output[i], out[i]).all().item()
            output_dict[i] = output[i].detach().cpu().numpy()
        print('is all close:', isallclose)
        output = output_dict
    else:
        diff = output - out
        print(f'diff mean: {diff.mean().item()}, sum: {diff.sum().item()}')
        print('diff:', diff)
        print('is all close:', torch.isclose(output, out).all().item())
        print('output shape:', output.shape)
        output = output.detach().cpu().numpy()

    np.savez(
        str(output_path / f'{file_name}_testing_param.npz'),
        input = x.detach().cpu().numpy(),
        output = output,
    )
    np.savetxt(
        str(output_path / f'{file_name}_input.txt'),
        x.detach().cpu().numpy(),
    )
    if isinstance(output, dict):
        for i in range(len(output)):
            np.savetxt(
                str(output_path / f'{file_name}_output_{i}.txt'),
                output[i],
            )
    else:
        np.savetxt(
            str(output_path / f'{file_name}_output.txt'),
            output,
        )


if __name__ == '__main__':
    main()

