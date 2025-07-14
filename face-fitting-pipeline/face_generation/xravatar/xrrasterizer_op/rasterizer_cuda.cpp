#include <torch/extension.h>
#include <cstdio>
#include <tuple>
// CUDA forward declarations

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterizer_forward_cuda(
    const at::Tensor &face_verts,
    int batch_size,
    int image_size,
    float blur_radius,
    int num_closest);

at::Tensor rasterizer_backward_cuda(
    const at::Tensor &face_verts,
    const at::Tensor &pix_to_face,
    const at::Tensor &grad_bary,
    const at::Tensor &grad_zbuf,
    const at::Tensor &grad_dists);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterizer_forward(
    const at::Tensor &face_verts,
    int batch_size,
    int image_size,
    float blur_radius,
    int faces_per_pixel)
{
    // Use the naive per-pixel implementation
    return rasterizer_forward_cuda(
        face_verts,
        batch_size,
        image_size,
        blur_radius,
        faces_per_pixel);
}

at::Tensor rasterizer_backward(
    const at::Tensor &face_verts,
    const at::Tensor &pix_to_face,
    const at::Tensor &grad_zbuf,
    const at::Tensor &grad_bary,
    const at::Tensor &grad_dists)
{
    return rasterizer_backward_cuda(face_verts,
                                    pix_to_face,
                                    grad_zbuf,
                                    grad_bary,
                                    grad_dists);
}

PYBIND11_MODULE(rasterizer, m)
{
    m.def("rasterizer_forward", &rasterizer_forward, "DRAW_NORMAL (CUDA)");
    m.def("rasterizer_backward", &rasterizer_backward, "DRAW_NORMAL (CUDA)");
}