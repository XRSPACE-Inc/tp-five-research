from torch.autograd import Function
try:
    import xrrasterizer.cuda.rasterizer as rc
except Exception:
    import os
    from torch.utils.cpp_extension import load
    import datetime

    st = datetime.datetime.now()
    module_path = os.path.dirname(__file__)
    rc = load('rasterizer',
              sources=[
                  os.path.join(module_path, 'rasterizer_cuda.cpp'),
                  os.path.join(module_path, 'rasterizer_cuda_kernel.cu')
              ],
              verbose=False)
    # print("xrrasterizer loaded in", (datetime.datetime.now() - st).total_seconds(), "s")


def rasterizer(face_verts,
               batch_size,
               image_size,
               blur_radius=0.01,
               faces_per_pixel=16):
    return RasterizerFunction.apply(
        face_verts, batch_size, image_size, blur_radius, faces_per_pixel)


class RasterizerFunction(Function):
    @staticmethod
    def forward(ctx,
                face_verts,
                batch_size,
                image_size,
                blur_radius=0.01,
                faces_per_pixel=None):
        if faces_per_pixel is None:
            faces_per_pixel = int(max(10000, face_verts.size(0) / 5))
        pix_to_face, zbuf, barycentric_coords, dists = rc.rasterizer_forward(
            face_verts,
            batch_size,
            image_size,
            blur_radius,
            faces_per_pixel)
        ctx.save_for_backward(face_verts, pix_to_face)
        return pix_to_face, zbuf, barycentric_coords, dists

    @staticmethod
    def backward(ctx,
                 grad_pix_to_face,
                 grad_zbuf,
                 grad_barycentric_coords,
                 grad_dists):
        face_verts, pix_to_face = ctx.saved_tensors
        grad_face_verts = rc.rasterizer_backward(
            face_verts,
            pix_to_face,
            grad_zbuf,
            grad_barycentric_coords,
            grad_dists)
        return grad_face_verts, None, None, None, None
