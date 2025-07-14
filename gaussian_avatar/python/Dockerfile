FROM avatar2d_base:latest

RUN pip install opencv-python lpips plyfile scikit-image loguru yacs

RUN pip install git+https://github.com/mattloper/chumpy

COPY pytorch3d/ /pytorch3d
COPY gaussian-splatting/submodules/ /gaussian-splatting/submodules

WORKDIR /pytorch3d

RUN pip install .

WORKDIR /gaussian-splatting/submodules/simple-knn

RUN pip install .

WORKDIR /gaussian-splatting/submodules/diff-gaussian-rasterization

RUN pip install .

WORKDIR /workspace

