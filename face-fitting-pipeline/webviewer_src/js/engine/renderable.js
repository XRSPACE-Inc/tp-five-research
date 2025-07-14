/* eslint-disable max-len */
import { setFrameDirty } from './utils.js';
import { gl } from './gl.js';
import { Shader } from './shader.js';
import { mat4, mat3 } from '../lib/gl-matrix/index.js';
import { Material } from './material.js';

const DrawingMode = Object.freeze({
  Normal: 0, WireFrame: 1, ShadingWithWireFrame: 2, Points: 3, Lines: 4,
});

export default class Renderable {
  constructor(material) {
    this.enabled = true;
    this.material = material;
    this.first = 0;
    this.size = 0;
    this.wireframe_size = 0;
    this.buffer = gl.createBuffer();
    this.wireframe_buffer = gl.createBuffer();
    this.attributeDatas = [];
    this.wireframeAttributeDatas = [];
    this.gameNode = null;
    this.drawingMode = DrawingMode.Normal;
    this.ui = false;
    this.castShadow = true;
  }

  drawErrorType(matrix, viewMat, projMat) {
    if (this.errorShader === undefined) {
      // eslint-disable-next-line no-console
      console.error('drawing error type material');
      this.errorShader = new Shader(Shader.FindShaderSource('error'));
    }
    this.errorShader.bind();
    this.errorShader.setMatrix4('viewMatrix', viewMat);
    this.errorShader.setMatrix4('projectionMatrix', projMat);
    this.errorShader.setMatrix4('modelMatrix', matrix);

    this.setAttributesForShader(this.errorShader, this.buffer, this.attributeDatas);

    gl.drawArrays(gl.TRIANGLES, this.first, this.size);

    this.disableAttribs(this.errorShader, this.attributeDatas);

    return this;
  }

  drawPointCloud(matrix, viewMat, projMat) {
    if (!this.pointCloudShader) {
      this.pointCloudShader = new Shader(Shader.FindShaderSource('pointcloud'));
    }
    this.pointCloudShader.bind();
    this.pointCloudShader.setMatrix4('viewMatrix', viewMat);
    this.pointCloudShader.setMatrix4('projectionMatrix', projMat);
    this.pointCloudShader.setMatrix4('modelMatrix', matrix);

    this.setAttributesForShader(this.pointCloudShader, this.buffer, this.attributeDatas);

    gl.drawArrays(gl.POINTS, this.first, this.size);

    this.disableAttribs(this.pointCloudShader, this.attributeDatas);

    this.pointCloudShader.unbind();

    return this;
  }

  drawWireframe(matrix, viewMat, projMat) {
    if (this.wireframeMaterial === undefined) {
      this.wireframeMaterial = new Material('debug');
      this.wireframeMaterial.setUniformData('color', [0.0, 1.0, 0.0, 1.0]);
    }
    const { shader } = this.wireframeMaterial;

    shader.bind();
    if (this.wireframeMaterial.hasKey('viewMatrix')) this.wireframeMaterial.setUniformData('viewMatrix', viewMat);
    if (this.wireframeMaterial.hasKey('projectionMatrix')) this.wireframeMaterial.setUniformData('projectionMatrix', projMat);
    if (this.wireframeMaterial.hasKey('modelMatrix')) this.wireframeMaterial.setUniformData('modelMatrix', matrix);
    this.wireframeMaterial.setUniformDatas();

    this.setAttributesForShader(shader, this.wireframe_buffer, this.wireframeAttributeDatas);

    gl.drawArrays(gl.LINES, this.first, this.wireframe_size);

    this.disableAttribs(shader, this.wireframeAttributeDatas);

    shader.unbind();

    return this;
  }

  drawWithMaterial(matrix, camera, Light, viewMat, projMat, material) {
    const mat = material || this.material;
    const hasLight = mat.hasIfDefine('LIGHT_COUNT');

    if (hasLight) {
      // mat.addOrUpdateDefine('SHADOW_MAP_SIZE_X', 512);
      // mat.addOrUpdateDefine('SHADOW_MAP_SIZE_Y', 512);
      mat.addOrUpdateDefine('LIGHT_COUNT', Light.lightCount());
    }

    const { shader } = mat;

    if (shader) {
      shader.bind();

      if (mat.hasKey('uCameraPosition')) mat.setUniformData('uCameraPosition', camera.transform.position);
      if (mat.hasKey('cameraResolution')) mat.setUniformData('cameraResolution', camera.resolution);
      if (mat.hasKey('viewMatrix')) mat.setUniformData('viewMatrix', viewMat);
      if (mat.hasKey('projectionMatrix')) mat.setUniformData('projectionMatrix', projMat);
      if (mat.hasKey('modelMatrix')) mat.setUniformData('modelMatrix', matrix);

      if (hasLight) {
        Light.SetLightUniformDatas(mat);
      }

      if (mat.hasKey('modelViewMatrix')) {
        const modelViewMatrix = mat4.multiply(mat4.create(), viewMat, matrix);
        mat.setUniformData('modelViewMatrix', modelViewMatrix);
      }
      if (mat.hasKey('normalMatrix')) {
        const modelViewMatrix = mat4.multiply(mat4.create(), viewMat, matrix);
        const normalMatrix = mat3.normalFromMat4(mat3.create(), modelViewMatrix);
        mat.setUniformData('normalMatrix', normalMatrix);
      }

      mat.setUniformDatas();
      mat.onRender();

      this.setAttributesForShader(shader, this.buffer, this.attributeDatas);
    }

    if (this.drawingMode === DrawingMode.Lines) {
      gl.lineWidth(0.01);
      gl.drawArrays(gl.LINES, this.first, this.size);
    } else {
      gl.drawArrays(gl.TRIANGLES, this.first, this.size);
    }

    if (shader) {
      this.disableAttribs(shader, this.attributeDatas);
      shader.unbind();
    }

    return this;
  }

  draw(matrix, camera, Light, viewMat, projMat, material) {
    if (this.enabled === false) return this;
    if (!material && !this.material) return this.drawErrorType(matrix, viewMat, projMat);

    // render point cloud
    if (this.drawingMode === DrawingMode.Points) return this.drawPointCloud(matrix, viewMat, projMat);
    // render
    if (this.drawingMode !== DrawingMode.WireFrame) this.drawWithMaterial(matrix, camera, Light, viewMat, projMat, material);
    // render wire frame last
    if (this.drawingMode === DrawingMode.WireFrame || this.drawingMode === DrawingMode.ShadingWithWireFrame) this.drawWireframe(matrix, viewMat, projMat);

    return this;
  }

  setAttribute(shader, name, sz, st, stri) {
    const size = sz || 3;
    const start = st || 0;
    const stride = stri || 0;
    const location = shader.attribLocation(name);
    if (location >= 0) {
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride * Float32Array.BYTES_PER_ELEMENT, start * Float32Array.BYTES_PER_ELEMENT);
    }
    return this;
  }

  setAttributesForShader(shader, buffer, pointers) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    let pointer;
    for (let c = 0, max = pointers.length; c < max; c += 1) {
      pointer = pointers[c];
      this.setAttribute(shader, pointer.name, pointer.size, pointer.offset, pointer.stride);
    }
    return this;
  }

  disableAttribs(shader, attrDatas) {
    let name; let
      location;
    for (let c = 0, max = attrDatas.length; c < max; c += 1) {
      name = attrDatas[c].name;
      location = shader.attribLocation(name);
      if (location >= 0) gl.disableVertexAttribArray(location);
    }
    return this;
  }

  uploadList(list) {
    const data = new Float32Array(list);
    return this.upload(data);
  }

  upload(data) {
    setFrameDirty();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  uploadListWireframe(list) {
    const data = new Float32Array(list);
    return this.uploadWireframe(data);
  }

  uploadWireframe(data) {
    setFrameDirty();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.wireframe_buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }
}

export { Renderable, DrawingMode };
