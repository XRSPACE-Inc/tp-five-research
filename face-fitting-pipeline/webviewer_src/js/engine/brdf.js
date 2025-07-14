/* eslint-disable max-len */
import { Material } from './material.js';
import { Quad } from './quad.js';
import { GameNode } from './gamenode.js';
import { RenderTexture, RenderTextureType } from './rendertexture.js';
import { Camera } from './camera.js';

export default class BRDF {
  constructor(size) {
    this.lutSize = size;
    this.quad = new GameNode(new Quad(new Material('brdf')), 'brdfNode', false);
    this.quad.renderable.enabled = false;
    this.camera = new Camera(this.lutSize, this.lutSize);
    this.camera.fixedSize = true;
    this.camera.zNear = 0.1;
    this.camera.zFar = 10.0;
    this.camera.ortho = true;
    this.camera.orthoSize = 1.0;
    this.camera.clearColor = true;
    this.camera.clearDepth = true;
    this.camera.backgroundColor = [1.0, 1.0, 1.0, 1.0];
    this.camera.enabled = false;
    this.renderTexture = new RenderTexture(RenderTextureType.Color, this.lutSize, this.lutSize);
    this.done = false;
  }

  RenderToBDRFMap() {
    this.quad.renderable.enabled = true;
    this.camera.enabled = true;
    this.renderTexture.bind();
    this.camera.render([this.quad], false);
    this.renderTexture.unbind();
    this.camera.enabled = false;
    this.quad.renderable.enabled = false;
    this.done = true;
  }

  getTexture() {
    if (!this.done) return null;
    return this.renderTexture.texture;
  }
}
let brdfTexture;
let brdfInstance;

function getBRDF() {
  if (!brdfInstance) {
    brdfInstance = new BRDF(512);
    brdfInstance.RenderToBDRFMap();
    brdfTexture = brdfInstance.getTexture();
  }
  return brdfTexture;
}

export { getBRDF };
