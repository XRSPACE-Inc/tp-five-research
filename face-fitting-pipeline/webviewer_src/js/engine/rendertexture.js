import { gl } from './gl.js';
import { Framebuffer } from './framebuffer.js';
import { Texture2D, TextureParamType } from './texture.js';

const RenderTextureType = Object.freeze(
  {
    Color: 0,
    Depth: 1,
  },
);

export default class RenderTexture {
  constructor(type, width, height) {
    this.type = type;
    this.width = width;
    this.height = height;
    this.framebuffer = new Framebuffer();
    let texFormat;
    let texType;
    this.framebuffer.bind();
    switch (this.type) {
      case RenderTextureType.Depth:
        texFormat = gl.DEPTH_COMPONENT;
        texType = gl.UNSIGNED_SHORT;
        break;
      default:
        texFormat = gl.RGB;
        texType = gl.UNSIGNED_BYTE;
        break;
    }

    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
    const option = { Wrap: TextureParamType.Clamp_To_Edge, Filter: TextureParamType.Nearest };
    this.texture = new Texture2D(texFormat, texType, this.width, this.height, null, option);

    switch (this.type) {
      case RenderTextureType.Depth:
        this.framebuffer.getDepthTexture(this.texture);
        break;
      default:
        this.framebuffer.getColor(this.texture);
        break;
    }
    this.framebuffer.unbind();
  }

  bind() {
    this.framebuffer.bind();
    gl.viewport(0, 0, this.width, this.height);
  }

  unbind() {
    this.framebuffer.unbind();
  }
}

export { RenderTexture, RenderTextureType };
