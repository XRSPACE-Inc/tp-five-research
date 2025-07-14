/* eslint-disable no-console */
/* eslint-disable max-classes-per-file */
import { gl } from './gl.js';

let framebufferBinding = null;

function checkFramebuffer() {
  const result = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  switch (result) {
    case gl.FRAMEBUFFER_UNSUPPORTED:
      console.error('framebuffer error:Framebuffer is unsupported');
      return false;
    case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
      console.error('framebuffer error:Framebuffer incomplete attachment');
      return false;
    case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
      console.error('framebuffer error:Framebuffer incomplete dimensions');
      return false;
    case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
      console.error('framebuffer error:Framebuffer incomplete missing attachment');
      return false;
    default:
      break;
  }
  return true;
}

export default class Framebuffer {
  constructor() {
    this.fbo = gl.createFramebuffer();
  }

  bind() {
    if (framebufferBinding !== this) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      framebufferBinding = this;
    }
    return this;
  }

  unbind() {
    if (framebufferBinding !== null) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      framebufferBinding = null;
    }
    return this;
  }

  getColor(texture, t) {
    let target = t;
    if (!target) target = texture.target;
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, target, texture.texture, 0);
    checkFramebuffer();
    return this;
  }

  getDepthTexture(texture, t) {
    let target = t;
    if (!target) target = texture.target;
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, target, texture.texture, 0);
    checkFramebuffer();
    return this;
  }

  getDepthBuffer(buffer) {
    gl.framebufferRenderbuffer(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, buffer.buffer,
    );
    checkFramebuffer();
    return this;
  }

  destroy() {
    return gl.deleteFramebuffer(this.fbo);
  }
}

class Renderbuffer {
  constructor() {
    this.buffer = gl.createRenderbuffer();
  }

  bind() {
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.buffer);
    return this;
  }

  unbind() {
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    return this;
  }

  setSize(width, height) {
    this.width = width;
    this.height = height;

    this.bind();
    gl.renderbufferStorage(gl.RENDERBUFFER, this.format, this.width, this.height);
    return this.unbind();
  }
}

class Depthbuffer extends Renderbuffer {
  constructor() {
    super();
    this.format = gl.DEPTH_COMPONENT16;
  }
}

export { Framebuffer, Depthbuffer, checkFramebuffer };
