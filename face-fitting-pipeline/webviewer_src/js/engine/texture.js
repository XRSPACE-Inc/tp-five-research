/* eslint-disable no-console */
/* eslint-disable max-classes-per-file */
import { gl } from './gl.js';
import { ReadFile } from './readFile.js';
import { setFrameDirty } from './utils.js';
import { Framebuffer } from './framebuffer.js';

let maxTexUnitCount;
let texID = 0;

const TextureParamType = Object.freeze(
  {
    Mipmap: 0,
    Mipmap_Nearest: 1,
    Linear: 2,
    Nearest: 3,
    Clamp_To_Edge: 4,
    Repeat: 5,
    Anisotropy: 6,
  },
);

function isPowerOf2(value) {
  // eslint-disable-next-line no-bitwise
  return (value & (value - 1)) === 0;
}

class Texture {
  constructor() {
    this.id = texID;
    texID += 1;
    this.unit = null;
    this.target = null;

    if (maxTexUnitCount === undefined) {
      maxTexUnitCount = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
    }

    this.texture = gl.createTexture();

    return this;
  }

  bind(unit = 0) {
    this.unit = unit;
    if (unit > maxTexUnitCount) console.log(`meet maxTexUnitCount ${unit} > ${maxTexUnitCount}`);
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(this.target, this.texture);
    return this;
  }

  unbind(u) {
    let unit = u;
    if (!unit) unit = this.unit;
    if (unit) {
      gl.activeTexture(gl.TEXTURE0 + unit, null);
      gl.bindTexture(this.target, null);
    }
    return this;
  }

  setParamType(type) {
    switch (type) {
      case TextureParamType.Mipmap:
        gl.texParameteri(this.target, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(this.target, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.generateMipmap(this.target);
        break;
      case TextureParamType.Mipmap_Nearest:
        gl.texParameteri(this.target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(this.target, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.generateMipmap(this.target);
        break;
      case TextureParamType.Linear:
        gl.texParameteri(this.target, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(this.target, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        break;
      case TextureParamType.Nearest:
        gl.texParameteri(this.target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(this.target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        break;
      case TextureParamType.Clamp_To_Edge:
        gl.texParameteri(this.target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(this.target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        break;
      case TextureParamType.Repeat:
        gl.texParameteri(this.target, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(this.target, gl.TEXTURE_WRAP_T, gl.REPEAT);
        break;
      case TextureParamType.Anisotropy: {
        const ext = gl.getExtension('WEBKIT_EXT_texture_filter_anisotropic');
        if (ext) {
          const max = gl.getParameter(ext.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
          gl.texParameterf(this.target, ext.TEXTURE_MAX_ANISOTROPY_EXT, max);
        }
        break;
      }
      default:
        break;
    }
    return this;
  }
}

class Texture2D extends Texture {
  /**
   * texture ref: https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/texImage2D
   * format type:
   * gl.ALPHA, gl.RGB, gl.RGBA, gl.LUMINANCE, gl.LUMINANCE_ALPHA
   * When using the WEBGL_depth_texture extension:
   * gl.DEPTH_COMPONENT, gl.DEPTH_STENCIL
   * When using the EXT_sRGB extension:
   * ext.SRGB_EXT, ext.SRGB_ALPHA_EXT
   * WebGL2:
   * gl.R8, gl.R16F, gl.R32F, gl.R8UI, gl.RG8, gl.RG16F, gl.RG32F, gl.RG8UI,
   * gl.RG16UI, gl.RG32UI, gl.RGB8, gl.SRGB8, gl.RGB565, gl.R11F_G11F_B10F,
   * gl.RGB9_E5, gl.RGB16F, gl.RGB32F, gl.RGB8UI, gl.RGBA8, gl.SRGB8_APLHA8,
   * gl.RGB5_A1, gl.RGB10_A2, gl.RGBA4, gl.RGBA16F, gl.RGBA32F, gl.RGBA8UI.
   *
   * type:
   * gl.UNSIGNED_BYTE: 8 bits per channel for gl.RGBA
   * gl.UNSIGNED_SHORT_5_6_5, gl.UNSIGNED_SHORT_4_4_4_4, gl.UNSIGNED_SHORT_5_5_5_1
   * When using the WEBGL_depth_texture extension:
   * gl.UNSIGNED_SHORT, gl.UNSIGNEDjNT, ext.UNSIGNEDjNT_24_8_WEBGL
   * When using the OES_texture_float extension:
   * gl.FLOAT
   * When using the OES_texture_half_float extension:
   * ext.HALF_FLOAT_OES (constant provided by the extension)
   * WebGL2:
   * gl.BYTE, gl.UNSIGNED_SHORT, gl.SHORT, gl.UNSIGNEDjNT, gl.INT, gl.HALF_FLOAT,
   * gl.FLOAT, gl.UNSIGNEDjNT_2_10_10_10_REV, gl.UNSIGNEDjNT_10F_11F_11F_REV,
   * gl.UNSIGNEDjNT_5_9_9_9_REV,
   * gl.UNSIGNEDjNT_24_8,
   * gl.FLOAT_32_UNSIGNEDjNT_24_8_REV (pixels must be null)
   *
   * @param {any} gl gl
   * @param {GLenum} format gl tex format
   * @param {GLenum} type texture type
   */
  constructor(
    f, t,
    width = 1,
    height = 1,
    pixels = undefined,
    options = { Filter: TextureParamType.Mipmap },
  ) {
    super();

    const format = f || gl.RGBA;
    const type = t || gl.UNSIGNED_BYTE;

    this.pixel = pixels;
    if (pixels === undefined && type === gl.UNSIGNED_BYTE) {
      this.pixel = new Uint8Array([160, 60, 60, 255]);
      setFrameDirty();
    }
    // level is mipmap level highest is 0
    this.level = 0;
    // format type
    this.internalFormat = format;
    this.srcFormat = format;
    // border must be 0
    this.border = 0;
    this.srcType = type;
    this.target = gl.TEXTURE_2D;
    this.width = width;
    this.height = height;

    // gl.activeTexture(gl.TEXTURE0 + maxTexUnitCount - 1);
    gl.bindTexture(this.target, this.texture);

    gl.texImage2D(this.target, this.level, this.internalFormat, this.width, this.height,
      this.border, this.srcFormat, this.srcType, this.pixel);

    if (options !== undefined) {
      if (options.Wrap !== undefined) {
        this.setParamType(options.Wrap);
      }
      if (options.Filter !== undefined) {
        this.setParamType(options.Filter);
      }
    }

    const err = gl.getError();
    if (err !== gl.NO_ERROR) {
      console.error(`texImage2D error gl error code : ${err}
      target: ${this.target}, (TEXTURE_2D :${gl.TEXTURE_2D})
      level: ${this.level}
      internalFormat: ${this.internalFormat}, (gl.RGB : ${gl.RGB}, gl.RGBA : ${gl.RGBA})
      width: ${this.width}
      height: ${this.height}
      border: ${this.border}
      format: ${this.srcFormat}, should be same as internalFormat
      srcType: ${this.srcType}, (gl.UNSIGNED_BYTE : ${gl.UNSIGNED_BYTE}, gl.FLOAT : ${gl.FLOAT})`);
    }

    this.image = null;
    gl.bindTexture(gl.TEXTURE_2D, null);
    // gl.activeTexture(gl.TEXTURE0);

    return this;
  }

  get imageLoaded() { return this.src === undefined || this.image !== null; }

  loadFromUrl(src, onloaded = undefined, token = undefined) {
  // Because images have to be download over the internet
  // they might take a moment until they are ready.
  // Until then put a single pixel in the texture so we can
  // use it immediately. When the image has finished downloading
  // we'll update the texture with the contents of the image.
    this.onloaded = onloaded;
    ReadFile.readImage(src, (image, self) => {
      gl.activeTexture(gl.TEXTURE0 + maxTexUnitCount - 1);
      gl.bindTexture(gl.TEXTURE_2D, self.texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
      gl.texImage2D(
        gl.TEXTURE_2D, self.level, self.internalFormat, self.srcFormat, self.srcType, image,
      );

      // eslint-disable-next-line no-param-reassign
      self.width = image.width;
      // eslint-disable-next-line no-param-reassign
      self.height = image.height;

      const genmipmap = self.mipmap && isPowerOf2(self.width) && isPowerOf2(self.height);
      if (genmipmap) {
        gl.generateMipmap(gl.TEXTURE_2D);
      } else {
        self.setParamType(TextureParamType.Clamp_To_Edge);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      }
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.activeTexture(gl.TEXTURE0);
      // eslint-disable-next-line no-param-reassign
      self.image = image;
      setFrameDirty();
      if (self.onloaded !== undefined) {
        self.onloaded();
      }

      return self;
    }, this, token);

    return this;
  }

  loadFromImage(image) {
    const self = this;
    gl.activeTexture(gl.TEXTURE0 + maxTexUnitCount - 1);
    gl.bindTexture(gl.TEXTURE_2D, self.texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
    gl.texImage2D(
      gl.TEXTURE_2D, self.level, self.internalFormat, self.srcFormat, self.srcType, image,
    );

    // eslint-disable-next-line no-param-reassign
    self.width = image.width;
    // eslint-disable-next-line no-param-reassign
    self.height = image.height;

    const genmipmap = self.mipmap && isPowerOf2(self.width) && isPowerOf2(self.height);
    if (genmipmap) {
      gl.generateMipmap(gl.TEXTURE_2D);
    } else {
      self.setParamType(TextureParamType.Clamp_To_Edge);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0);
    // eslint-disable-next-line no-param-reassign
    self.image = image;
    setFrameDirty();

    return this;
  }

  loadFromColorArray(arr, width, height, onloaded = undefined) {
  // Because images have to be download over the internet
  // they might take a moment until they are ready.
  // Until then put a single pixel in the texture so we can
  // use it immediately. When the image has finished downloading
  // we'll update the texture with the contents of the image.
    this.onloaded = onloaded;
    const image = new ImageData(new Uint8ClampedArray(arr), width, height);

    gl.activeTexture(gl.TEXTURE0 + maxTexUnitCount - 1);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
    gl.texImage2D(
      gl.TEXTURE_2D, this.level, this.internalFormat, this.srcFormat, this.srcType, image,
    );

    this.width = image.width;
    this.height = image.height;

    const genmipmap = this.mipmap && this.isPowerOf2(this.width) && this.isPowerOf2(this.height);
    if (genmipmap) {
      gl.generateMipmap(gl.TEXTURE_2D);
    } else {
      this.setParamType(TextureParamType.Clamp_To_Edge);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0);
    this.image = image;
    setFrameDirty();
    if (this.onloaded !== undefined) {
      this.onloaded();
    }

    return this;
  }

  setSize(width, height) {
    this.width = width;
    this.height = height;

    gl.texImage2D(
      this.target, this.level, this.internalFormat,
      this.width, this.height, this.border, this.srcFormat, this.srcType, null,
    );

    return this;
  }

  read(o) {
    let out = o;
    if (!out) out = new Uint8Array(this.width * this.height * 4);
    if (this.fbo) this.fbo.bind();
    else this.fbo = new Framebuffer(gl).bind().getColor(this);
    gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, out);
    this.fbo.unbind();
    return out;
  }

  toPNG() {
    const canvas = document.createElement('canvas');
    canvas.height = this.height;
    canvas.width = this.width;
    const ctx = canvas.getContext('2d');
    const imgdata = ctx.createImageData(this.width, this.height);
    imgdata.data.set(this.read(), 0);
    ctx.putImageData(imgdata, 0, 0);
    const url = canvas.toDataURL('image/png');
    const data = atob(url.split(',')[1]);
    const result = new Uint8Array(data.length);
    let i;
    let j;
    let ref;
    for (i = 0, j = 0, ref = data.length;
      ref >= 0 ? j < ref : j > ref;
      i = (ref >= 0 ? (j += 1) : (j -= 1))) {
      result[i] = data.charCodeAt(i);
    }
    return result;
  }
}

let defaultTextures;
function getDefaultTextures() {
  if (!defaultTextures) {
    defaultTextures = {
      empty: new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, 1, 1,
        new Uint8Array([0, 0, 0, 0])),
      white: new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, 1, 1,
        new Uint8Array([255, 255, 255, 255])),
      black: new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, 1, 1,
        new Uint8Array([0, 0, 0, 255])),
      red: new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, 1, 1,
        new Uint8Array([255, 0, 0, 255])),
      green: new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, 1, 1,
        new Uint8Array([0, 255, 0, 255])),
      blue: new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, 1, 1,
        new Uint8Array([0, 0, 255, 255])),
      yellow: new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, 1, 1,
        new Uint8Array([255, 255, 0, 255])),
      cyan: new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, 1, 1,
        new Uint8Array([0, 255, 255, 255])),
      magenta: new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, 1, 1,
        new Uint8Array([255, 0, 255, 255])),
    };
  }
  return defaultTextures;
}

export default Texture2D;
export {
  TextureParamType,
  isPowerOf2,
  Texture,
  Texture2D,
  getDefaultTextures,
};
