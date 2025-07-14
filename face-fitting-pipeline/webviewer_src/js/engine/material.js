import { Shader } from './shader.js';
import { Texture2D } from './texture.js';
import {
  vec2, vec3, vec4, mat2, mat3, mat4,
} from '../lib/gl-matrix/index.js';
import { gl } from './gl.js';
import { GameNode } from './gamenode.js';

const BlendType = Object.freeze(
  {
    NoBlend: 0,
    noblend: 0,
    Alpha: 1,
    alpha: 1,
    Add: 2,
    add: 2,
  },
);

function NoBlend() {
  gl.disable(gl.BLEND);
}

function AlphaBlend() {
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
}

function AddBlend() {
  gl.enable(gl.BLEND);
  gl.blendColor(1, 1, 1, 1);
  gl.blendFunc(gl.ONE, gl.CONSTANT_COLOR);
}

// function SpriteBlend() {
//   gl.enable(gl.BLEND);
//   gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
// }

const blendFunctions = [NoBlend, AlphaBlend, AddBlend];

const CullingType = Object.freeze(
  {
    Back: 0,
    back: 0,
    Front: 1,
    front: 1,
    FrontAndBack: 2,
    frontandback: 2,
    Off: 3,
    off: 3,
  },
);

function BackCulling() {
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
}

function FrontCulling() {
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.FRONT);
}

function FrontAndBackCulling() {
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.FRONT_AND_BACK);
}

function OffCulling() {
  gl.disable(gl.CULL_FACE);
}

const cullingFunctions = [BackCulling, FrontCulling, FrontAndBackCulling, OffCulling];

export default class Material {
  constructor(shaderName, blendType, cullingType) {
    this.parameters = {};
    this.pkeys = [];
    this.ifdefines = [];
    this.defines = [];
    this.defineVals = [];
    this.additionalDefines = [];
    this.updateShader(shaderName);
    this.blendType = blendType || BlendType.NoBlend;
    this.blendFunction = blendFunctions[this.blendType];
    this.cullingType = cullingType || CullingType.Back;
    this.cullingFunction = cullingFunctions[this.cullingType];
  }

  get getShader() { return this.shader; }

  updateShader(shaderName = undefined) {
    if (shaderName) {
      this.ifdefines.length = 0;
      this.defines.length = 0;
      this.defineVals.length = 0;
      this.shaderName = shaderName;
    }

    let shaderSrc = Shader.FindShaderSource(this.shaderName);
    if (!shaderSrc) shaderSrc = Shader.FindShaderSource('debug');
    if (shaderName) {
      // console.log(shaderName);
      this.parseFromShader(shaderSrc);
    } else {
      this.updateParameters();
      let defines = '';
      let c;
      let defineObj;
      for (c = this.additionalDefines.length - 1; c >= 0; c -= 1) {
        defineObj = this.additionalDefines[c];
        if (defineObj.val > 0) {
          defines += `#define ${defineObj.key} ${defineObj.val}\n`;
        }
      }
      shaderSrc = {
        name: shaderSrc.name,
        vs: shaderSrc.vs.replace('#pragma DEFINES', defines).replace('#pragma UNIFORMS', ''),
        fs: shaderSrc.fs.replace('#pragma DEFINES', defines).replace('#pragma UNIFORMS', ''),
      };
    }
    // console.log(shaderSrc.fs);
    this.shader = new Shader(shaderSrc);
  }

  updateParameters() {
    let c;
    // let i;
    let param;
    let key;
    let def;
    for (c = this.defines.length - 1; c >= 0; c -= 1) {
      def = this.defines[c];
      if (def in this.parameters) {
        param = this.parameters[def];
        if (param.arrayVal !== this.defineVals[c]) {
          param.data = undefined;
          this.createParameter(key);
        }
      }
      // for (i = this.pkeys.length - 1; i >= 0; i -= 1) {
      //   key = this.pkeys[i];
      //   param = this.parameters[key];
      //   if (param.name === this.defines[c]) {
      //     if (param.arrayVal !== this.defineVals[c]) {
      //       param.data = undefined;
      //       this.createParameter(key);
      //     }
      //   }
      // }
    }
  }

  parseFromShader(shaderSrc) {
    this.parameters = {};
    this.pkeys = [];
    // console.log(shaderSrc.name);
    // console.log('vs');
    this.parseFromStr(shaderSrc.vs);
    // console.log('fs');
    this.parseFromStr(shaderSrc.fs);
    return this;
  }

  getParameter(name) {
    // let c;
    // let param;
    let key;
    if (name in this.parameters) {
      return this.parameters[key].data;
    }
    // for (c = this.pkeys.length - 1; c >= 0; c -= 1) {
    //   key = this.pkeys[c];
    //   param = this.parameters[key];
    //   if (param && param.name === name) {
    //     return param.data;
    //   }
    // }
    return null;
  }

  /**
   * @param {string} name name of uniform data
   * @param {any} data data of uniform data
   * @returns {boolean} true when data setted.
   */
  setUniformData(name, data) {
    // console.slog(name);
    // console.log(this.parameters);
    if (name in this.parameters) {
      const param = this.parameters[name];
      param.data = data;
      param.expired = true;
      return true;
    }
    // let c;
    // let param;
    // let key;
    // for (c = this.pkeys.length - 1; c >= 0; c -= 1) {
    //   key = this.pkeys[c];
    //   param = this.parameters[key];
    //   if (param && param.name === name) {
    //     param.data = data;
    //     param.expired = true;
    //     return true;
    //   }
    // }

    return false;
  }

  setUniformArrayData(name, index, data) {
    if (name in this.parameters) {
      let param = this.parameters[name];
      if (!param.data) {
        param = this.createParameter(name);
      }
      param.data[index] = data;
      param.expired = true;
      return true;
    }
    // let c;
    // let param;
    // let key;
    // for (c = this.pkeys.length - 1; c >= 0; c -= 1) {
    //   key = this.pkeys[c];
    //   param = this.parameters[key];
    //   if (param && param.array && param.name === name) {
    //     if (!param.data) {
    //       param = this.createParameter(key);
    //     }
    //     param.data[index] = data;
    //     param.expired = true;
    //     return true;
    //   }
    // }

    return false;
  }

  hasKey(key) {
    return key in this.parameters;
    // for (let c = this.parameters.length - 1; c >= 0; c -= 1) {
    //   if (this.parameters[c].name === key) return true;
    // }

    // return false;
  }

  hasDefine(key) {
    for (let c = this.defines.length - 1; c >= 0; c -= 1) {
      if (this.defines[c] === key) return true;
    }

    return false;
  }

  hasIfDefine(key) {
    for (let c = this.ifdefines.length - 1; c >= 0; c -= 1) {
      if (this.ifdefines[c] === key) return true;
    }
    return false;
  }

  addOrUpdateDefine(key, val) {
    let index = -1;
    for (let c = this.defines.length - 1; c >= 0; c -= 1) {
      if (this.defines[c] === key) {
        index = c;
        break;
      }
    }
    if (index === -1) {
      // console.log(`insert define ${key}: ${val}`);
      this.additionalDefines.push({ key, val });
      this.defines.push(key);
      this.defineVals.push(val);
    } else {
      if (this.defineVals[index] === val) return;
      this.defineVals[index] = val;
    }
    this.updateShader();
  }

  removeDefine(key) {
    for (let i = this.additionalDefines.length - 1; i >= 0; i -= 1) {
      if (this.additionalDefines[i].key === key) {
        this.additionalDefines.splice(i, 1);
        break;
      }
    }
    for (let c = this.defines.length - 1; c >= 0; c -= 1) {
      if (this.defines[c] === key) {
        this.defines.splice(c, 1);
        this.defineVals.splice(c, 1);
        break;
      }
    }
    this.updateShader();
  }

  getData(key) {
    return this.parameters[key].data;
  }

  createParameter(key) {
    const param = this.parameters[key];
    switch (param.type) {
      case 'sampler2D':
        if (!param.array) {
          switch (param.name) {
            case 'uNormal':
              param.data = new Texture2D(
                undefined, undefined, 1, 1, new Uint8Array([128, 127, 255, 255]),
              );
              break;
            case 'uSpecular':
              param.data = new Texture2D(
                undefined, undefined, 1, 1, new Uint8Array([70, 70, 70, 255]),
              );
              break;
            case 'uEnvir':
              param.data = new Texture2D(
                undefined, undefined, 1, 1, new Uint8Array([0, 0, 0, 255]),
              );
              break;
            default:
              param.data = new Texture2D();
              break;
          }
        } else {
          param.data = [new Texture2D(
            undefined, undefined, 1, 1, new Uint8Array([255, 255, 255, 255]),
          )];
          param.texArr = new Int32Array(param.arrayVal);
        }
        break;
      case 'vec4':
        if (!param.array) param.data = vec4.create();
        else param.data = new Float32Array(param.arrayVal * 4);
        break;
      case 'vec3':
        if (!param.array) param.data = vec3.create();
        else param.data = new Float32Array(param.arrayVal * 3);
        break;
      case 'vec2':
        if (!param.array) param.data = vec2.create();
        else param.data = new Float32Array(param.arrayVal * 2);
        break;
      case 'mat2':
        if (!param.array) param.data = mat2.create();
        else param.data = new Float32Array(param.arrayVal * 4);
        break;
      case 'mat3':
        if (!param.array) param.data = mat3.create();
        else param.data = new Float32Array(param.arrayVal * 9);
        break;
      case 'mat4':
        if (!param.array) param.data = mat4.create();
        else param.data = new Float32Array(param.arrayVal * 16);
        break;
      case 'float':
      case 'double':
        if (!param.array) param.data = 0.0;
        else param.data = new Float32Array(param.arrayVal);
        break;
      case 'bool':
      case 'int':
        if (!param.array) param.data = 0;
        else param.data = new Int32Array(param.arrayVal);
        break;
      default:
        break;
    }
    param.expired = true;
    return param;
  }

  /**
   * update all datas.
   */
  setUniformDatas() {
    let c;
    let max;
    let param;
    let key;
    let texunit = 0;
    // console.log(`=================== ${this.shaderName}`);
    for (c = 0, max = this.pkeys.length; c < max; c += 1) {
      key = this.pkeys[c];
      param = this.parameters[key];
      if (!param.data) {
        param = this.createParameter(key);
      }
      if (param.expired === undefined || param.expired) {
        param.expired = false;
        switch (param.type) {
          case 'sampler2D':
            param.data.bind(texunit);
            // console.log(param);
            this.shader.setInt(param.name, texunit);
            texunit += 1;
            param.data.unbind(texunit);
            param.expired = true;
            break;
          case 'vec4':
            // eslint-disable-next-line max-len
            this.shader.setVec4(param.name, param.data);
            break;
          case 'vec3':
            // console.log(`name: ${param.name} data: ${param.data}`);
            this.shader.setVec3(param.name, param.data);
            break;
          case 'vec2':
            this.shader.setVec2(param.name, param.data);
            break;
          case 'mat2':
            this.shader.setMatrix2(param.name, param.data);
            break;
          case 'mat3':
            this.shader.setMatrix3(param.name, param.data);
            break;
          case 'mat4':
            this.shader.setMatrix4(param.name, param.data);
            break;
          case 'float':
          case 'double':
            if (param.array) this.shader.setFv(param.name, param.data);
            else this.shader.setFloat(param.name, param.data);
            break;
          case 'bool':
          case 'int':
            if (param.array) this.shader.Iv(param.name, param.data);
            else this.shader.setInt(param.name, param.data);
            break;
          default:
            break;
        }
        const err = gl.getError();
        if (err !== gl.NO_ERROR) {
          // eslint-disable-next-line no-console
          console.error(`error code : ${err}`);
        }
      }
    }
  }

  setBlendType(blendType) {
    this.blendType = blendType || BlendType.NoBlend;
    if (typeof this.blendType === 'string') {
      this.blendType = BlendType[this.blendType.toLowerCase()];
    }
    this.blendFunction = blendFunctions[this.blendType];
    GameNode.sortRenderingGroup();
  }

  setCullingType(cullingType) {
    this.cullingType = cullingType || CullingType.Back;
    if (typeof this.cullingType === 'string') {
      this.cullingType = CullingType[this.cullingType.toLowerCase()];
    }
    this.cullingFunction = cullingFunctions[this.cullingType];
  }

  onRender() {
    this.blendFunction();
    this.cullingFunction();
  }

  parseFromStr(text) {
    const str = text.replace('#pragma DEFINES', '').replace('#pragma UNIFORMS', '');

    let i = 0;
    let end;

    do {
      i = str.indexOf('#define', i);
      if (i >= 0) {
        i += 8;
        end = str.indexOf(' ', i);
        this.defines.push(str.substring(i, end));
        this.defineVals.push(parseFloat(str.substring(end, str.indexOf('\n', end))));
      }
    }
    while (i >= 0);

    // console.log(str);

    const lines = str.split('\n');
    let type;
    let ifdef;
    let arg;
    let arrayVal;
    let words;
    let word;
    let line;
    let c;
    let d;
    let max;
    let dmax;
    let array = false;
    for (c = 0, max = lines.length; c <= max; c += 1) {
      line = lines[c];
      if (line.length !== 0) {
        if (line.includes('{')) break;
        if (line.includes('uniform') || line.includes('#ifdef')) {
          array = line.includes('[') || line.includes(']');
          words = line.split(' ');
          type = null;
          arg = null;
          ifdef = null;
          arrayVal = null;

          for (d = 0, dmax = words.length; d < dmax; d += 1) {
            word = words[d];
            switch (word) {
              case '':
              case 'uniform':
                break;
              case '#ifdef':
                if (word) ifdef = word;
                break;
              case 'sampler2D':
              case 'vec4':
              case 'vec3':
              case 'vec2':
              case 'mat2':
              case 'mat3':
              case 'mat4':
              case 'float':
              case 'double':
              case 'bool':
              case 'int':
                type = word;
                break;
              default:
                if (word) {
                  if (word.includes(';')) {
                    word = word.replace(';', '');
                  }
                  if (word.includes('[')) {
                    const ind = word.indexOf('[');
                    const defName = word.substring(ind + 1, word.indexOf(']'));
                    if (this.defines.includes(defName)) {
                      // console.log(`${line}:${word}[${arrayVal}]`);
                      arrayVal = this.defineVals[this.defines.indexOf(defName)];
                    } else if (typeof defName === 'string') {
                      arrayVal = 1;
                      // console.log(`${line}:${word}[?0] defines : ${this.defines}`);
                    } else {
                      // eslint-disable-next-line radix
                      arrayVal = parseInt(defName);
                      // console.log(`${line}:${word}[int ${defName}]`);
                    }
                    word = word.substring(0, ind);
                  }
                  arg = word;
                }
                break;
            }
            if (arg !== null) {
              if (type !== null) {
                this.pkeys.push(arg);
                this.parameters[arg] = {
                  name: arg, type, array, arrayVal,
                };
                break;
              } else if (ifdef !== null) {
                this.ifdefines.push(arg);
              }
            }
          }
        }
      }
    }
    // console.log(this.parameters);
  }
}

export { Material, BlendType, CullingType };
