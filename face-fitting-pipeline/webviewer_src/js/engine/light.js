import { Camera } from './camera.js';
import { RenderTexture, RenderTextureType } from './rendertexture.js';
import { Material } from './material.js';
import { GameNode } from './gamenode.js';
import { Sprite } from './UI/sprite.js';
import { Rect } from './UI/rect.js';
import { DebugDraw } from './debugDraw.js';
import { mat4 } from '../lib/gl-matrix/index.js';
// import { mulPoint } from './utils.js';

const allLights = [];
let lightSize = 0;
let lightPositions;
let lightDirections;
let lightColors;
let lightMatrices;

function UpdateLightParams() {
  lightSize = allLights.length;
  lightPositions = new Float32Array(lightSize * 3);
  lightDirections = new Float32Array(lightSize * 3);
  lightColors = new Float32Array(lightSize * 3);
  lightMatrices = new Float32Array(lightSize * 16);
}

export default class Light {
  constructor() {
    this.renderTexture = new RenderTexture(RenderTextureType.Depth, 1536, 1536);
    this.camera = new Camera(1536.0, 1536.0);
    this.camera.fixedSize = true;
    // this.camera.setFOV(45);
    this.camera.zNear = 0.1;
    this.camera.zFar = 10.0;
    this.camera.ortho = true;
    this.camera.orthoSize = 1.0;
    this.camera.clearColor = true;
    this.camera.clearDepth = true;
    this.camera.backgroundColor = [1.0, 1.0, 1.0, 1.0];
    this.camera.enabled = false;
    this.shadowMaterial = new Material('shadow');
    this.color = [1.0, 1.0, 1.0];
    this.index = allLights.length;
    this.debug = false;
    allLights.push(this);
    UpdateLightParams();
  }

  get transform() { return this.camera.transform; }

  // render depth
  render(nodeGroup) {
    if (this.quad || this.debug) {
      DebugDraw.get()?.clear();
      DebugDraw.get()?.addOrigLine(
        this.camera.transform.position, this.camera.transform.matrix,
      );
    }
    this.camera.enabled = true;
    this.renderTexture.bind();
    this.camera.render(nodeGroup, false, undefined, this.shadowMaterial);
    this.renderTexture.unbind();
    this.camera.enabled = false;
  }

  debugRenderTexture() {
    if (!this.quad) {
      this.quad = new GameNode(new Sprite(undefined, this.renderTexture.texture), `debug_shadowmap_${this.index}`);
      this.quad = this.quad.renderable;
      this.quad.rect = new Rect(0, 0, 100, 100).set(
        -256 * (1 + this.index),
        -256,
        256,
        256,
      );
    }
  }

  static Update() {
    let i;
    let j;
    let f;
    let m;
    let c;
    let p;
    let l;
    const imax = allLights.length;
    // console.log(allLights);
    // console.slog(lightPositions);
    for (i = 0; i < imax; i += 1) {
      l = allLights[i];
      p = l.camera.transform.position;
      f = l.camera.transform.forward;
      m = mat4.multiply(mat4.create(), l.camera.projectionMatrix, l.camera.transform.inverseMatrix);
      c = l.color;
      for (j = 0; j < 3; j += 1) {
        lightPositions[(i * 3) + j] = p[j];
        lightDirections[(i * 3) + j] = f[j];
        lightColors[(i * 3) + j] = c[j];
      }
      for (j = 0; j < 16; j += 1) {
        lightMatrices[(i * 16) + j] = m[j];
      }
    }

    // let point = mulPoint(lightMatrices, [1, 1, 1]);
    // console.log(point);
    // console.log(lightMatrices);
    // console.log(lightColors);
  }

  static SetLightUniformDatas(mat) {
    let i;
    let n;
    const imax = allLights.length;
    for (i = 0; i < imax; i += 1) {
      n = `shadowmap${i}`;
      if (mat.hasKey(n)) mat.setUniformData(n, allLights[i].renderTexture.texture);
    }
    if (mat.hasKey('lightPositions')) mat.setUniformData('lightPositions', lightDirections);
    if (mat.hasKey('lightDirections')) mat.setUniformData('lightDirections', lightDirections);
    if (mat.hasKey('lightColors')) mat.setUniformData('lightColors', lightColors);
    if (mat.hasKey('lightMatrices')) mat.setUniformData('lightMatrices', lightMatrices);
    if (mat.hasDefine('SKIN')) {
      if (mat.hasKey('uTransColor')) mat.setUniformData('uTransColor', [1.0, 0.25, 0.125, 1.0]);
      if (mat.hasKey('uFresnelColor')) mat.setUniformData('uFresnelColor', [0.0, 0.0, 0.0, 0.3]);
      if (mat.hasKey('uSubdermisColor')) mat.setUniformData('uSubdermisColor', [1.0, 0.55, 0.45]);
      if (mat.hasKey('uTransScatter')) mat.setUniformData('uTransScatter', 0.45);
      if (mat.hasKey('uFresnelOcc')) mat.setUniformData('uFresnelOcc', 1.0);
      if (mat.hasKey('uFresnelGlossMask')) mat.setUniformData('uFresnelGlossMask', 1.0);
      if (mat.hasKey('uTransSky')) mat.setUniformData('uTransSky', 0.0);
      if (mat.hasKey('uTransIntegral')) mat.setUniformData('uTransIntegral', 0.374193);
      if (mat.hasKey('uSkinTransDepth')) mat.setUniformData('uSkinTransDepth', 0.0);
      if (mat.hasKey('uSkinShadowBlur')) mat.setUniformData('uSkinShadowBlur', 1.44122);
      if (mat.hasKey('uNormalSmooth')) mat.setUniformData('uNormalSmooth', 1.0);
      if (mat.hasKey('uHorizonOcclude')) mat.setUniformData('uHorizonOcclude', 1.78070998);
      if (mat.hasKey('uFresnel')) mat.setUniformData('uFresnel', [0.12274912, 0.20372236, 0.219281]);
      if (mat.hasKey('uTexRangeTranslucency')) mat.setUniformData('uTexRangeTranslucency', [0.96875, 0.96875, 0.015625, 0.015625]);
    } else {
      if (mat.hasKey('uFresnelColor')) mat.setUniformData('uFresnelColor', [1.0, 1.0, 1.0, 0.3]);
      if (mat.hasKey('uNormalSmooth')) mat.setUniformData('uNormalSmooth', 1.0);
      if (mat.hasKey('uHorizonOcclude')) mat.setUniformData('uHorizonOcclude', 1.0);
    }
    if (mat.hasKey('uSH9Coeffs')) {
      // mat.setUniformData('uSH9Coeffs', [
      //   0.4207298755645752, 0.2736755311489105, 0.20377586781978607, 0,
      //   0.47618910670280457, 0.34059688448905945, 0.27080512046813965, 0,
      //   -0.12397418916225433, -0.07783327996730804, -0.047592807561159134, 0,
      //   0.028830628842115402, 0.02182578295469284, 0.018897922709584236, 0,
      //   0.04665800929069519, 0.033903948962688446, 0.033427607268095016, 0,
      //   -0.2064231038093567, -0.13301518559455872, -0.08447179943323135, 0,
      //   -0.06476601958274841, -0.04491056501865387, -0.03271108493208885, 0,
      //   -0.014419540762901306, -0.011562635190784931, -0.014340583235025406, 0,
      //   -0.25701797008514404, -0.17173852026462555, -0.12372913956642151, 0,
      // ]);
      mat.setUniformData('uSH9Coeffs', [
        0.1, 0.09, 0.1, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
      ]);
    }
  }

  static lightCount() {
    return lightSize;
  }

  static getLights() {
    return allLights;
  }
}

export { Light, allLights };
