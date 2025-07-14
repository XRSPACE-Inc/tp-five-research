/* eslint-disable no-param-reassign */
import { vec3, mat4, quat } from '../lib/gl-matrix/index.js';

function copySign(a, b) {
  return b < 0 ? -Math.abs(a) : Math.abs(a);
}

function toEulerAngle(q, eulerOut) {
  // roll (x-axis rotation)
  const sinr = +2.0 * (q[3] * q[0] + q[1] * q[2]);
  const cosr = +1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1]);
  eulerOut[0] = Math.atan2(sinr, cosr);

  // pitch (y-axis rotation)
  const sinp = +2.0 * (q[3] * q[1] - q[2] * q[0]);
  // use 90 degrees if out of range
  if (Math.abs(sinp) >= 1) eulerOut[1] = copySign(Math.PI / 2, sinp);
  else eulerOut[1] = Math.asin(sinp);

  // yaw (z-axis rotation)
  const siny = +2.0 * (q[3] * q[2] + q[0] * q[1]);
  const cosy = +1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
  eulerOut[2] = Math.atan2(siny, cosy);

  return eulerOut;
}

// just in case you need that function also
function createFromAxisAngle(axis, angle) {
  const halfAngle = angle * 0.5;
  const s = Math.sin(halfAngle);
  const q = quat.create();
  q[0] = axis[0] * s;
  q[1] = axis[1] * s;
  q[2] = axis[2] * s;
  q[3] = Math.cos(halfAngle);
  return q;
}

function mat4MulVec3(matrix, vec) {
  return [matrix[0] * vec[0] + matrix[4] * vec[1] + matrix[8] * vec[2],
    matrix[1] * vec[0] + matrix[5] * vec[1] + matrix[9] * vec[2],
    matrix[2] * vec[0] + matrix[6] * vec[1] + matrix[10] * vec[2]];
}

export default class Transform {
  constructor() {
    this.pos = vec3.create();
    this.rot = quat.create();
    this.eul = vec3.create();
    this.scal = vec3.set(vec3.create(), 1.0, 1.0, 1.0);

    return this;
  }

  get position() { return this.pos; }

  set position(position) { this.pos = position; }

  get euler() {
    if (!this.eul) this.eul = toEulerAngle(this.rot, vec3.create());
    return this.eul;
  }

  set euler(euler) {
    this.eul = euler;
    this.rot = undefined;
  }

  get rotation() {
    if (!this.rot) this.rot = quat.fromEuler(quat.create(), this.eul[0], this.eul[1], this.eul[2]);
    return this.rot;
  }

  set rotation(rotation) {
    this.rot = rotation;
    this.eul = undefined;
  }

  get scale() { return this.scal; }

  set scale(scale) { this.scal = scale; }

  /**
   * transformation.
  */
  get matrix() {
    return mat4.fromRotationTranslationScale(mat4.create(), this.rotation, this.pos, this.scal);
  }

  /**
   * inversed transformation.
  */
  get inverseMatrix() {
    return mat4.invert(mat4.create(), this.matrix);
  }

  /**
   *  forward(0.0, 0.0, 1.0) of this transform.
   */
  get forward() {
    return vec3.normalize(vec3.create(), mat4MulVec3(this.matrix, [0.0, 0.0, 1.0]));
  }

  /**
   *  back(0.0, 0.0, -1.0) of this transform.
   */
  get back() {
    return vec3.normalize(vec3.create(), mat4MulVec3(this.matrix, [0.0, 0.0, -1.0]));
  }

  /**
   *  right(1.0, 0.0, 0.0) of this transform.
   */
  get right() {
    return vec3.normalize(vec3.create(), mat4MulVec3(this.matrix, [1.0, 0.0, 0.0]));
  }

  /**
   *  left(-1.0, 0.0, 0.0) of this transform.
   */
  get left() {
    return vec3.normalize(vec3.create(), mat4MulVec3(this.matrix, [-1.0, 0.0, 0.0]));
  }

  /**
   *  up(0.0, 1.0, 0.0) of this transform.
   */
  get up() {
    return vec3.normalize(vec3.create(), mat4MulVec3(this.matrix, [0.0, 1.0, 0.0]));
  }

  /**
   *  down(0.0, -1.0, 0.0) of this transform.
   */
  get down() {
    return vec3.normalize(vec3.create(), mat4MulVec3(this.matrix, [0.0, -1.0, 0.0]));
  }

  lookAt(pnt, up = [0.0, 1.0, 0.0]) {
    const m = mat4.targetTo(mat4.create(), this.pos, pnt, up);
    mat4.getRotation(this.rot, m);
    this.eul = undefined;
  }

  equals(other) {
    return mat4.equals(this.matrix, other.matrix);
  }
}

export {
  copySign,
  toEulerAngle,
  createFromAxisAngle,
  mat4MulVec3,
  Transform,
};
