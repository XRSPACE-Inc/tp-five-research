import { input } from '../engine/inputmanager.js';
import { allLights } from '../engine/light.js';
import { timer } from '../engine/timer.js';
import { setFrameDirty, clamp } from '../engine/utils.js';
import { vec3 } from '../lib/gl-matrix/index.js';
import { addOnStart } from './app.js';

function updateLightPos(light, targetDist, lookAtTarget) {
  const { transform } = light;
  const pos = vec3.scale(
    vec3.create(), transform.forward, targetDist,
  );
  vec3.add(pos, pos, lookAtTarget);
  transform.position = pos;
}

export default class LightControl {
  constructor(app) {
    this.light = app.mainLight;
    this.direction = 1.0;
    this.mouseControl = false;
    this.lightRotate = 0;
    this.scale_speed = 1.0;
    this.target_dist = 3.0;
    this.scale_min = 1.0;
    this.scale_max = 10.0;
    this.maxDist = 0.0;
    this.look_at_target = vec3.create();
    this.alt = false;
    input.eventListeners.push(this);
    app.addEvent('OnLoadMesh', this.setScale, this);
  }

  OnKeyDown(e) {
    if (e.key === 'l') {
      this.lightRotate += 1;
      if (this.lightRotate > 2) this.lightRotate = 0;
      if (this.lightRotate > 0) this.direction *= -1.0;
      this.light.debug = this.lightRotate > 0;
    } else if (e.key === 'Shift') {
      this.mouseControl = true;
      this.light.debug = this.mouseControl;
      input.lockTarget = this;
    } else if (e.key === 'Alt') {
      this.alt = true;
    }
  }

  OnKeyUp() {
    this.mouseControl = false;
    this.alt = false;
    this.light.debug = this.mouseControl;
    input.lockTarget = undefined;
  }

  OnMouseOutWindow() {
    if (this.mouseControl) {
      this.mouseControlOut = true;
    }
    if (this.alt) {
      this.altOut = true;
    }
    this.mouseControl = false;
    this.alt = false;
    if (this.light.debug) {
      this.light.debug = this.mouseControl;
    }
  }

  OnMouseEnterWindow() {
    if (this.mouseControlOut) {
      this.mouseControlOut = false;
      this.mouseControl = input.keys.Shift !== 0;
      this.light.debug = this.mouseControl;
    }
    if (this.altOut) {
      this.altOut = false;
      this.alt = input.keys.Alt !== 0;
    }
  }

  OnTouchStart() {
    if (this.mouseControl) {
      return true;
    }
    return false;
  }

  OnTouch(e) {
    if (input.lockTarget !== this) return;

    if (e.x2 !== undefined && e.y2 !== undefined) {
      // two finger
      return;
    }

    const { deltaX, deltaY } = e;
    if (e.type === 4) {
      // wheel button
    } else if (this.alt) {
      this.target_dist -= (deltaX - deltaY) * timer.deltaTime * 0.05;
    } else {
      // left / right button
      const qEuler = this.light.transform.euler;

      qEuler[0] = clamp(qEuler[0] + (deltaY * 0.2), -90.0, 90.0);
      qEuler[1] += deltaX * 0.2;

      this.light.transform.euler = qEuler;
      // console.log(this.light.transform.euler);
    }
    updateLightPos(this.light, this.target_dist, this.look_at_target);
  }

  OnMouseWheel(d) {
    if (input.lockTarget !== this) return;

    let delta = d;
    if (delta === undefined || Number.isNaN(delta)) delta = 0;
    else delta = clamp(delta, -1.0, 1.0);

    delta *= this.scale_speed;
    this.target_dist = clamp(this.target_dist - delta, this.scale_min, this.scale_max);
    updateLightPos(this.light, this.target_dist, this.look_at_target);
    // console.log(`tar_dist ${this.target_dist}`);
  }

  update(dt) {
    if (this.lightRotate > 0) {
      setFrameDirty();
      const e = this.light.transform.euler;
      e[1] += dt * 30.0 * this.direction;
      this.light.transform.euler = e;
      updateLightPos(this.light, this.target_dist, this.look_at_target);
    }
  }

  setScale(nodes) {
    if (!this.light || !nodes[0]) return;
    const allMax = vec3.set(vec3.create(), Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE);
    const allMin = vec3.set(vec3.create(), Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
    nodes.forEach((node) => {
      // console.log(this.node.renderable.aabb);
      let min = node.renderable.aabb[0];
      let max = node.renderable.aabb[1];
      // console.log(this.node.transform);
      min = vec3.multiply(vec3.create(), min, node.transform.scale);
      max = vec3.multiply(vec3.create(), max, node.transform.scale);
      vec3.add(min, min, node.transform.position);
      vec3.add(max, max, node.transform.position);
      vec3.min(allMin, allMin, min);
      vec3.max(allMax, allMax, max);
    });

    const dist = vec3.distance(allMin, allMax);
    this.maxDist = dist;
    // console.log(`max dist ${this.maxDist}, dist: ${dist}`);
    const center = vec3.add(vec3.create(), allMin, allMax);
    vec3.scale(center, center, 0.5);
    this.look_at_target = center;

    const targetDist = dist * 1.5;
    this.target_dist = targetDist;

    const close = clamp(Math.min(Math.min(allMin[0], allMin[1]), allMin[2]), 1e-4, 1.0);

    allLights.forEach((light) => {
      const cam = light.camera;
      cam.orthoSize = dist * 0.6;
      // if (cam.zFar < dist * 5.0) {
      //   cam.zFar = dist * 5.0;
      //   console.slog(`zFar: ${cam.zFar}`);
      // }
      cam.zFar = dist * 5.0;
      // console.log(`zFar: ${cam.zFar}`);

      // if (cam.zNear > close) {
      //   cam.zNear = close;
      //   console.slog(`zNear: ${cam.zNear}`);
      // }
      cam.zNear = close;
      // console.log(`zNear: ${cam.zNear}`);

      cam.updateProjectionMatrix();
      updateLightPos(light, targetDist, center);
    });
    // console.log(`tar_dist ${this.target_dist}`);
  }
}

addOnStart(LightControl);

export { LightControl };
