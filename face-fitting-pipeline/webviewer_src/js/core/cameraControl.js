import { input } from '../engine/inputmanager.js';
import { setFrameDirty, clamp, isMobile } from '../engine/utils.js';
import { vec3 } from '../lib/gl-matrix/index.js';
import { Rect } from '../engine/UI/rect.js';
import { Button } from '../engine/UI/button.js';
import { DebugDraw } from '../engine/debugDraw.js';
import { timer } from '../engine/timer.js';
import { addOnStart } from './app.js';
// import Slider from '../engine/UI/slider.js';

export default class CameraControl {
  constructor(app) {
    this.lastTouchDistSqr = undefined;
    this.distance_to_object = 9.0;
    this.target_dist = 10.0;

    this.node = app.node;
    this.camera = app.camera;

    this.scale_min = 0.4;
    this.scale_max = 20.0;
    this.scale_speed = 1.0;
    this.scale_update_speed = 200.0;
    this.move_speed = 0.1;

    this.setScale();
    this.distance_to_object = this.target_dist - 0.01;
    this.look_at_target = vec3.create();
    this.look_at_target_offset = vec3.create();

    if (!isMobile) {
      const rect = new Rect(0, 0, 0, 0);
      rect.setRelative(0.0, 0.9, 0.2, 0.05);
      this.button1 = new Button(rect);
      this.button1.setText('Reset Camera');
      this.button1.setTextColor([1.0, 0.0, 0.0, 1.0]);
      this.button1.notify.push(this);
      this.button1.enabled = false;
    }

    this.alt = false;

    app.addEvent('OnLoadMesh', this.setScale, this);
    input.eventListeners.push(this);
  }

  OnClick(button) {
    if (this.button1 === button) {
      this.look_at_target_offset = vec3.create();
      this.distance_to_object = this.target_dist;
      const pos = vec3.scale(vec3.create(), this.camera.transform.forward, this.distance_to_object);
      vec3.add(pos, pos, this.look_at_target);
      vec3.add(this.camera.transform.position, pos, this.look_at_target_offset);
      this.button1.enabled = false;
      DebugDraw.get()?.clear();
    }
  }

  OnTouchStart(e) {
    this.lastTouchDistSqr = undefined;

    if (e.type === 4) {
      // wheel button
      DebugDraw.get()?.addLine([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]);
      DebugDraw.get()?.addLine([0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 1.0]);
      DebugDraw.get()?.addLine([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.0]);
    }

    return false;
  }

  // eslint-disable-next-line class-methods-use-this
  // OnTouchEnd() {
  //   DebugDraw.get()?.clear();
  // }

  update(dt) {
    if (this.distance_to_object !== this.target_dist) {
      setFrameDirty();
      const dir = Math.sign(this.target_dist - this.distance_to_object);
      this.distance_to_object += dir * dt * this.scale_update_speed;
      const newDir = Math.sign(this.target_dist - this.distance_to_object);
      if (dir !== newDir) this.distance_to_object = this.target_dist;
      const pos = vec3.scale(vec3.create(), this.camera.transform.forward, this.distance_to_object);
      vec3.add(pos, pos, this.look_at_target);
      vec3.add(this.camera.transform.position, pos, this.look_at_target_offset);
    }
  }

  OnMouseWheel(d) {
    if (input.lockTarget) return;
    let delta = d;
    if (delta === undefined || Number.isNaN(delta)) delta = 0;
    else delta = clamp(delta, -1.0, 1.0);

    delta *= this.scale_speed;
    this.target_dist = clamp(this.target_dist - delta, this.scale_min, this.scale_max);
  }

  OnTouch(e) {
    if (input.anyObjectTouched) return;
    if (input.lockTarget) return;

    if (e.x2 !== undefined && e.y2 !== undefined) {
      // two finger
      let dx = e.x - e.x2;
      let dy = e.y - e.y2;
      if (dx === undefined || Number.isNaN(dx)) dx = 0;
      if (dy === undefined || Number.isNaN(dy)) dy = 0;
      const distSqr = dx * dx + dy * dy;
      if (this.lastTouchDistSqr !== undefined) {
        this.OnMouseWheel(-(this.lastTouchDistSqr - distSqr) * timer.deltaTime * 0.0001);
      }
      this.lastTouchDistSqr = distSqr;
      return;
    }

    const { deltaX, deltaY } = e;
    if (e.type === 4) {
      // wheel button
      const target = this.look_at_target_offset;
      target[1] += deltaY * this.move_speed;
      const right = vec3.create();
      vec3.scale(right, this.camera.transform.right, -deltaX * this.move_speed);
      vec3.add(target, target, right);
      this.look_at_target_offset = target;
      this.button1.enabled = true;
    } else if (this.alt && e.type === 2) {
      this.distance_to_object -= (deltaX - deltaY) * timer.deltaTime * 0.05;
      this.target_dist = this.distance_to_object;
    } else {
      // left / right button
      const qEuler = this.camera.transform.euler;

      qEuler[0] = clamp(qEuler[0] - (deltaY * 0.2), -90.0, 90.0);
      qEuler[1] -= deltaX * 0.2;

      this.camera.transform.euler = qEuler;
    }
    const pos = vec3.scale(vec3.create(), this.camera.transform.forward, this.distance_to_object);
    vec3.add(pos, pos, this.look_at_target);
    vec3.add(this.camera.transform.position, pos, this.look_at_target_offset);
  }

  setScale(nodes) {
    if (!nodes) return;
    // console.log(nodes);
    const allMax = [Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE];
    const allMin = [Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE];
    // console.log(`min : ${allMin}, max : ${allMax}`);
    nodes.forEach((node) => {
      // console.log(node.renderable.aabb);
      let min = node.renderable.aabb[0];
      let max = node.renderable.aabb[1];

      // console.log(node.transform);
      min = vec3.multiply(vec3.create(), min, node.transform.scale);
      max = vec3.multiply(vec3.create(), max, node.transform.scale);
      vec3.add(min, min, node.transform.position);
      vec3.add(max, max, node.transform.position);

      vec3.min(allMin, allMin, min);
      vec3.max(allMax, allMax, max);
      // console.log(`min: ${min}, max: ${max}, allMin: ${allMin}, allMax: ${allMax}`);
    });

    this.look_at_target = vec3.set(
      vec3.create(),
      (allMin[0] + allMax[0]) * 0.5,
      (allMin[1] + allMax[1]) * 0.5,
      (allMin[2] + allMax[2]) * 0.5,
    );

    // console.log(`min: ${allMin}, max: ${allMax}, look_at: ${this.look_at_target}`);
    const dist = vec3.distance(allMin, allMax);
    // console.log(`dist: ${dist}`);
    this.scale_min = 0.2 * dist;
    this.scale_max = 3.0 * dist;
    this.scale_speed = clamp(dist * 0.1, 0.000001, 1000000.0);
    this.scale_update_speed = this.scale_speed * 20.0;
    // console.log(this.scale_update_speed);
    this.target_dist = dist;
    this.move_speed = dist * 0.001;
    this.distance_to_object = this.target_dist - 0.01;
    // console.log(this.camera);
    // console.log(`far: ${dist * 5.0}`);
    // if (this.camera.zFar < dist * 5.0) {
    //   this.camera.zFar = dist * 5.0;
    //   // console.log(`zFar: ${this.camera.zFar}`);
    // }
    this.camera.zFar = dist * 5.0;
    // console.slog(`zFar: ${this.camera.zFar}`);

    const close = clamp(Math.min(Math.min(allMin[0], allMin[1]), allMin[2]), 0.001, 1.0);
    // console.log(`close: ${close}`);
    // if (this.camera.zNear > close) {
    //   this.camera.zNear = close;
    //   // console.log(`zFar: ${this.camera.zNear}`);
    // }
    this.camera.zNear = close;
    // console.slog(`zNear: ${this.camera.zNear}`);

    this.camera.updateProjectionMatrix();
  }

  OnKeyDown(e) {
    if (e.key === 'Alt') {
      this.alt = true;
    }
  }

  OnKeyUp(e) {
    if (e.key === 'Alt') {
      this.alt = false;
    }
  }

  OnMouseOutWindow() {
    if (this.alt) {
      this.altOut = true;
    }
    this.alt = false;
  }

  OnMouseEnterWindow() {
    if (this.altOut) {
      this.altOut = false;
      this.alt = input.keys.Alt !== 0;
    }
  }
}

addOnStart(CameraControl);

export { CameraControl };
