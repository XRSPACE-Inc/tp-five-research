/* eslint-disable prefer-destructuring */
/* eslint-disable max-len */
import { Rect } from './rect.js';
import { GameNode } from '../gamenode.js';
import { Sprite } from './sprite.js';
import { input } from '../inputmanager.js';
import { clamp } from '../utils.js';
import { vec2 } from '../../lib/gl-matrix/index.js';

export default class Joystick {
  constructor(rect, type = 0) {
    this.enabledTag = true;
    this.type = type;
    this.rect = rect || new Rect(0, 0, 100, 30);
    this.rect.onChangeTargets.push(this);
    let node = new GameNode(new Sprite(undefined, undefined, 'sprite_circle'), 'joystick_thumb');
    this.thumb = node.renderable;
    const shader = type === 0 ? 'sprite' : 'sprite_circle';
    node = new GameNode(new Sprite(undefined, undefined, shader), 'joystick_background');
    this.background = node.renderable;
    this.background.color = [1.0, 1.0, 1.0, 0.2];
    this.center = [0, 0];
    this.OnRectChanged(this.rect);

    this.thumb.rect.x = this.center[0];
    this.thumb.rect.y = this.center[1];
    this.thumb.color = [1.0, 1.0, 1.0, 0.5];

    this.thumbPressing = false;
    this.lockX = false;
    this.lockY = false;
    this.x = 0;
    this.y = 0;
    this.radius = Math.min(this.rect.width, this.rect.height) / 2.0;
    this.onChangeTargets = [];

    input.eventListeners.push(this);
  }

  set enabled(yes) {
    if (this.enabledTag !== yes) {
      this.enabledTag = yes;
      this.thumb.enabled = this.enabledTag;
      this.background.enabled = this.enabledTag;
    }
  }

  get enabled() {
    return this.enabledTag;
  }

  get widthSize() { return this.rect.width - this.thumb.rect.width; }

  get heightSize() { return this.rect.height - this.thumb.rect.height; }

  setThumbPosition(e) {
    if (this.type === 0) {
      if (!this.lockX) this.thumb.rect.x = clamp(e.x - this.thumb.rect.width * 0.5, this.rect.x, this.rect.x + this.widthSize);
      if (!this.lockY) this.thumb.rect.y = clamp(e.y - this.thumb.rect.height * 0.5, this.rect.y, this.rect.y + this.heightSize);
      this.onValueChange((2.0 * (this.thumb.rect.x - this.center[0])) / this.widthSize, (2.0 * (this.thumb.rect.y - this.center[1])) / this.heightSize);
    } else {
      let dir = [e.x - this.center[0] - this.thumb.rect.width * 0.5, e.y - this.center[1] - this.thumb.rect.height * 0.5];
      if (vec2.length(dir) > this.radius) {
        dir = vec2.normalize(dir, dir);
        dir = vec2.scale(dir, dir, this.radius);
      }
      if (this.lockX) dir[0] = 0;
      if (this.lockY) dir[1] = 0;
      this.thumb.rect.x = this.center[0] + dir[0];
      this.thumb.rect.y = this.center[1] + dir[1];
      this.onValueChange(dir[0] / this.radius, dir[1] / this.radius);
    }
  }

  OnTouchStart(e) {
    if (this.enabled === false) return false;
    if (this.rect.contains([e.x, e.y])) {
      this.thumbPressing = true;
      this.setThumbPosition(e);
      this.thumb.color = [1.0, 1.0, 1.0, 0.2];
      for (let i = this.onChangeTargets.length - 1; i >= 0; i -= 1) {
        if (this.onChangeTargets[i].OnJoystickTouchStart !== undefined) this.onChangeTargets[i].OnJoystickTouchStart(this);
      }
      return true;
    }
    return false;
  }

  OnTouchEnd() {
    this.thumb.color = [1.0, 1.0, 1.0, 0.5];
    this.thumbPressing = false;
    for (let i = this.onChangeTargets.length - 1; i >= 0; i -= 1) {
      if (this.onChangeTargets[i].OnJoystickTouchEnd !== undefined) this.onChangeTargets[i].OnJoystickTouchEnd(this);
    }
  }

  OnTouch(e) {
    if (this.enabled === false) return;
    if (this.thumbPressing) {
      this.setThumbPosition(e);
    }
  }

  OnMouseOutWindow() {
    if (this.thumbPressing) {
      this.thumb.color = [1.0, 1.0, 1.0, 0.5];
    }
  }

  OnMouseEnterWindow() {
    if (!input.mouseDown && this.thumbPressing) {
      this.thumbPressing = false;
    }
    if (this.thumbPressing) {
      this.thumb.color = [1.0, 1.0, 1.0, 0.2];
    }
  }

  OnRectChanged(rect) {
    this.rect = rect;
    this.radius = Math.min(this.rect.width, this.rect.height) / 2.0;
    this.background.material.setUniformData('radius', this.radius);
    // console.log(this.rect);
    this.background.rect.width = this.rect.width;
    this.background.rect.height = this.rect.height;
    this.background.rect.x = this.rect.x;
    this.background.rect.y = this.rect.y;
    // console.log(this.background.rect);

    this.thumb.rect.y = this.rect.y;
    this.thumb.rect.height = this.rect.height * 0.3;
    this.thumb.rect.width = this.rect.width * 0.3;
    this.thumb.rect.x = this.rect.x;
    // console.log(this.thumb.rect);
    this.center = [
      this.rect.x + this.rect.width * 0.5 - this.thumb.rect.width * 0.5,
      this.rect.y + this.rect.height * 0.5 - this.thumb.rect.height * 0.5,
    ];
  }

  onValueChange(x, y) {
    this.x = x;
    this.y = y;
    // console.log(x + ", " + y);
    for (let i = this.onChangeTargets.length - 1; i >= 0; i -= 1) {
      if (this.onChangeTargets[i].OnJoystickValue !== undefined) this.onChangeTargets[i].OnJoystickValue(this, this.x, this.y);
    }
  }

  ResetThumb() {
    this.thumb.rect.x = this.center[0];
    this.thumb.rect.y = this.center[1];
  }
}

export { Joystick };
