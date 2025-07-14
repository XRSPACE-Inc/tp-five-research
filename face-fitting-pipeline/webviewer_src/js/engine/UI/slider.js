/* eslint-disable max-len */
import { Rect } from './rect.js';
import { GameNode } from '../gamenode.js';
import { Sprite } from './sprite.js';
import { input } from '../inputmanager.js';
import { clamp } from '../utils.js';

export default class Slider {
  constructor(rect) {
    this.enabledTag = true;
    this.rect = rect || new Rect(0, 0, 100, 30);
    this.rect.onChangeTargets.push(this);
    let node = new GameNode(new Sprite(), 'slider_thumb');
    this.thumb = node.renderable;
    this.thumb.depth = 1;
    node = new GameNode(new Sprite(), 'slider_background');
    this.background = node.renderable;
    this.background.color = [0.3, 0.3, 0.3, 1.0];
    this.background.depth = 0;
    this.OnRectChanged(this.rect);

    this.thumbPressing = false;
    this.sliderValue = 0;
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

  setSliderValue(value) {
    // console.log(value);
    this.sliderValue = value;
    this.thumb.rect.x = value * this.widthSize + this.rect.x;
  }

  OnTouchStart(e) {
    if (this.enabled === false) return false;
    if (this.rect.contains([e.x, e.y])) {
      this.thumbPressing = true;
      this.thumb.rect.x = clamp(e.x - this.thumb.rect.width * 0.5, this.rect.x, this.rect.x + this.widthSize);
      this.onValueChange((this.thumb.rect.x - this.rect.x) / this.widthSize);
      this.thumb.color = [0.6, 0.6, 0.6, 1.0];
      return true;
    }
    return false;
  }

  OnTouchEnd() {
    this.thumb.color = [1.0, 1.0, 1.0, 1.0];
    this.thumbPressing = false;
  }

  OnTouch(e) {
    if (this.enabled === false) return;
    if (this.thumbPressing) {
      this.thumb.rect.x = clamp(e.x - this.thumb.rect.width * 0.5, this.rect.x, this.rect.x + this.widthSize);
      this.onValueChange((this.thumb.rect.x - this.rect.x) / this.widthSize);
    }
  }

  OnMouseOutWindow() {
    if (this.thumbPressing) {
      this.thumb.color = [1.0, 1.0, 1.0, 1.0];
    }
  }

  OnMouseEnterWindow() {
    if (!input.mouseDown && this.thumbPressing) {
      this.thumbPressing = false;
    }
    if (this.thumbPressing) {
      this.thumb.color = [0.6, 0.6, 0.6, 1.0];
    }
  }

  OnRectChanged(rect) {
    this.rect = rect;
    // console.log(this.rect);
    this.background.rect.width = this.rect.width;
    this.background.rect.height = this.rect.height * 0.25;
    this.background.rect.x = this.rect.x;
    this.background.rect.y = this.rect.y + this.rect.height * 0.25 + this.background.rect.height * 0.5;
    // console.log(this.background.rect);

    this.thumb.rect.y = this.rect.y;
    this.thumb.rect.height = this.rect.height;
    this.thumb.rect.width = this.rect.width * 0.1;
    // this.thumb.rect.x = this.rect.x;
    this.thumb.rect.x = this.sliderValue * this.widthSize + this.rect.x;
    // console.log(this.thumb.rect);
  }

  onValueChange(sliderValue) {
    this.sliderValue = sliderValue;
    for (let i = this.onChangeTargets.length - 1; i >= 0; i -= 1) {
      if (this.onChangeTargets[i].OnValue !== undefined) this.onChangeTargets[i].OnValue(this, this.sliderValue);
    }
  }
}

export { Slider };
