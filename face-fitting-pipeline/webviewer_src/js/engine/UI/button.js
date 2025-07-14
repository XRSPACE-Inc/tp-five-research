import { Rect } from './rect.js';
import { GameNode } from '../gamenode.js';
import { Text } from './text.js';
import { Sprite } from './sprite.js';
import { input } from '../inputmanager.js';
import { setFrameDirty } from '../utils.js';

export default class Button {
  constructor(rect) {
    this.enabledTag = true;
    this.rect = rect || new Rect(0, 0, 100, 30);
    this.rect.onChangeTargets.push(this);

    let node = new GameNode(new Text(), 'buttontextTag');
    this.text = node.renderable;
    node = new GameNode(new Sprite(), 'button_background');
    this.background = node.renderable;
    this.background.rect = this.rect;
    this.color = [1.0, 1.0, 1.0, 0.6];
    this.pressedColor = [0.5, 0.5, 0.5, 1.0];
    this.background.color = this.color;

    this.text.rect.set(this.rect.x, this.rect.y, this.rect.width, this.rect.height);

    this.pressing = false;
    this.onClick = undefined;
    this.notify = [];

    input.eventListeners.push(this);
  }

  set depth(d) {
    this.background.depth = d;
  }

  get depth() {
    return this.background.depth;
  }

  set enabled(yes) {
    if (this.enabledTag !== yes) {
      this.enabledTag = yes;
      this.background.enabled = this.enabledTag;
      this.text.enabled = this.enabledTag;
    }
  }

  get enabled() {
    return this.enabledTag;
  }

  setText(text) {
    // console.log(`change text to ${text}`);
    if (this.textTag !== text) {
      setFrameDirty();
      this.textTag = text;
      this.text.setText(text);
    }
    return this;
  }

  setTextColor(col) {
    this.text.setTextColor(col);
    return this;
  }

  setColor(col) {
    this.color = col;
    if (!this.pressing) this.background.color = this.color;
  }

  setPressedColor(col) {
    this.pressedColor = col;
    if (this.pressing) this.background.color = this.pressedColor;
  }

  OnTouchStart(e) {
    if (!this.enabledTag) return false;
    if (this.rect.contains([e.x, e.y])) {
      this.pressing = true;
      this.background.color = this.pressedColor;
      return true;
    }
    return false;
  }

  OnTouchEnd(e) {
    if (!this.enabledTag) return;
    if (this.pressing) {
      if (this.rect.contains([e.x, e.y])) {
        if (this.onClick !== undefined) this.onClick();
        const self = this;
        this.notify.forEach((t) => {
          if (t.OnClick !== undefined) t.OnClick(self);
        });
      }
      this.background.color = this.color;
    }
    this.pressing = false;
  }

  OnRectChanged(rect) {
    // console.log(`${this.text.text}: (${this.rect}), text: (${this.text.rect})`);
    this.rect = rect;
    this.text.rect.set(this.rect.x, this.rect.y, this.rect.width, this.rect.height);
  }

  OnMouseOutWindow() {
    if (this.pressing) {
      this.background.color = this.color;
    }
  }

  OnMouseEnterWindow() {
    if (!input.mouseDown && this.pressing) {
      this.pressing = false;
    }
    if (this.pressing) {
      this.background.color = this.pressedColor;
    }
  }
}

export { Button };
