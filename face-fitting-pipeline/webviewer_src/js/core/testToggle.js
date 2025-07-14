import { setFrameDirty } from '../engine/utils.js';
import { Button } from '../engine/UI/button.js';
import { Rect } from '../engine/UI/rect.js';
import { addOnStart } from './app.js';

export default class TestToggle {
  constructor(app) {
    this.renderable = app.node.renderable;
    this.state = 0.0;
    const rect = new Rect(0, 0, 0, 0).setRelative(0.0, 0.95, 0.05, 0.05);
    this.button = new Button(rect);
    this.button.setText(this.state);
    this.button.setTextColor([1.0, 0.0, 0.0, 1.0]);
    this.button.notify.push(this);
  }

  OnClick() {
    if (this.renderable.material) {
      this.state += 1.0;
      if (this.state >= 3.0) this.state = 0.0;
      this.button.setText(this.state);
      this.renderable.material.setUniformData('testFloat', this.state);
      setFrameDirty();
    }
  }
}

addOnStart(TestToggle);

export { TestToggle };
