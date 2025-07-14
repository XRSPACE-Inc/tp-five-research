import { GameNode } from '../engine/gamenode.js';
import { Rect } from '../engine/UI/rect.js';
import { Slider } from '../engine/UI/slider.js';
import { Text } from '../engine/UI/text.js';
import { addOnStart } from './app.js';

export default class TestSlider {
  constructor(app) {
    this.renderable = app.node.renderable;
    this.slider = new Slider(new Rect(0, 0, 1, 1).setRelative(0.03, 0.5, 0.1, 0.05));
    this.slider.onChangeTargets.push(this);
    this.value = 0.0;
    const node = new GameNode(new Text(), 'testSlider_text_node');
    this.text = node.renderable;
    this.text.rect = new Rect().setRelative(0.03, 0.55, 0.1, 0.05);
    this.text.setText('0');
    this.text.setTextColor([1.0, 1.0, 1.0, 1.0]);
    this.scale = 1.0;
  }

  OnValue(slider, value) {
    if (this.renderable.material) {
      if (this.value !== value) {
        this.value = value;
        // console.log(value * 10000.0);
        this.renderable.material.setUniformData('uFresnelOcc', this.value * this.scale);
        this.text.setText(this.value * this.scale);
      }
    }
  }
}

addOnStart(TestSlider);

export { TestSlider };
