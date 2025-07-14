import { isMobile } from '../engine/utils.js';
import { GameNode } from '../engine/gamenode.js';
import { Rect } from '../engine/UI/rect.js';
import { Button } from '../engine/UI/button.js';
import { Slider } from '../engine/UI/slider.js';
import { Text } from '../engine/UI/text.js';
import { addOnStart } from './app.js';
import { input } from '../engine/inputmanager.js';

let controlEnabled = true;
function SetControlEnabled(yes) {
  controlEnabled = yes;
}

export default class BlendShapeControl {
  constructor(app) {
    app.addEvent('OnLoadMesh', this.init, this);
    input.eventListeners.push(this);
  }

  init(nodes) {
    this.mesh = undefined;
    if (!nodes[0]) return;
    const mesh = nodes[0].renderable;
    if (!mesh || mesh.blendShapeSize <= 0) return;
    if (this.mesh) {
      this.mesh = mesh;
    } else {
      // if (!controlEnabled) return;
      this.mesh = mesh;
      this.page = 0;

      const smallSize = isMobile ? 200 : 60;
      // const largeSize = isMobile ? 400 : 100;

      let rect = new Rect(0, 0, 0, 0);
      rect.setRelative(0.7, 0.9, 0.2, 0.1);
      rect.height = smallSize;
      rect.width = smallSize;

      this.leftButton = new Button(rect);
      this.leftButton.setText('<');
      this.leftButton.setTextColor([1.0, 0.0, 0.0, 1.0]);
      this.leftButton.notify.push(this);

      rect = new Rect(0, 0, 0, 0);
      rect.setRelative(0.8, 0.9, 0.2, 0.1);
      rect.x = this.leftButton.rect.x + this.leftButton.rect.width + 20;
      rect.height = smallSize;

      rect = new Rect(0, 0, 0, 0);
      rect.setRelative(1.0, 0.9, 0.2, 0.1);
      rect.height = smallSize;
      rect.width = smallSize;
      rect.x -= rect.width + 20;
      this.rightButton = new Button(rect);
      this.rightButton.setText('>');
      this.rightButton.setTextColor([1.0, 0.0, 0.0, 1.0]);
      this.rightButton.notify.push(this);

      this.sliders = [];
      this.texts = [];
      let slider;
      let text;
      let node;
      for (let i = 0; i < 16; i += 1) {
        rect = new Rect(0, 0, 0, 0);
        rect.setRelative(0.89, 0.8 - i * 0.05, 0.1, 0.04);
        rect.height = Math.min(rect.height, smallSize);
        slider = new Slider(rect);
        slider.onChangeTargets.push(this);
        rect = new Rect(0, 0, 0, 0);
        rect.setRelative(0.67, 0.8 - i * 0.05, 0.2, 0.05);
        node = new GameNode(new Text(), `blendshape_control_text${i}`);
        text = node.renderable;
        text.rect = rect;
        this.sliders.push(slider);
        this.texts.push(text);
      }
    }
    // this.testButton = new Button(new Rect(200, 200, 200, 200));
    // const me = this;
    // this.testButton.onClick = () => {
    //   me.setEnable(!me.enabled);
    // };

    this.onPage();
    // console.log(`set enable to ${controlEnabled}`);
    if (isMobile) this.setEnable(controlEnabled);
    else this.setEnable(false);
  }

  setEnable(yes) {
    if (this.mesh && this.enabled !== yes) {
      this.enabled = yes;
      this.leftButton.enabled = this.enabled;
      this.rightButton.enabled = this.enabled;
      for (let i = 0; i < 16; i += 1) {
        this.sliders[i].enabled = this.enabled;
        this.texts[i].enabled = this.enabled;
      }
    }
  }

  get pageSize() {
    // eslint-disable-next-line radix
    return (parseInt(this.mesh.blendShapeSize / 16)
      + (this.mesh.blendShapeSize % 16 === 0 ? 0 : 1));
  }

  onPage() {
    // console.log(`onPage :${this.page} / ${this.pageSize}`);
    for (let j = 0, i; j < 16; j += 1) {
      i = j + this.page * 16;

      if (i < this.mesh.blendShapeSize) {
        this.sliders[j].enabled = true;
        this.texts[j].enabled = true;
        const n = this.mesh.blendShapeNames[i];
        this.sliders[j].setSliderValue(this.mesh.blendWeights[this.mesh.BlendShapeNameToIndex(n)]);
        // console.log(i + ": " + n);
        this.texts[j].setText(`${i}:${n}`);
        this.texts[j].setTextColor([1.0, 1.0, 0.0, 1.0]);
      } else {
        this.sliders[j].enabled = false;
        this.texts[j].enabled = false;
      }
    }
  }

  OnClick(button) {
    if (button === this.leftButton) {
      this.page -= 1;
      if (this.page < 0) this.page = this.pageSize - 1;
      this.onPage();
    } else if (button === this.rightButton) {
      this.page += 1;
      if (this.page >= this.pageSize) this.page = 0;
      this.onPage();
    }
  }

  OnValue(slider, value) {
    for (let i = this.sliders.length - 1, j; i >= 0; i -= 1) {
      if (this.sliders[i] === slider) {
        j = this.page * 16 + i;
        this.mesh.SetBlendWeight(j, value);
        break;
      }
    }
  }

  OnKeyDown(e) {
    if (e.key === 'b') {
      this.setEnable(!this.enabled);
    }
  }
}

addOnStart(BlendShapeControl);

export { BlendShapeControl, SetControlEnabled };
