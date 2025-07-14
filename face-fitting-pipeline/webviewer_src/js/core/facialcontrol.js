import { GameNode } from '../engine/gamenode.js';
import { Rect } from '../engine/UI/rect.js';
import { Button } from '../engine/UI/button.js';
import { Slider } from '../engine/UI/slider.js';
import { Text } from '../engine/UI/text.js';
import { Joystick } from '../engine/UI/joystick.js';
import { clamp, isMobile } from '../engine/utils.js';
import { SetControlEnabled } from './blendshapeControl.js';
import { addOnStart } from './app.js';
import { input } from '../engine/inputmanager.js';

export default class FacialControl {
  constructor(app) {
    app.addEvent('OnLoadMesh', this.CheckIsFacial, this);
    this.AUmap = [
      { name: 'AU1', items: ['browInnerUp_L', 'browInnerUp_R'] },
      { name: 'AU2', items: ['browOuterUp_L', 'browOuterUp_R'] },
      { name: 'AU4', items: ['browDown_L', 'browDown_R'] },
      { name: 'AU6', items: ['cheekSquint_L', 'cheekSquint_R'] },
      { name: 'AU9', items: ['noseSneer_L', 'noseSneer_R'] },
      { name: 'AU10', items: ['mouthShrugUpper'] },
      { name: 'AU11', items: ['mouthUpperUp_L', 'mouthUpperUp_R'] },
      { name: 'AU12', items: ['mouthSmile_L', 'mouthSmile_R'] },
      { name: 'AU14', items: ['mouthDimple_L', 'mouthDimple_R'] },
      { name: 'AU15', items: ['mouthFrown_L', 'mouthFrown_R'] },
      { name: 'AU16', items: ['mouthLowerDown_L', 'mouthLowerDown_R'] },
      { name: 'AU17', items: ['mouthShrugLower'] },
      { name: 'AU18', items: ['mouthPucker'] },
      { name: 'AU20', items: ['mouthStretch_L', 'mouthStretch_R'] },
      { name: 'AU22', items: ['mouthFunnel'] },
      { name: 'AU24', items: ['mouthPress_L', 'mouthPress_R'] },
      { name: 'AU27', items: ['jawOpen'] },
      { name: 'AU28', items: ['mouthRollLower', 'mouthRollUpper'] },
      { name: 'AU29', items: ['jawForward'] },
      { name: 'AU30L', items: ['jawLeft'] },
      { name: 'AU30R', items: ['jawRight'] },
      { name: 'AU33', items: ['cheekPuff_L', 'cheekPuff_R'] },
    ];
    input.eventListeners.push(this);
  }

  CheckIsFacial(nodes) {
    this.mesh = undefined;
    if (!nodes[0]) return;
    const mesh = nodes[0].renderable;
    if (!mesh || mesh.blendShapeSize <= 0) {
      return;
    }
    let contains = false;
    for (let j = mesh.blendShapeNames.length - 1; j >= 0; j -= 1) {
      for (let i = this.AUmap.length - 1; i >= 0; i -= 1) {
        if (this.AUmap[i].items.includes(mesh.blendShapeNames[j])) {
          contains = true;
          break;
        }
      }
      if (contains) break;
    }
    SetControlEnabled(!contains);
    if (contains) {
      this.init(mesh);
    }
    if (isMobile) this.setEnable(contains);
    else this.setEnable(false);
  }

  init(mesh) {
    if (this.mesh) {
      this.mesh = mesh;
    } else {
      this.mesh = mesh;
      this.state = 1;
      this.page = 0;

      const smallSize = isMobile ? 200 : 60;
      const largeSize = isMobile ? 400 : 100;

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
      this.stateButton = new Button(rect);
      // this.stateButton.setText("All");
      this.stateButton.setTextColor([1.0, 0.0, 0.0, 1.0]);
      this.stateButton.notify.push(this);

      rect = new Rect(0, 0, 0, 0);
      rect.setRelative(1.0, 0.9, 0.2, 0.1);
      rect.height = smallSize;
      rect.width = smallSize;
      rect.x -= rect.width + 20;
      this.stateButton.rect.width = rect.x - 20 - this.stateButton.rect.x;
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

      this.fixedModeItems = [];
      this.buttonActions = [];
      rect = new Rect();
      rect.setRelative(0.03, 0.575, 0.2, 0.2);
      rect.width = largeSize;
      rect.height = largeSize;
      this.createJoystickItem(rect, 'brow', [
        'browInnerUp_L',
        'browInnerUp_R',
        'browOuterUp_L',
        'browOuterUp_R',
        'browDown_L',
        'browDown_R',
      ]);

      rect = new Rect();
      rect.setRelative(0.03, 0.45, 0.2, 0.2);
      rect.width = largeSize;
      rect.height = largeSize;
      this.createJoystickItem(rect, 'eyes', [
        'eyeLookOut_R',
        'eyeLookIn_R',
        'eyeLookDown_R',
        'eyeLookUp_R',
        'eyeLookIn_L',
        'eyeLookOut_L',
        'eyeLookDown_L',
        'eyeLookUp_L',
      ]);

      rect = new Rect();
      rect.setRelative(0.03, 0.325, 0.2, 0.2);
      rect.width = largeSize;
      rect.height = largeSize;
      this.createJoystickItem(rect, 'nose', [
        'noseSneer_L',
        'noseSneer_R',
      ]);
      rect = new Rect();
      rect.setRelative(0.03, 0.2, 0.2, 0.2);
      rect.width = largeSize;
      rect.height = largeSize;
      this.createJoystickItem(rect, 'mouth', [
        'mouthRight',
        'mouthLeft',
        'mouthShrugLower',
        'mouthShrugUpper',
      ]);

      rect = new Rect();
      rect.setRelative(0.03, 0.2, 0.2, 0.2);
      rect.x += largeSize + 10;
      rect.y += smallSize;
      rect.width = smallSize;
      rect.height = 0.5 * smallSize;
      const puckerButton = new Button(rect);
      puckerButton.setText('pucker');
      puckerButton.setTextColor([1.0, 1.0, 0.0, 1.0]);
      puckerButton.notify.push(this);
      this.buttonActions.push((b, self) => {
        if (b === puckerButton) {
          const index = self.mesh.BlendShapeNameToIndex('mouthPucker');
          self.mesh.SetBlendWeight(index, self.mesh.blendWeights[index] === 1 ? 0 : 1);
        }
      });
      this.fixedModeItems.push(puckerButton);

      rect = new Rect();
      rect.setRelative(0.03, 0.2, 0.2, 0.2);
      rect.x += 2 * largeSize + 20;
      rect.width = largeSize;
      rect.height = largeSize;
      this.createJoystickItem(rect, 'jaw', [
        'jawForward',
        'jawOpen',
        'jawLeft',
        'jawRight',
        'mouthClose',
      ]);
      const pos = [rect.x, rect.y, rect.width, rect.height];
      rect = new Rect(pos[0], pos[1], pos[2], pos[3]);
      rect.x += largeSize + 10;
      rect.y += smallSize;
      rect.width = smallSize;
      rect.height = 0.5 * smallSize;
      const closeButton = new Button(rect);
      this.mouthCloseMode = false;
      closeButton.setText('close');
      closeButton.setTextColor([1.0, 1.0, 0.0, 1.0]);
      closeButton.notify.push(this);
      this.buttonActions.push((b, self) => {
        if (b === closeButton) {
          // eslint-disable-next-line no-param-reassign
          self.mouthCloseMode = !self.mouthCloseMode;
          self.mesh.SetBlendWeight(self.mesh.BlendShapeNameToIndex('mouthClose'), self.mouthCloseMode
            ? self.mesh.blendWeights[self.mesh.BlendShapeNameToIndex('jawOpen')] : 0);
        }
      });
      this.fixedModeItems.push(closeButton);

      rect = new Rect();
      rect.setRelative(0.03, 0.075, 0.2, 0.2);
      rect.width = largeSize;
      rect.height = largeSize;
      this.createJoystickItem(rect, 'mouth2', [
        'mouthSmile_L',
        'mouthSmile_R',
        'mouthFrown_L',
        'mouthFrown_R',
      ]);
    }

    this.cachedValues = [];
    this.onPage();
  }

  setEnable(yes) {
    if (this.mesh && this.enabled !== yes) {
      this.enabled = yes;
      this.stateButton.enabled = this.enabled;
      this.leftButton.enabled = this.enabled;
      this.rightButton.enabled = this.enabled;
      for (let i = 0; i < 16; i += 1) {
        this.sliders[i].enabled = this.enabled;
        this.texts[i].enabled = this.enabled;
      }
      this.fixedModeItems.forEach((val) => {
        const item = val;
        item.enabled = yes;
      });
    }
  }

  createJoystickItem(rect, name, keys, lockX, lockY) {
    const joystick = new Joystick(rect);
    if (lockX) joystick.lockX = true;
    if (lockY) joystick.lockY = true;
    joystick.onChangeTargets.push(this);
    const rx = rect.x;
    const rry = rect.relative[1];
    const smallSize = isMobile ? 200 : 60;
    const largeSize = isMobile ? 400 : 100;
    const rect1 = new Rect();
    rect1.setRelative(0, rry, 0.2, 0.2);
    rect1.x = rx + largeSize + 10;
    rect1.width = smallSize;
    rect1.height = 0.5 * smallSize;
    const button = new Button(rect1);
    button.setText(name);
    button.setTextColor([1.0, 1.0, 0.0, 1.0]);
    button.tag = 'joystick_button';
    button.notify.push(this);
    this.fixedModeItems.push(joystick);
    this.fixedModeItems.push(button);
    joystick.tag = name;
    this.buttonActions.push((b, self) => {
      if (b === button) {
        for (let i = 0; i < keys.length; i += 1) {
          self.mesh.SetBlendWeight(self.mesh.BlendShapeNameToIndex(keys[i]), 0);
          joystick.ResetThumb();
        }
      }
    });
  }

  get itemSize() {
    if (this.state === 0) {
      return this.mesh.blendShapeSize;
    }
    return this.AUmap.length;
  }

  get pageSize() {
    if (this.state === 0) {
      // eslint-disable-next-line radix
      return (parseInt(this.mesh.blendShapeSize / 16)
        + (this.mesh.blendShapeSize % 16 === 0 ? 0 : 1));
    }
    // eslint-disable-next-line radix
    return parseInt(this.AUmap.length / 16) + (this.AUmap.length % 16 === 0 ? 0 : 1);
  }

  getBlendShapeName(i) {
    if (this.state === 0) return this.mesh.blendShapeNames[i];
    const au = this.AUmap[i];
    let n = au.name;
    if (this.state === 2) {
      n += ' (';
      for (let j = 0, jmax = au.items.length; j < jmax; j += 1) {
        n += au.items[j];
        if (j !== jmax - 1) {
          n += ', ';
        }
      }
      n += ')';
    }
    return n;
  }

  onPage() {
    if (this.state === 3) {
      this.stateButton.setText('Fixed');
      for (let i = 0; i < 16; i += 1) {
        this.sliders[i].enabled = false;
        this.texts[i].enabled = false;
      }
      for (let i = 0; i < this.fixedModeItems.length; i += 1) {
        if (this.fixedModeItems[i].tag === 'joystick_button') this.OnClick(this.fixedModeItems[i]);
        this.fixedModeItems[i].enabled = true;
      }
    } else {
      for (let i = 0; i < this.fixedModeItems.length; i += 1) {
        if (this.fixedModeItems[i].enabled && this.fixedModeItems[i].tag === 'joystick_button') this.OnClick(this.fixedModeItems[i]);
        this.fixedModeItems[i].enabled = false;
      }
      if (this.state === 0) {
        this.stateButton.setText('All');
      } else if (this.state === 1) {
        this.stateButton.setText('AU');
      } else {
        this.stateButton.setText('AU Full');
      }
      // console.log(`onPage :${this.page} / ${this.pageSize}`);
      for (let j = 0, i; j < 16; j += 1) {
        i = j + this.page * 16;

        if (i < this.itemSize) {
          this.sliders[j].enabled = true;
          this.texts[j].enabled = true;
          const n = this.getBlendShapeName(i);
          if (this.state === 0) {
            // eslint-disable-next-line max-len
            this.sliders[j].setSliderValue(this.mesh.blendWeights[this.mesh.BlendShapeNameToIndex(n)]);
          } else this.sliders[j].setSliderValue(this.cachedValues[i] ? this.cachedValues[i] : 0);
          // console.log(i + ": " + n);
          this.texts[j].setText(`${i}:${n}`);
          this.texts[j].setTextColor([1.0, 1.0, 0.0, 1.0]);
        } else {
          this.sliders[j].enabled = false;
          this.texts[j].enabled = false;
        }
      }
    }
  }

  setAU(i, value) {
    const au = this.AUmap[i];
    for (let j = 0; j < au.items.length; j += 1) {
      this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex(au.items[j]), value);
    }
  }

  OnClick(button) {
    if (button === this.stateButton) {
      this.state += 1;
      if (this.state === 4) this.state = 0;
      this.page = 0;
      if (this.state === 0) this.cachedValues = [];
      this.onPage();
    }
    if (button === this.leftButton) {
      this.page -= 1;
      if (this.page < 0) this.page = this.pageSize - 1;
      this.onPage();
    } else if (button === this.rightButton) {
      this.page += 1;
      if (this.page >= this.pageSize) this.page = 0;
      this.onPage();
    }
    for (let i = this.buttonActions.length - 1; i >= 0; i -= 1) {
      this.buttonActions[i](button, this);
    }
  }

  OnValue(slider, value) {
    for (let i = this.sliders.length - 1, j; i >= 0; i -= 1) {
      if (this.sliders[i] === slider) {
        j = this.page * 16 + i;
        if (this.state === 0) {
          this.mesh.SetBlendWeight(j, value);
        } else {
          this.setAU(j, value);
          this.cachedValues[j] = value;
        }
        break;
      }
    }
  }

  OnJoystickValue(joystick, x, y) {
    if (joystick.tag === 'mouth') {
      if (x > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthRight'), 0);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthLeft'), x);
      } else if (x < 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthRight'), -x);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthLeft'), 0);
      }
      if (y > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthShrugLower'), y);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthShrugUpper'), y);
      } else if (y < 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthShrugLower'), 0);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthShrugUpper'), 0);
      }
    } else if (joystick.tag === 'nose') {
      this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('noseSneer_L'), clamp((y * 0.5 + 0.5) + x, 0, 1));
      this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('noseSneer_R'), clamp((y * 0.5 + 0.5) - x, 0, 1));
    } else if (joystick.tag === 'eyes') {
      // right eye
      if (x > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookIn_R'), x);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookOut_R'), 0);
      } else {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookIn_R'), 0);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookOut_R'), -x);
      }
      if (y > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookUp_R'), y);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookDown_R'), 0);
      } else {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookUp_R'), 0);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookDown_R'), -y);
      }

      // left eye
      if (x > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookIn_L'), 0);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookOut_L'), x);
      } else {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookIn_L'), -x);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookOut_L'), 0);
      }
      if (y > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookUp_L'), y);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookDown_L'), 0);
      } else {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookUp_L'), 0);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('eyeLookDown_L'), -y);
      }
    } else if (joystick.tag === 'brow') {
      if (x > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browDown_L'), clamp(-y, 0, 1));
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browDown_R'), clamp(-y - x, 0, 1));
      } else {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browDown_L'), clamp(-y + x, 0, 1));
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browDown_R'), clamp(-y, 0, 1));
      }
      if (x > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browInnerUp_L'), clamp(y, 0, 1));
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browInnerUp_R'), clamp(y - x, 0, 1));
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browOuterUp_L'), clamp(y, 0, 1));
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browOuterUp_R'), clamp(y - x, 0, 1));
      } else {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browInnerUp_L'), clamp(y + x, 0, 1));
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browInnerUp_R'), clamp(y, 0, 1));
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browOuterUp_L'), clamp(y + x, 0, 1));
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('browOuterUp_R'), clamp(y, 0, 1));
      }
    } else if (joystick.tag === 'jaw') {
      if (x > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('jawLeft'), x);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('jawRight'), 0);
      } else {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('jawLeft'), 0);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('jawRight'), -x);
      }
      if (y > 0) {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('jawForward'), y);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('jawOpen'), 0);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthClose'), 0);
      } else {
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('jawOpen'), -y);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('jawForward'), 0);
        this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthClose'), this.mouthCloseMode ? -y : 0);
      }
    } else if (joystick.tag === 'mouth2') {
      this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthSmile_L'), clamp(y - clamp(-x, 0, 1), 0, 1));
      this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthSmile_R'), clamp(y - clamp(x, 0, 1), 0, 1));
      this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthFrown_L'), clamp(-y - clamp(-x, 0, 1), 0, 1));
      this.mesh.SetBlendWeight(this.mesh.BlendShapeNameToIndex('mouthFrown_R'), clamp(-y - clamp(x, 0, 1), 0, 1));
    }
  }

  OnKeyDown(e) {
    if (e.key === 'b') {
      this.setEnable(!this.enabled);
    }
  }
}

addOnStart(FacialControl);

export { FacialControl };
