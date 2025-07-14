import { Rect } from '../engine/UI/rect.js';
import { Button } from '../engine/UI/button.js';
import { Text } from '../engine/UI/text.js';
import { GameNode } from '../engine/gamenode.js';
import { resourceManager } from '../engine/resourcemanager.js';

export default class ModelControl {
  constructor(mesh) {
    this.mesh = mesh;

    let rect = new Rect(0, 0, 0, 0);
    rect.setRelative(0.7, 0.9, 0.2, 0.05);
    // this.slider = new Slider(rect);
    // this.slider.onChangeTargets.push(this);

    let rect2 = new Rect(0, 0, 0, 0);
    rect2.setRelative(0.7, 0.9, 0.1, 0.1);
    rect2.width = rect2.height;
    rect2.x = rect.x - rect2.width * 2;
    this.buttonLeft = new Button(rect2);
    this.buttonLeft.setText('<<');
    this.buttonLeft.setTextColor([1.0, 0.0, 0.0, 1.0]);
    this.buttonLeft.notify.push(this);

    rect2 = new Rect(0, 0, 0, 0);
    rect2.setRelative(0.9, 0.9, 0.1, 0.1);
    rect2.width = rect2.height;
    rect2.x = rect.x + rect.width + rect2.width;
    this.buttonRight = new Button(rect2);
    this.buttonRight.setText('>>');
    this.buttonRight.setTextColor([1.0, 0.0, 0.0, 1.0]);
    this.buttonRight.notify.push(this);

    rect = new Rect(0, 0, 0, 0);
    rect.setRelative(0.8, 0.9, 0.1, 0.1);
    const node = new GameNode(new Text(), 'model_control_text');
    this.text = node.renderable;
    this.text.rect = rect;

    this.blendShapeIndex = 0;

    this.fileIndex = 0;
    this.fileCount = -1;
    let file = null;
    do {
      if (this.fileIndex < 10) file = resourceManager.getZipFile(`0${this.fileIndex}.txt`);
      else file = resourceManager.getZipFile(`${this.fileIndex}.txt`);
      this.fileCount = this.fileIndex;
      this.fileIndex += 1;
    }
    while (file !== null && file !== undefined);
    this.fileIndex = 0;

    this.LoadWeights(resourceManager.getZipFile('00.txt').text);

    this.text.setText(this.CurrentName);
    this.text.setTextColor([1.0, 1.0, 1.0, 1.0]);
  }

  ResetAllWeight() {
    for (let i = this.mesh.blendWeights.length - 1; i >= 0; i -= 1) {
      this.mesh.SetBlendWeight(i, 0.0);
    }
  }

  SetWeight(name, value) {
    const index = this.mesh.BlendShapeNameToIndex(name);
    if (index !== -1) {
      this.mesh.SetBlendWeight(index, value * 0.01);
    } else {
      console.warn(`no index for ${name}`);
    }
  }

  LoadWeights(blendShapeDatas) {
    if (blendShapeDatas !== undefined) {
      const jsonObj = JSON.parse(blendShapeDatas);
      const dataOutput = jsonObj.outputDatas;
      for (let i = dataOutput.length - 1; i >= 0; i -= 1) {
        const { weights } = dataOutput[i];
        if (weights !== undefined) {
          for (let j = weights.length - 1; j >= 0; j -= 1) {
            this.SetWeight(weights[j].name, weights[j].value);
          }
        }
      }
    }
  }

  get CurrentName() { return this.fileIndex; }

  OnClick(button) {
    if (button === this.buttonLeft) {
      this.fileIndex -= 1;
      if (this.fileIndex < 0) this.fileIndex = this.fileCount - 1;
    } else if (button === this.buttonRight) {
      this.fileIndex += 1;
      if (this.fileIndex >= this.fileCount) this.fileIndex = 0;
    }

    this.text.setText(this.CurrentName);
    this.ResetAllWeight();
    if (this.fileIndex < 10) this.LoadWeights(resourceManager.getZipFile(`0${this.fileIndex}.txt`).data);
    else this.LoadWeights(resourceManager.getZipFile(`${this.fileIndex}.txt`).data);
  }

  OnValue(value) {
    this.mesh.SetBlendWeight(this.blendShapeIndex, value);
    this.text.setText(this.CurrentName);
  }
}
