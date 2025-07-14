import { Widget } from './widget.js';
import { inverseDevicePixelRatio, realScreenSize } from '../utils.js';

const container = document.getElementById('container');
const textSizeCalculator = document.createRange();
export default class Text extends Widget {
  constructor() {
    super(null);
    this.enabledTag = true;

    this.comparer = [-1, -1];

    this.div = document.createElement('div');
    this.textNode = document.createTextNode('');
    this.div.className = 'text_id';
    this.div.style = 'white-space: pre;';
    this.style = this.div.style;
    this.style.left = `${Math.floor(this.rect.x * inverseDevicePixelRatio)}px`;
    this.style.top = `${Math.floor(realScreenSize[1] - (this.rect.y + this.rect.height) * inverseDevicePixelRatio)}px`;
    this.style.fontSize = '20px';
    this.style.color = 'rgba(0.0, 0.0, 0.0, 1.0)';
    this.color = [0.0, 0.0, 0.0, 1.0];
    this.text = '';
    this.div.appendChild(this.textNode);

    this.borders = [0, 0];
    this.useBorderX = true;
    this.useBorderY = true;
    this.textSize = [1, 1];

    container.appendChild(this.div);
    this.rect.onChangeTargets.push(this);
  }

  setText(text, updateRect = false) {
    if (this.text !== text) {
      this.text = text;
      this.textNode.nodeValue = this.text;

      this.OnRectChanged(this.rect, updateRect);
      if (!this.enabled) this.style.display = 'none';
      else this.style.display = 'block';
    }
    return this;
  }

  setTextColor(col) {
    this.color = col;
    this.style.color = `rgba(${this.color[0] * 255}, ${this.color[1] * 255}, ${this.color[2] * 255}, ${this.color[3] * 255})`;
    return this;
  }

  setTextSize(size, updateRect = false) {
    this.style.fontSize = `${size}px`;
    if (!this.inputText && updateRect) {
      textSizeCalculator.selectNodeContents(this.textNode);
      const rects = textSizeCalculator.getClientRects();
      const len = (this.text.match(/\n/g) || []).length;
      this.rect.set(this.rect.x, this.rect.y, rects[0].width, rects[0].height * len);
    }
    return this;
  }

  OnRectChanged(rect, updateRect) {
    this.rect = rect;

    textSizeCalculator.selectNodeContents(this.textNode);
    const rects = textSizeCalculator.getClientRects();
    if (rects.length > 0) {
      const len = (this.text.match(/\n/g) || [0]).length;
      const rw = rects[0].width;
      const rh = rects[0].height * len;
      this.textSize[0] = rw;
      this.textSize[1] = rh;
      this.borders[0] = (this.rect.width - rw) * 0.5;
      this.borders[1] = (this.rect.height - rh) * 0.5;
      if (updateRect) {
        this.rect.set(this.rect.x, this.rect.y, rw, rh);
      }
    }

    if (this.useBorderX) this.style.left = `${Math.floor((this.rect.x + this.borders[0]))}px`;
    else this.style.left = `${Math.floor((this.rect.x))}px`;
    if (this.useBorderY) this.style.top = `${Math.floor(realScreenSize[1] - (this.rect.y + this.textSize[1] + this.borders[1]))}px`;
    else this.style.top = `${Math.floor(realScreenSize[1] - (this.rect.y + this.textSize[1]))}px`;
  }

  // since we using html to draw text...
  draw() {
    if (this.comparer[0] !== this.rect.x || this.comparer[1] !== this.rect.y) {
      this.comparer[0] = this.rect.x;
      this.comparer[1] = this.rect.y;
      this.OnRectChanged(this.rect, false);
    }
    if (this.enabled !== this.enabledTag) {
      this.enabledTag = this.enabled;
      if (!this.enabled) this.style.display = 'none';
      else this.style.display = 'block';
    }
  }
}

export { Text };
