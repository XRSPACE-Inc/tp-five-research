import { Widget } from './widget.js';
import { inverseDevicePixelRatio, realScreenSize } from '../utils.js';
import { input } from '../inputmanager.js';

const TextFieldType = Object.freeze(
  {
    String: 0,
    Integer: 1,
    Float: 2,
  },
);

const container = document.getElementById('container');
export default class TextField extends Widget {
  constructor(textFieldType = 0) {
    super(null);
    this.textFieldType = textFieldType;
    this.enabledTag = true;

    this.comparer = [-1, -1];

    this.input = document.createElement('INPUT');
    this.input.onchange = this.OnInputText.bind(this);
    this.input.size = 10;
    this.div = document.createElement('div');
    this.textNode = document.createTextNode('');
    this.textNodeDiv = document.createElement('span');
    this.textNodeDiv.className = 'textNode_id';
    this.textNodeDiv.style = 'white-space: pre;';
    this.div.className = 'text_id';
    // this.div.style = 'white-space: pre;';
    this.style = this.div.style;
    const halfWidth = this.rect.width * 0.4;
    this.style.left = `${Math.floor(this.rect.x * inverseDevicePixelRatio)}px`;
    // this.style.width = `${Math.floor(this.rect.width * inverseDevicePixelRatio)}px`;
    this.style.height = `${Math.floor(this.rect.height * inverseDevicePixelRatio)}px`;
    this.style.top = `${Math.floor(realScreenSize[1] - (this.rect.y + this.rect.height) * inverseDevicePixelRatio)}px`;
    this.style.fontSize = '20px';
    this.style.color = 'rgba(0.0, 0.0, 0.0, 1.0)';
    this.color = [0.0, 0.0, 0.0, 1.0];
    this.text = '';
    this.textNodeStyle = this.textNodeDiv.style;
    this.textNodeStyle.display = 'inline-block';
    this.textNodeStyle.width = `${Math.floor(halfWidth * inverseDevicePixelRatio)}px`;
    this.inputStyle = this.input.style;
    this.inputStyle.width = `${Math.floor(halfWidth * inverseDevicePixelRatio)}0px`;
    this.inputStyle.fontSize = '20px';
    this.inputStyle.color = 'rgba(255, 255, 255, 1.0)';
    this.textNodeDiv.appendChild(this.textNode);
    this.div.appendChild(this.textNodeDiv);
    this.div.appendChild(this.input);

    this.borders = [0, 0];
    this.useBorderX = true;
    this.useBorderY = true;
    this.textSize = [1, 1];

    container.appendChild(this.div);
    this.rect.onChangeTargets.push(this);
    this.backgroundColor = 'rgba(0.0, 0.0, 0.0, 0.5)';
    this.backgroundColorPressed = 'rgba(0.0, 0.0, 0.0, 0.8)';
    this.pressing = false;
    this.inputStyle.backgroundColor = this.backgroundColor;
    this.borders[1] = this.rect.height;
    input.eventListeners.push(this);
    this.onInputText = undefined;
    this.onInputTextTargets = [];
    this.floatFractionDigits = 2;
  }

  OnInputText() {
    const { value } = this.input;
    if (this.onInputText) {
      this.onInputText(this, value);
    }
    this.onInputTextTargets.forEach((tar) => {
      if (tar.OnInputText) tar.OnInputText(this, value);
    });
  }

  setText(text) {
    if (this.text !== text) {
      this.text = text;
      this.textNode.nodeValue = this.text;

      this.OnRectChanged(this.rect);
    }
    return this;
  }

  setFieldText(text) {
    switch (this.textFieldType) {
      case TextFieldType.Integer:
        this.input.value = parseInt(text, 10);
        break;
      case TextFieldType.Float:
        this.input.value = parseFloat(text).toFixed(this.floatFractionDigits);
        break;
      default:
        this.input.value = text;
        break;
    }
  }

  setTextColor(col) {
    this.color = col;
    this.style.color = `rgba(${this.color[0] * 255}, ${this.color[1] * 255}, ${this.color[2] * 255}, ${this.color[3]})`;
    this.inputStyle.color = `rgba(${this.color[0] * 255}, ${this.color[1] * 255}, ${this.color[2] * 255}, ${this.color[3]})`;
    return this;
  }

  setTextSize(size) {
    const pxSize = `${size}px`;
    this.inputStyle.fontSize = pxSize;
    this.style.fontSize = pxSize;
    return this;
  }

  OnRectChanged(rect) {
    this.rect = rect;

    const halfWidth = this.rect.width * 0.4;
    this.style.left = `${Math.floor(this.rect.x * inverseDevicePixelRatio)}px`;
    // this.style.width = `${Math.floor(halfWidth * inverseDevicePixelRatio)}px`;
    this.style.height = `${Math.floor(this.rect.height * inverseDevicePixelRatio)}px`;
    this.style.top = `${Math.floor(realScreenSize[1] - (this.rect.y + this.rect.height) * inverseDevicePixelRatio)}px`;
    this.textNodeStyle.width = `${Math.floor(halfWidth * inverseDevicePixelRatio)}px`;
    this.inputStyle.width = `${Math.floor(halfWidth * inverseDevicePixelRatio)}px`;
  }

  // since we using html to draw text...
  draw() {
    if (this.comparer[0] !== this.rect.x || this.comparer[1] !== this.rect.y) {
      this.comparer[0] = this.rect.x;
      this.comparer[1] = this.rect.y;
      this.OnRectChanged(this.rect);
    }
    if (this.enabled !== this.enabledTag) {
      this.enabledTag = this.enabled;
      if (!this.enabled) this.style.display = 'none';
      else this.style.display = 'block';
    }
  }

  // for input text
  OnTouchStart(e) {
    if (!this.enabledTag) return false;
    // console.log(`this.rect: ${this.rect}, e: (${e.x}, ${e.y})`);
    if (this.rect.contains([e.x, e.y])) {
      this.pressing = true;
      this.inputStyle.backgroundColor = this.backgroundColorPressed;
      return true;
    }
    return false;
  }

  OnTouch(e) {
    if (!this.enabledTag) return;
    if (!this.pressing) return;
    // not work for string.
    if (this.textFieldType === TextFieldType.String) return;
    if (e.x2 !== undefined && e.y2 !== undefined) return;
    const { deltaX } = e;
    if (deltaX !== 0) {
      let { value } = this.input;
      if (this.textFieldType === TextFieldType.Integer) {
        value = parseInt(value, 10);
        if (deltaX > 0) {
          this.setFieldText(value + 1);
        } else {
          this.setFieldText(value - 1);
        }
      } else if (this.textFieldType === TextFieldType.Float) {
        value = parseFloat(value);
        this.setFieldText(value + deltaX * 0.01);
      }
      this.OnInputText();
    }
  }

  OnTouchEnd(e) {
    if (!this.enabledTag) return;
    if (this.pressing) {
      if (this.rect.contains([e.x, e.y])) {
        this.input.focus();
      }
      this.inputStyle.backgroundColor = this.backgroundColor;
    }
    this.pressing = false;
  }

  OnMouseOutWindow() {
    if (this.pressing) {
      this.inputStyle.backgroundColor = this.backgroundColor;
    }
  }

  OnMouseEnterWindow() {
    if (!input.mouseDown && this.pressing) {
      this.pressing = false;
    }
    if (this.pressing) {
      this.inputStyle.backgroundColor = this.backgroundColorPressed;
    }
  }
}

export { TextField, TextFieldType };
