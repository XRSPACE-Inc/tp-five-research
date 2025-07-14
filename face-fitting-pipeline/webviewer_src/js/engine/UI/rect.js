import { setFrameDirty, screenSize, resizeListeners } from '../utils.js';

export default class Rect {
  constructor(x = 0, y = 0, width = 1, height = 1) {
    this.parent = undefined;
    this.array = [0, 0, 1, 1];
    this.relative = [];
    this.onChangeTargets = [];
    this.set(x, y, width, height);
  }

  get x() { return this.array[0]; }

  set x(x) {
    if (this.array[0] !== x) {
      this.array[0] = x;
      if (this.relative.length > 0) this.relative[0] = undefined;
      this.onChanged();
    }
  }

  get y() { return this.array[1]; }

  set y(y) {
    if (this.array[1] !== y) {
      this.array[1] = y;
      if (this.relative.length > 0) this.relative[1] = undefined;
      this.onChanged();
    }
  }

  get width() { return this.array[2]; }

  set width(width) {
    if (this.array[2] !== width) {
      this.array[2] = width;
      if (this.relative.length > 0) this.relative[2] = undefined;
      this.onChanged();
    }
  }

  get height() { return this.array[3]; }

  set height(height) {
    if (this.array[3] !== height) {
      this.array[3] = height;
      if (this.relative.length > 0) this.relative[3] = undefined;
      this.onChanged();
    }
  }

  get toArray() {
    return this.array;
  }

  setRelative(x, y, width, height) {
    if (this.relative.length === 0) {
      this.relative = [0, 0, 0, 0];
      resizeListeners.push(this);
    }
    this.relative[0] = x;
    this.relative[1] = y;
    this.relative[2] = width;
    this.relative[3] = height;
    if (x) this.array[0] = (x >= 0 ? x : 1.0 + x) * screenSize[0];
    if (y) this.array[1] = (y >= 0 ? y : 1.0 + y) * screenSize[1];
    if (width) this.array[2] = (width > 0 ? width : -width) * screenSize[0];
    if (height) this.array[3] = (height > 0 ? height : -height) * screenSize[1];

    this.onChanged();
    return this;
  }

  set(x, y, width, height) {
    this.array[0] = (x >= 0 ? x : (screenSize[0] + x));
    this.array[1] = (y >= 0 ? y : (screenSize[1] + y));
    this.array[2] = width;
    this.array[3] = height;
    // console.log(`set: (${x}, ${y}, ${width}, ${height}) this.array: ${this.array}`);
    if (this.relative.length > 0) {
      this.relative.length = 0;
      const index = resizeListeners.indexOf(this);
      if (index > -1) {
        resizeListeners.splice(index, 1);
      }
    }
    // console.log(`screenSize : (${screenSize[0]}, ${screenSize[1]})`);
    this.onChanged();
    return this;
  }

  contains(vec) {
    return (
      vec[0] >= this.array[0]
      && vec[1] >= this.array[1]
      && vec[0] <= this.max[0]
      && vec[1] < this.max[1]
    );
  }

  OnResize(oldSize, newSize) {
    if (this.relative.length > 0) {
      const x = this.array[0] / oldSize[0];
      const y = this.array[1] / oldSize[1];
      const width = this.array[2] / oldSize[0];
      const height = this.array[3] / oldSize[1];
      this.array[0] = x * newSize[0];
      this.array[1] = y * newSize[1];
      if (this.relative.length > 2 && this.relative[2]) {
        this.array[2] = width * newSize[0];
      }
      if (this.relative.length > 3 && this.relative[3]) {
        this.array[3] = height * newSize[1];
      }
      this.onChanged();
    }
    return this;
  }

  onChanged() {
    setFrameDirty();
    this.min = [this.array[0], this.array[1]];
    this.max = [this.array[0] + this.array[2], this.array[1] + this.array[3]];
    for (let i = this.onChangeTargets.length - 1; i >= 0; i -= 1) {
      if (this.onChangeTargets[i].OnRectChanged) {
        this.onChangeTargets[i].OnRectChanged(this);
      }
    }
  }

  toLocal(vec) {
    return [vec[0] - this.min[0], vec[1] - this.min[1]];
  }

  toString() {
    return `(${this.x}, ${this.y}, ${this.width}, ${this.height})`;
  }
}

export { Rect };
