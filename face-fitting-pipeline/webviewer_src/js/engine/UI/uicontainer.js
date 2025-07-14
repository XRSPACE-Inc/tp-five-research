import { GameNode } from '../gamenode.js';
import { input } from '../inputmanager.js';
import { Rect } from './rect.js';
import { Sprite } from './sprite.js';
import { Text } from './text.js';
import {
  clamp, screenSize, resizeListeners, flagContains,
} from '../utils.js';

const AnchorType = Object.freeze(
  {
    Free: 0,
    Left: 1,
    Right: 2,
    TopLeft: 5,
    TopRight: 6,
    BottomLeft: 9,
    BottomRight: 10,
  },
);

const Anchor = Object.freeze(
  {
    Free: 0,
    Left: 1,
    Right: 2,
    Top: 4,
    Bottom: 8,
  },
);

export default class UIContainer {
  constructor(rect) {
    this.enabledTag = true;
    this.rect = rect || new Rect(0, 0, 100, 100);
    // this.rect.onChangeTargets.push(this);

    this.color = [0.3, 0.3, 0.6, 0.3];
    this.pressedColor = [0.1, 0.1, 0.6, 0.3];

    let node = new GameNode(new Sprite(), 'uicontainer_background');
    this.background = node.renderable;
    this.background.color = this.color;

    node = new GameNode(new Text().setText('container').setTextColor([0.6, 0.6, 0.9, 1.0]), 'uicontainer_title');
    this.title = node.renderable;

    this.widgets = [];

    this.allowGrowth = false;
    this.border = [15, 45, 15, 15];
    this.realRect = new Rect(
      this.rect.x,
      this.rect.y,
      this.rect.width + this.border[0] + this.border[2],
      this.rect.height + this.border[1] + this.border[3],
    );
    this.background.rect = this.realRect;
    this.Refresh();

    this.title.rect = new Rect(0, 0, this.realRect.width, 50);

    this.canDrag = true;
    this.pressing = false;
    this.pressOffset = undefined;
    this.anchorTypeTag = AnchorType.Free;
    input.eventListeners.push(this);
    resizeListeners.push(this);
  }

  set enabled(yes) {
    if (this.enabledTag !== yes) {
      this.enabledTag = yes;
      this.title.enabled = this.enabledTag;
      this.background.enabled = this.enabledTag;
      this.widgets.forEach((v) => {
        const w = v;
        w.enabled = this.enabledTag;
      });
      this.Refresh();
    }
  }

  get enabled() {
    return this.enabledTag;
  }

  set anchorType(type) {
    if (this.anchorTypeTag !== type) {
      this.anchorTypeTag = type;
      // this.canDrag = type !== 0;
      this.Refresh();
    }
  }

  get anchorType() {
    return this.anchorTypeTag;
  }

  addWidget(w) {
    this.widgets.push(w);
    this.Refresh();
  }

  Refresh(screen = undefined) {
    const size = screen || screenSize;
    const oldHeight = this.realRect.height;
    if (this.allowGrowth) {
      let h = this.border[1] + this.border[3];
      let w = this.rect.width + this.border[0] + this.border[2];
      this.widgets.forEach((v) => {
        h += v.rect.height + 5;
        w = Math.max(w, v.rect.width + this.border[0] + this.border[2]);
      });
      this.realRect.width = Math.max(w, this.rect.width);
      this.realRect.height = Math.max(h, this.rect.height);
    }
    if (this.anchorTypeTag !== AnchorType.Free) {
      const [sw, sh] = size;
      if (this.anchorTypeTag === Anchor.Left) {
        this.realRect.x = 0;
        this.realRect.y = 0;
        this.realRect.height = sh;
      } else if (this.anchorTypeTag === Anchor.Right) {
        this.realRect.x = sw - this.realRect.width;
        this.realRect.y = 0;
        this.realRect.height = sh;
      } else {
        if (flagContains(this.anchorTypeTag, Anchor.Left)) {
          this.realRect.x = 0;
        } else if (flagContains(this.anchorTypeTag, Anchor.Right)) {
          this.realRect.x = sw - this.realRect.width;
        }
        if (flagContains(this.anchorTypeTag, Anchor.Top)) {
          this.realRect.y = sh - this.realRect.height;
        } else if (flagContains(this.anchorTypeTag, Anchor.Bottom)) {
          this.realRect.y = 0;
        }
      }
    } else if (oldHeight !== this.realRect.height) {
      this.realRect.y -= this.realRect.height - oldHeight;
    }
    // console.slog('===========');
    // console.log(`x: ${this.realRect.x}, y: ${this.realRect.y}`);

    let { x, y } = this.realRect;
    x += this.border[0];
    y += this.realRect.height - this.border[1];
    this.widgets.forEach((v) => {
      y -= v.rect.height + 5;
      v.rect.set(x, y, v.rect.width, v.rect.height);
    });

    this.title.rect.x = this.realRect.x;
    this.title.rect.y = this.realRect.y + this.realRect.height - this.title.rect.height;
    // console.log(`x: ${this.realRect.x}, y: ${this.realRect.y}`);
  }

  OnTouchStart(e) {
    if (!this.enabledTag) return false;
    if (!this.canDrag) return false;
    if (this.realRect.contains([e.x, e.y])) {
      this.pressOffset = this.realRect.toLocal([e.x, e.y]);
      this.pressing = true;
      this.background.color = this.pressedColor;
      return true;
    }
    return false;
  }

  OnTouchEnd() {
    if (this.pressing) {
      this.pressing = false;
      this.background.color = this.color;
    }
  }

  OnTouch(e) {
    if (!this.enabledTag) return;
    if (!this.canDrag) return;
    if (this.pressing) {
      this.realRect.x = clamp(e.x - this.pressOffset[0], 0, screenSize[0] - this.realRect.width);
      this.realRect.y = clamp(e.y - this.pressOffset[1], 0, screenSize[1] - this.realRect.height);
      // console.log(this.realRect.toString());
      this.Refresh();
    }
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

  OnResize(s, newSize) {
    // console.log(this);
    this.Refresh(newSize);
  }
}

export { UIContainer, AnchorType };
