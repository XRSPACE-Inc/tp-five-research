import { Light } from '../engine/light.js';
import { Sprite } from '../engine/UI/sprite.js';
import { GameNode } from '../engine/gamenode.js';
import { addOnStart } from './app.js';
import { Rect } from '../engine/UI/rect.js';

export default class TestFramebuffer {
  constructor(app) {
    this.cam = app.camera;
    this.light = new Light();
    this.light.transform.euler = [-45.0, 45.0, 45.0];
    // console.log(this.light.transform.forward);
    this.light.transform.position = this.cam.transform.position;
    // this.light.transform.rotation = this.cam.transform.rotation;
    this.light.color = [0.4, 0.38, 0.28];

    const node = new GameNode(new Sprite(undefined, this.light.renderTexture.texture), 'button_background');
    this.quad = node.renderable;
    this.quad.rect = new Rect(0, 0, 100, 100);
    const quadsize = 0.1;
    this.quad.rect.setRelative(-quadsize, -quadsize, quadsize, quadsize);
    // this.quad.color = [1.0, 1.0, 1.0, 0.3];
    // app.node.renderable.material.setUniformData('uAlbedo', this.light.renderTexture.texture);
  }

  update() {
    this.light.transform.position = this.cam.transform.position;
    // this.light.transform.rotation = this.cam.transform.rotation;
  }
}

addOnStart(TestFramebuffer);

export { TestFramebuffer };
