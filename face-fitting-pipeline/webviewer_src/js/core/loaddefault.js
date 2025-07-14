import { GameNode } from '../engine/gamenode.js';
import { input } from '../engine/inputmanager.js';
import { Button } from '../engine/UI/button.js';
import { Rect } from '../engine/UI/rect.js';
import { Sprite } from '../engine/UI/sprite.js';
import { screenSize } from '../engine/utils.js';
import { addOnStart } from './app.js';

const allFolders = [
  'output',
];

export default class LoadDefault {
  constructor(app) {
    this.app = app;
    this.enabledTag = true;
    this.buttonWidth = 300;
    this.buttonHeight = 100;
    const src = new GameNode(new Sprite(), 'image_src');
    src.renderable.rect = new Rect(0, 300, 300, 300);
    src.renderable.depth = -2;
    src.renderable.enabled = false;
    this.src = src.renderable;
    this.returnButton = new Button(new Rect(0, screenSize[1] - 55, 50, 50)).setText('<<');
    this.returnButton.enabled = false;
    this.buttons = [];
    this.buttonImages = [];
    let x = 10;
    let y = 10;
    allFolders.forEach((folder) => {
      const rect = new Rect(x, y, this.buttonWidth, this.buttonHeight);
      const button = new Button(rect).setText(`${folder}\n(click me to load)`);
      button.onClick = () => {
        this.returnButton.enabled = true;
        this.EnableButtons(false);
        this.loadFolder(folder);
      };
      this.buttons.push(button);
      const image = new GameNode(new Sprite(`./webresources/${folder}/i_src.png`), `preview_img${folder}`);
      image.renderable.rect = new Rect(x, y, this.buttonHeight, this.buttonHeight);
      image.renderable.depth = 1;
      image.renderable.enabled = true;
      this.buttonImages.push(image.renderable);
      y += 5 + this.buttonHeight;
      if (y > screenSize[1]) {
        y = 10;
        x += 5 + this.buttonWidth;
      }
    });
    this.returnButton.onClick = () => {
      this.EnableButtons(true);
      this.returnButton.enabled = false;
    };
    input.eventListeners.push(this);
  }

  set enabled(yes) {
    if (this.enabledTag !== yes) {
      this.enabledTag = yes;
      if (!yes) {
        this.EnableButtons(yes);
      }
      this.returnButton.enabled = yes;
      this.src.enabled = yes;
    }
  }

  get enabled() {
    return this.enabledTag;
  }

  EnableButtons(yes) {
    // eslint-disable-next-line no-param-reassign
    this.buttons.forEach((b) => { b.enabled = yes; });
    // eslint-disable-next-line no-param-reassign
    this.buttonImages.forEach((i) => { i.enabled = yes; });
  }

  loadFolder(folder) {
    const { app } = this;
    const { src } = this;
    app.loadMesh(`./webresources/${folder}/mesh.obj`, undefined, (node) => {
      app.loadTexture('./webresources/shared_gloss.jpg');
      app.loadTexture('./webresources/shared_translucency.jpg');
      const nodes = [node];
      nodes[0].name = 'head';
      nodes[0].renderable.material.addOrUpdateDefine('SKIN', 1);
      app.loadTexture(`./webresources/${folder}/albedo.png`, undefined, nodes);
      app.loadTexture(`./webresources/${folder}/normal.png`, undefined, nodes);
      app.loadTexture('./webresources/shared_envir.png', undefined, nodes);

      app.loadMesh(`./webresources/${folder}/eye.obj`, undefined, (eyeNode) => {
        const eye = [eyeNode];
        eye[0].name = 'eye';
        app.loadTexture('./webresources/eyeBall_col.png', undefined, eye);
        app.loadTexture('./webresources/eyeBall_gloss.jpg', undefined, eye);
        app.loadTexture('./webresources/eyeBall_normal.jpg', undefined, eye);
        app.loadTexture('./webresources/eyeBall_specular.jpg', undefined, eye);

        app.loadMesh(`./webresources/${folder}/eyelen.obj`, undefined, (eyelenNode) => {
          const eyelen = [eyelenNode];
          eyelen[0].name = 'eyelen';
          eyelen[0].renderable.material.setBlendType(2);
          app.loadTexture('./webresources/eyeLensBlended_col.png', undefined, eyelen);
          app.loadTexture('./webresources/eyeLensBlended_gloss.jpg', undefined, eyelen);
          app.loadTexture('./webresources/eyeLensBlended_normal.png', undefined, eyelen);
          app.loadTexture('./webresources/shared_envir.png', undefined, eyelen);
        }, undefined, 'eyelen', 'pbr_specular');
      }, undefined, 'eye', 'pbr_specular');

      app.loadTexture(`./webresources/${folder}/specular.png`, undefined, nodes);
      app.loadTexture(`./webresources/${folder}/cavity.png`, undefined, nodes);

      src.texture.loadFromUrl(`./webresources/${folder}/i_src.png`);
      src.enabled = true;
    }, undefined, 'head');
  }

  OnKeyDown(e) {
    if (e.key === 'd') {
      this.enabled = !this.enabledTag;
    }
  }
}

addOnStart(LoadDefault);

export { LoadDefault };
