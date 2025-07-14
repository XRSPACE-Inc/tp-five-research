import { GameNode } from '../engine/gamenode.js';
import { Text } from '../engine/UI/text.js';
import { Rect } from '../engine/UI/rect.js';
import { UIContainer } from '../engine/UI/uicontainer.js';
import { addOnStart } from './app.js';

export default class TestContainer {
  constructor() {
    this.container = new UIContainer(new Rect(100, 100, 300, 300));
    let node = new GameNode(new Text().setText('aaa').setRectSize(100, 50), 'text');
    this.container.addWidget(node.renderable);
    node = new GameNode(new Text().setText('bbb').setRectSize(100, 50), 'text');
    this.container.addWidget(node.renderable);
  }
}

addOnStart(TestContainer);

export { TestContainer };
