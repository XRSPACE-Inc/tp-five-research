import { Transform } from './transformation.js';

const gameNodeGroup = [];
const debugNodes = [];
const emptyGameNodeIDs = [];
let nodeID = -1;
let renderGroup = [];
let shadowGroup = [];
let renderGroupUI = [];
let needToSort = false;
let needToSortShadow = false;
let needToSortUI = false;

export default class GameNode {
  constructor(renderable, name = undefined, debugNode = false) {
    this.renderable = renderable;
    if (this.renderable !== null && this.renderable !== undefined) this.renderable.gameNode = this;
    this.transform = new Transform();
    this.components = [];
    this.id = GameNode.getEmptyID();
    this.ui = this.renderable.ui;
    if (debugNode) {
      this.name = (name !== undefined) ? name : `unnamed node (${this.id}): `;
    } else {
      this.name = `node (${this.id}): `;
      if (name !== undefined) this.name += name;
    }

    if (debugNode) debugNodes.push(this);
    gameNodeGroup[this.id] = this;
    GameNode.sortRenderingGroup();

    return this;
  }

  static getEmptyID() {
    if (emptyGameNodeIDs.length > 0) {
      return emptyGameNodeIDs.pop();
    }
    nodeID += 1;
    return nodeID;
  }

  static getGroup() {
    return gameNodeGroup;
  }

  static getDebugGroup() {
    return debugNodes;
  }

  static sortRenderingGroup() {
    needToSort = true;
    needToSortUI = true;
    needToSortShadow = true;
  }

  static getRenderingGroup() {
    if (needToSort) {
      needToSort = false;
      renderGroup = [];
      gameNodeGroup.forEach((node) => {
        if (node && node.renderable
          && !node.renderable.ui
          && node.renderable.material && node.renderable.material.blendType === 0) {
          renderGroup.push(node);
        }
      });
      gameNodeGroup.forEach((node) => {
        if (
          node
          && node.renderable
          && !node.renderable.ui
          && node.renderable.material
          && (node.renderable.material.blendType === 2
            || node.renderable.material.blendType === 1)) {
          renderGroup.push(node);
        }
      });
    }
    return renderGroup;
  }

  static getLightingGroup() {
    if (needToSortShadow) {
      needToSortShadow = false;
      shadowGroup = [];
      gameNodeGroup.forEach((node) => {
        if (node && node.renderable
          && !node.renderable.ui
          && node.renderable.castShadow
          && node.renderable.material && node.renderable.material.blendType === 0) {
          shadowGroup.push(node);
        }
      });
    }
    return shadowGroup;
  }

  static getRenderingGroupUI() {
    if (needToSortUI) {
      needToSortUI = false;
      renderGroupUI = [];
      gameNodeGroup.forEach((node) => {
        if (node && node.renderable
          && node.renderable.ui) {
          renderGroupUI.push(node);
        }
      });
      renderGroupUI = renderGroupUI.sort((a, b) => a.renderable.depth - b.renderable.depth);
    }
    return renderGroupUI;
  }

  static destroy(gameNode) {
    gameNodeGroup[gameNode.id] = null;
    emptyGameNodeIDs.push(gameNode.id);
    return true;
  }

  update(dt) {
    for (let c = this.components.length - 1; c >= 0; c -= 1) {
      const com = this.components[c];
      if (!com) {
        this.components.splice(c, 1);
      } else {
        com.update?.(dt);
      }
    }
  }

  render(camera, Light, viewMat, projMat, material) {
    if (this.renderable) {
      this.renderable.draw(this.transform.matrix, camera, Light, viewMat, projMat, material);
    }
  }

  addComponent(component) {
    if (component) {
      this.components.push(component);
    }
  }

  getComponent(componentName) {
    for (let c = this.components.length - 1; c >= 0; c -= 1) {
      if (this.components[c].constructor.name === componentName) return this.components[c];
    }
    return null;
  }

  removeComponent(componentName) {
    for (let c = this.components.length - 1; c >= 0; c -= 1) {
      if (this.components[c].constructor.name === componentName) {
        this.components.splice(c, 1);
        break;
      }
    }
  }
}

export { GameNode };
