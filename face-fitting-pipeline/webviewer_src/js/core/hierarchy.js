/* eslint-disable class-methods-use-this */
import { GameNode } from '../engine/gamenode.js';
import { input } from '../engine/inputmanager.js';
import { Button } from '../engine/UI/button.js';
import { Text } from '../engine/UI/text.js';
import { Rect } from '../engine/UI/rect.js';
import { UIContainer, AnchorType } from '../engine/UI/uicontainer.js';
import { addOnStart } from './app.js';
import { TextField, TextFieldType } from '../engine/UI/textField.js';
import { getDefaultTextures } from '../engine/texture.js';

export default class Hierarchy {
  constructor(a) {
    const app = a;
    this.hierarchy = new UIContainer(new Rect(0, 100, 300, 300));
    this.hierarchy.allowGrowth = true;
    this.hierarchy.title.setText('Nodes');
    this.hierarchy.pressedColor = this.hierarchy.color;
    this.addNodeButton = new Button(new Rect(0, 0, 300, 25));
    this.addNodeButton.setText('Create');
    this.addNodeButton.notify.push(this);
    this.CreateNewNode = function onCreateNewNode() {
      this.RefreshLater();
      return app.GetNextNode();
    };
    this.hierarchy.addWidget(this.addNodeButton);
    this.itemButtons = [];
    this.nodeStatus = {};

    this.lastClicked = undefined;
    app.addEvent('PosToNode', this.PosToNode, this);
    app.addEvent('OnLoadMesh', this.RefreshLater, this);

    this.currentIndex = undefined;
    this.currentNode = undefined;

    this.inspector = new UIContainer(new Rect(400, 100, 300, 100));
    this.inspector.allowGrowth = true;
    this.inspector.title.setText('Inspector');
    this.inspector.pressedColor = this.inspector.color;
    this.inspEnableBtn = new Button(new Rect(0, 0, 300, 25));
    this.inspEnableBtn.onClick = () => {
      if (this.currentNode !== undefined) {
        this.currentNode.renderable.enabled = !this.currentNode.renderable.enabled;
      }
    };
    this.inspector.addWidget(this.inspEnableBtn);
    this.inspRenderingModelBtn = new Button(new Rect(0, 0, 300, 50));
    this.inspRenderingModelBtn.onClick = () => {
      if (this.currentIndex !== undefined && this.currentNode !== undefined) {
        const i = this.currentIndex;
        const nodeStatus = this.nodeStatus[i];
        if (!nodeStatus.renderingStatus) {
          nodeStatus.renderingStatus = this.currentNode.renderable.material.blendType;
        }
        nodeStatus.renderingStatus += 1;
        if (nodeStatus.renderingStatus === 4) {
          nodeStatus.renderingStatus = 0;
        }
        this.currentNode.renderable.material.setBlendType(nodeStatus.renderingStatus);
        let typeStr;
        switch (nodeStatus.renderingStatus) {
          case 1:
            typeStr = 'Alpha';
            break;
          case 2:
            typeStr = 'Add';
            break;
          default:
            typeStr = 'No Blend';
            break;
        }
        this.inspRenderingModelBtn.setText(typeStr);
      }
    };
    this.inspector.addWidget(this.inspRenderingModelBtn);
    this.inspChangeShaderBtn = new Button(new Rect(0, 0, 300, 50));
    this.inspChangeShaderBtn.onClick = () => {
      if (this.currentIndex !== undefined && this.currentNode !== undefined) {
        const i = this.currentIndex;
        const nodeStatus = this.nodeStatus[i];
        if (!nodeStatus.shaderStatus) {
          nodeStatus.shaderStatus = 0;
        }
        nodeStatus.shaderStatus += 1;
        if (nodeStatus.shaderStatus === 6) {
          nodeStatus.shaderStatus = 0;
        }
        let shaderStr;
        let shaderName;
        switch (nodeStatus.shaderStatus) {
          case 1:
            shaderStr = 'Shader: PBR Specular';
            shaderName = 'pbr_specular';
            break;
          case 2:
            shaderStr = 'Shader: UV Test';
            shaderName = 'uv';
            break;
          case 3:
            shaderStr = 'Shader: Normal Test';
            shaderName = 'normal';
            break;
          case 4:
            shaderStr = 'Shader: Tangent Test';
            shaderName = 'tangent';
            break;
          case 5:
            shaderStr = 'Shader: Texture Test';
            shaderName = 'texture';
            break;
          default:
            shaderStr = 'Shader: PBR';
            shaderName = 'pbr';
            break;
        }
        this.currentNode.renderable.material.updateShader(shaderName);
        this.inspChangeShaderBtn.setText(shaderStr);
      }
    };
    this.inspector.addWidget(this.inspChangeShaderBtn);

    this.inspChangeColor = new Button(new Rect(0, 0, 300, 50));
    this.inspChangeColor.onClick = () => {
      if (this.currentIndex !== undefined && this.currentNode !== undefined) {
        const nodeStatus = this.nodeStatus[this.currentIndex];
        if (!nodeStatus.colorStatus) {
          nodeStatus.colorStatus = 0;
          nodeStatus.cachedTexture = this.currentNode.renderable.material.getData('uAlbedo');
        }
        nodeStatus.colorStatus += 1;
        if (nodeStatus.colorStatus === 9) {
          nodeStatus.colorStatus = 0;
        }
        let tex;
        let colorName;
        switch (nodeStatus.colorStatus) {
          case 1:
            tex = getDefaultTextures().white;
            colorName = 'white';
            break;
          case 2:
            tex = getDefaultTextures().red;
            colorName = 'red';
            break;
          case 3:
            tex = getDefaultTextures().green;
            colorName = 'green';
            break;
          case 4:
            tex = getDefaultTextures().blue;
            colorName = 'blue';
            break;
          case 5:
            tex = getDefaultTextures().yellow;
            colorName = 'yellow';
            break;
          case 6:
            tex = getDefaultTextures().cyan;
            colorName = 'cyan';
            break;
          case 7:
            tex = getDefaultTextures().magenta;
            colorName = 'magenta';
            break;
          case 8:
            tex = getDefaultTextures().black;
            colorName = 'black';
            break;
          default:
            tex = nodeStatus.cachedTexture;
            colorName = 'texture';
            break;
        }
        this.currentNode.renderable.material.setUniformData('uAlbedo', tex);
        this.inspChangeColor.setText(colorName);
      }
    };
    this.inspector.addWidget(this.inspChangeColor);

    this.inspChangeCulling = new Button(new Rect(0, 0, 300, 50));
    this.inspChangeCulling.onClick = () => {
      if (this.currentIndex !== undefined && this.currentNode !== undefined) {
        const i = this.currentIndex;
        const nodeStatus = this.nodeStatus[i];
        if (!nodeStatus.CullingStatus) {
          nodeStatus.CullingStatus = this.currentNode.renderable.material.cullingType;
        }
        nodeStatus.CullingStatus += 1;
        if (nodeStatus.CullingStatus === 4) {
          nodeStatus.CullingStatus = 0;
        }
        let cullingStr;
        switch (nodeStatus.CullingStatus) {
          case 1:
            cullingStr = 'CullingMode: Front';
            break;
          case 2:
            cullingStr = 'CullingMode: Front And Back';
            break;
          case 3:
            cullingStr = 'CullingMode: Off';
            break;
          default:
            cullingStr = 'CullingMode: Back';
            break;
        }
        this.currentNode.renderable.material.setCullingType(nodeStatus.CullingStatus);
        this.inspChangeCulling.setText(cullingStr);
      }
    };
    this.inspector.addWidget(this.inspChangeCulling);

    this.inspSkinBtn = new Button(new Rect(0, 0, 300, 50));
    this.inspSkinBtn.onClick = () => {
      if (this.currentIndex !== undefined && this.currentNode !== undefined) {
        const nodeStatus = this.nodeStatus[this.currentIndex];
        if (nodeStatus.SkinTag === undefined) nodeStatus.SkinTag = false;
        nodeStatus.SkinTag = !nodeStatus.SkinTag;
        if (nodeStatus.SkinTag) {
          this.currentNode.renderable.material.addOrUpdateDefine('SKIN', 1);
        } else {
          this.currentNode.renderable.material.removeDefine('SKIN');
        }
        this.inspSkinBtn.setText(`${nodeStatus.SkinTag ? 'Skin Mode: Enabled' : 'Skin Mode: Disabled'}`);
      }
    };
    this.inspector.addWidget(this.inspSkinBtn);

    this.inspScaleText = new GameNode(new TextField(TextFieldType.Float).setText('Scale : 1.0').setTextColor([0.6, 0.6, 0.9, 1.0]), 'inspector_scale_text').renderable;
    this.inspScaleText.rect = new Rect(0, 0, 300, 50);
    this.inspScaleText.onInputTextTargets.push(this);
    this.inspector.addWidget(this.inspScaleText);

    this.inspVertexInfoText = new GameNode(new Text().setText('').setTextColor([0.6, 0.6, 0.9, 1.0]), 'inspector_vertex_text').renderable;
    this.inspVertexInfoText.rect = new Rect(0, 0, 300, 50);
    // this.inspRenderingModelBtn = new Button(new Rect(0, 0, 300, 50));
    this.inspector.addWidget(this.inspVertexInfoText);
    this.inspector.anchorType = AnchorType.Right;

    this.Refresh();
    this.hierarchy.Refresh();
    this.hierarchy.enabled = false;
    this.inspector.enabled = false;
    this.inspectorEnable = false;
    input.eventListeners.push(this);
  }

  RefreshLater() {
    this.update = this.Refresh;
  }

  Refresh() {
    const nodes = GameNode.getDebugGroup();
    let i;
    const imax = nodes.length;
    if (this.itemButtons.length < nodes.length) {
      i = this.itemButtons.length;
      let b;
      for (; i < imax; i += 1) {
        b = new Button(new Rect(0, 0, 300, 50));
        // b.setColor([1.0, 1.0, 1.0, 0.7]);
        b.notify.push(this);
        b.enabled = this.hierarchy.enabled;
        this.hierarchy.addWidget(b);
        this.itemButtons.push(b);
        this.nodeStatus[i] = {};
      }
    }
    this.itemButtons.forEach((button, j) => {
      button.setText(`${nodes[j].name}`);
    });
    this.hierarchy.Refresh();
    this.RefreshInspector();
    this.update = undefined;
  }

  // update() {
  //   // console.log('here');
  //   this.Refresh();
  // }

  OnClick(button) {
    if (this.addNodeButton === button) {
      this.CreateNewNode();
      return;
    }
    let i;
    const imax = this.itemButtons.length;
    for (i = 0; i < imax; i += 1) {
      if (button === this.itemButtons[i]) {
        const nodes = GameNode.getDebugGroup();
        if (this.currentIndex === undefined || this.currentIndex !== i) {
          this.inspector.enabled = true;
        } else {
          this.inspector.enabled = !this.inspector.enabled;
        }
        this.inspectorEnable = this.inspector.enabled;
        this.currentIndex = i;
        this.currentNode = nodes[i];
        this.RefreshInspector();
        this.lastClicked = i;
        break;
      }
    }
  }

  RefreshInspector() {
    if (this.currentNode === undefined || this.currentIndex === undefined) return;
    this.inspector.title.setText(this.currentNode.name);
    this.inspEnableBtn.setText(`${this.currentNode.renderable.enabled ? 'enabled' : 'disabled'}`);
    const nodeStatus = this.nodeStatus[this.currentIndex];
    if (!nodeStatus.renderingStatus) {
      nodeStatus.renderingStatus = this.currentNode.renderable.material.blendType;
    }
    let typeStr = 'BlendMode : ';
    switch (nodeStatus.renderingStatus) {
      case 1:
        typeStr += 'Alpha';
        break;
      case 2:
        typeStr += 'Add';
        break;
      default:
        typeStr += 'No Blend';
        break;
    }
    this.inspRenderingModelBtn.setText(typeStr);
    let shaderStr;
    switch (nodeStatus.shaderStatus) {
      case 1:
        shaderStr = 'Shader: PBR Specular';
        break;
      case 2:
        shaderStr = 'Shader: UV Test';
        break;
      case 3:
        shaderStr = 'Shader: Normal Test';
        break;
      case 4:
        shaderStr = 'Shader: Tangent Test';
        break;
      case 5:
        shaderStr = 'Shader: Texture Test';
        break;
      default:
        shaderStr = 'Shader: PBR';
        break;
    }
    this.inspChangeShaderBtn.setText(shaderStr);
    let colorName;
    switch (nodeStatus.colorStatus) {
      case 1:
        colorName = 'white';
        break;
      case 2:
        colorName = 'red';
        break;
      case 3:
        colorName = 'green';
        break;
      case 4:
        colorName = 'blue';
        break;
      default:
        colorName = 'texture';
        break;
    }
    this.inspChangeColor.setText(colorName);
    if (this.currentNode.renderable.meshFaceStart !== undefined) {
      this.inspVertexInfoText.setText(`vertex : ${this.currentNode.renderable.meshVertexCount}, subface : ${this.currentNode.renderable.meshFaceCount}`);
    } else {
      this.inspVertexInfoText.setText(`vertex : ${this.currentNode.renderable.meshVertexCount}, face : ${this.currentNode.renderable.meshFaceCount}`);
    }

    let cullingStr;
    switch (nodeStatus.CullingStatus) {
      case 1:
        cullingStr = 'CullingMode: Front';
        break;
      case 2:
        cullingStr = 'CullingMode: Front And Back';
        break;
      case 3:
        cullingStr = 'CullingMode: Off';
        break;
      default:
        cullingStr = 'CullingMode: Back';
        break;
    }
    this.inspChangeCulling.setText(cullingStr);

    const scale = this.currentNode.transform.scale[0];
    this.inspScaleText.setText('Scale :');
    this.inspScaleText.setFieldText(`${scale.toFixed(2)}`);
    this.inspSkinBtn.setText(`${nodeStatus.SkinTag ? 'Skin Mode: Enabled' : 'Skin Mode: Disabled'}`);
    this.inspector.Refresh();
  }

  PosToNode(...args) {
    const ls = args;
    this.RefreshLater();
    if (this.hierarchy.enabled) {
      // console.log(this);
      // console.log(ls[1]);
      if (this.addNodeButton.rect.contains(ls[1])) {
        ls[0][0] = this.CreateNewNode();
        return;
      }
      let i;
      const imax = this.itemButtons.length;
      for (i = 0; i < imax; i += 1) {
        if (this.itemButtons[i].rect.contains(ls[1])) {
          const nodes = GameNode.getDebugGroup();
          ls[0][0] = nodes[i];
          // console.log(ls[0][0].name);
          return;
        }
      }
    }
    ls[0][0] = undefined;
  }

  OnKeyDown(e) {
    if (e.key === 'h') {
      this.hierarchy.enabled = !this.hierarchy.enabled;
      this.inspector.enabled = this.inspectorEnable && this.hierarchy.enabled;
    }
  }

  OnInputText(textField, text) {
    if (textField === this.inspScaleText) {
      if (this.currentIndex !== undefined && this.currentNode !== undefined) {
        const scaleNext = parseFloat(text);
        const scale = this.currentNode.transform.scale[0];
        if (scaleNext !== scale) {
          this.currentNode.transform.scale = [scaleNext, scaleNext, scaleNext];
        }
      }
    }
  }
}

addOnStart(Hierarchy);

export { Hierarchy };
