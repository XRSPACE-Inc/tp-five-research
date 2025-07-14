import { getBRDF } from '../engine/brdf.js';
import { addOnStart } from './app.js';


export default class BRDFSetter {
  constructor(app) {
    this.brdfTexture = getBRDF();
    app.addEvent('OnLoadMesh', this.SetBRDF, this);
  }

  SetBRDF(nodes) {
    if (!nodes) return;
    nodes.forEach(node => {
      node.renderable.material.setUniformData('uBRDF', this.brdfTexture);
    });
  }
}

addOnStart(BRDFSetter);

export { BRDFSetter };
