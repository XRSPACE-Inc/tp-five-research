import { Renderable } from './renderable.js';

export default class Quad extends Renderable {
  constructor(material) {
    super(material);

    this.attributeDatas = [
      {
        name: 'position',
        size: 3,
        offset: 0,
        stride: 8,
      },
      {
        name: 'normal',
        size: 3,
        offset: 3,
        stride: 8,
      },
      {
        name: 'texcoord',
        size: 2,
        offset: 6,
        stride: 8,
      },
    ];

    this.size = 6;
    const vertices = [
      // x y z nx ny nz tx ty
      1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
      -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
      1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
      -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ];

    this.uploadList(vertices);
  }
}

export { Quad };
