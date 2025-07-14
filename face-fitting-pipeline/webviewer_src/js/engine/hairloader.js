import { BinaryLoader } from './binaryloader.js';

export default function HairLoader(contents) {
  const binloader = new BinaryLoader(contents);

  const aabb = [
    [binloader.readFloat(), binloader.readFloat(), binloader.readFloat()],
    [binloader.readFloat(), binloader.readFloat(), binloader.readFloat()],
  ];

  const bufferSize = binloader.readInt();
  const vertexBuffer = new Float32Array(new Uint8Array(binloader.readBytes(bufferSize)).buffer);
  // eslint-disable-next-line radix
  const vertexSize = parseInt(bufferSize / 4);

  return {
    vertexBuffer,
    vertexSize,
    aabb,
  };
}

export { HairLoader };
