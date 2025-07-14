/* eslint-disable no-bitwise */
export default class BinaryLoader {
  constructor(contents) {
    this.bytes = new Uint8Array(contents);
    // console.log(contents);
    // console.log(this.bytes);
  }

  readInt() {
    const a = this.bytes;
    // eslint-disable-next-line no-mixed-operators
    const c = a[0] | a[1] << 8 | a[2] << 16 | a[3] << 24;
    this.bytes = a.subarray(4);
    return c;
  }

  readFloat() {
    let a = new Uint8Array(this.bytes);
    a = new Float32Array(a.buffer);
    this.bytes = this.bytes.subarray(4);
    return a[0];
  }

  readBytes(size) {
    if (this.bytes.length < size) throw new Error(`this.bytes.length:${this.bytes.length} < ${size}`);
    const data = this.bytes.subarray(0, size);
    this.bytes = this.bytes.subarray(size);
    return data;
  }
}
export { BinaryLoader };
