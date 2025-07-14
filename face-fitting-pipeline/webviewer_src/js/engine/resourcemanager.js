import { ReadFile } from './readFile.js';
import JSZip from '../lib/jszip.min.js';

export default class ResourceManager {
  constructor() {
    this.list = [];
    this.text_ids = [];
    this.texts = [];
    this.zipDatas = {};
    this.onZipLoaded = undefined;
    this.zipCount = 0;
  }

  add(...args) {
    let c; let
      max;
    const st = this.list.length;
    for (c = st, max = st + args.length; c < max; c += 1) {
      this.list.push(args[c - st]);
      this.text_ids.push(c);
    }
  }

  addData(name, data) {
    this.texts[this.list.length] = data;
    this.list.push(name);
  }

  load() {
    let c;
    for (c = this.list.length - 1; c >= 0; c -= 1) {
      ReadFile.read(this.list[c], this.onLoad, { _this: this, index: this.text_ids[c] });
    }
  }

  loadZip(zipFile, token) {
    ReadFile.readBinary(zipFile, this.onLoadZipFile, this, token);
  }

  onLoadZipFile(data, self) {
    const zip = new JSZip();
    zip.loadAsync(data).then((contents) => {
      const keys = Object.keys(contents.files);
      keys.forEach((filename) => {
        const f = contents.file(filename);
        if (f) {
          self.zipCount += 1;
          const filenamelist = filename.split('.');
          const ext = filenamelist[filenamelist.length - 1].toLowerCase();
          let asyncText = 'string';
          if (ext === 'png' || ext === 'jpg') {
            asyncText = 'uint8array';
          } else if (ext === 'fbx') {
            asyncText = 'arraybuffer';
          }
          f.async(asyncText).then((content) => {
            self.addFileFromZip(content, { filename });
          });
        }
      });
    });
    return this;
  }

  addFileFromZip(content, passData) {
    if (passData === undefined) {
      // eslint-disable-next-line no-console
      console.log('onLoadZipFile error token is undefined');
      return;
    }
    const { filename } = passData;
    this.zipDatas[filename] = content;
    this.zipCount -= 1;
    if (this.zipCount === 0 && this.onZipLoaded !== undefined) {
      this.onZipLoaded();
    }
  }

  onLoad(t, token) {
    if (token === undefined) return null;
    let text = t;
    const { _this } = token;
    const { index } = token;
    text = text.replace(/(?:\\[rn]|[\r\n]+)+/g, '\n');
    _this.texts[index] = text;
    _this.text_ids.splice(0, 1);
    return this;
  }

  printAll() {
    let c;
    for (c = this.list.length - 1; c >= 0; c -= 1) {
      // eslint-disable-next-line no-console
      console.log(this.list[c]);
    }
  }

  get loadFinished() {
    return this.text_ids.length === 0;
  }

  clear() {
    this.text_ids.length = 0;
    this.list.length = 0;
  }

  getFile(file) {
    let c;
    for (c = this.list.length - 1; c >= 0; c -= 1) {
      if (this.list[c] === file) return { name: file, data: this.texts[c] };
    }
    return null;
  }

  get(file) {
    return this.zipDatas[file];
  }

  loadZipTexture(path, targetTexture) {
    const data = this.get(path);
    if (!data) {
      // eslint-disable-next-line no-console
      console.error(`cannot find texture in zip ${path}`);
      return false;
    }
    const src = `data:image;base64,${ReadFile.arrayBufferToBase64(data)}`;
    const image = new Image();
    image.src = src;
    image.onload = () => {
      targetTexture.loadFromImage(image);
    };
    return true;
  }
}

const resourceManager = new ResourceManager();

export { resourceManager };
