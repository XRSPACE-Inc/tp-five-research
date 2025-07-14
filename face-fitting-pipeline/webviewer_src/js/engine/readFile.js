let tempCanvas;

function getImagePixels(image) {
  if (!image || !image.width || !image.height) {
    return null;
  }
  if (tempCanvas === undefined) {
    tempCanvas = document.createElement('canvas');
  }
  tempCanvas.width = image.width;
  tempCanvas.height = image.height;
  const ctx = tempCanvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  const imageData = ctx.getImageData(0, 0, image.width, image.height);
  return new Uint8Array(imageData.data.buffer);
}

export default class ReadFile {
  static read(path, callback, passData = undefined, token = undefined) {
    const rawFile = new XMLHttpRequest();
    rawFile.passData = passData;
    rawFile.open('GET', path, true);
    rawFile.overrideMimeType('text/plain');
    if (token) rawFile.setRequestHeader('Authorization', `Bearer ${token}`);
    rawFile.onreadystatechange = function onreadystatechange() {
      if (rawFile.readyState === 4) {
        if (rawFile.status === 200 || rawFile.status === 0) {
          callback(rawFile.responseText, rawFile.passData);
          return;
        }
      }
      callback(null, rawFile.passData);
    };
    rawFile.send();
  }

  static arrayBufferToBase64(bytes) {
    let binary = '';
    const len = bytes.byteLength;
    for (let i = 0; i < len; i += 1) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  static readImage(url, callback, passData = undefined, token = undefined) {
    const rawFile = new XMLHttpRequest();
    rawFile.passData = passData;
    rawFile.open('GET', url, true);
    rawFile.responseType = 'arraybuffer';
    if (token) rawFile.setRequestHeader('Authorization', `Bearer ${token}`);
    rawFile.onreadystatechange = function onreadystatechange() {
      if (rawFile.readyState === 4) {
        if (rawFile.status === 200 || rawFile.status === 0) {
          const data = new Uint8Array(rawFile.response);
          const src = `data:image;base64,${ReadFile.arrayBufferToBase64(data)}`;
          const image = new Image();
          image.src = src;
          image.onload = function onload() {
            callback(this, passData);
          };
        }
      }
    };
    rawFile.send();
  }

  static readBinary(url, callback, passData = undefined, token = undefined) {
    const rawFile = new XMLHttpRequest();
    rawFile.passData = passData;
    rawFile.open('GET', url, true);
    rawFile.responseType = 'arraybuffer';
    if (token) rawFile.setRequestHeader('Authorization', `Bearer ${token}`);
    rawFile.onreadystatechange = function onreadystatechange() {
      if (rawFile.readyState === 4) {
        if (rawFile.status === 200 || rawFile.status === 0) {
          callback(rawFile.response, passData);
        }
      }
    };
    rawFile.send();
  }

  // static readTexture(path, callback, passData = undefined, token = undefined) {
  //     var rawFile = new XMLHttpRequest();
  //     rawFile.passData = passData;
  //     rawFile.open("GET", path, true);
  //     rawFile.responseType = "arraybuffer";
  //     if (token) rawFile.setRequestHeader('Authorization', 'Bearer ' + token);
  //     rawFile.onreadystatechange = function () {
  //         if (rawFile.readyState === 4) {
  //             if (rawFile.status === 200 || rawFile.status === 0) {
  //                 var data = new Uint8Array(rawFile.response);
  //                 var base64 = ReadFile.arrayBufferToBase64(data);
  //                 callback("data:image;base64," + base64, rawFile.passData);
  //             }
  //         }
  //     };
  //     rawFile.send();
  // }
}

export { getImagePixels, ReadFile };
