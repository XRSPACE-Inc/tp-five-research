/* eslint-disable prefer-destructuring */
/* eslint-disable max-len */
/* eslint-disable no-bitwise */
import { Renderable, DrawingMode } from './renderable.js';
import { Material, BlendType } from './material.js';
import { vec3, vec2 } from '../lib/gl-matrix/index.js';
import { ObjLoader, generateNormals } from './objLoader.js';
import { HairLoader } from './hairloader.js';
import { gl } from './gl.js';
import { TextureParamType, Texture2D } from './texture.js';
import { isMobile } from './utils.js';
import { Shader } from './shader.js';
// import { DebugDraw } from './debugDraw.js';

function addLine(face1, face2, wireframeMap, wireframeIndices) {
  let add = false;
  if (face1 in wireframeMap) {
    if (!wireframeMap[face1].includes(face2)) {
      wireframeMap[face1].push(face2);
      add = true;
    }
  } else if (face2 in wireframeMap) {
    if (!wireframeMap[face2].includes(face1)) {
      wireframeMap[face2].push(face1);
      add = true;
    }
  } else {
    // eslint-disable-next-line no-param-reassign
    wireframeMap[face1] = [face2];
    add = true;
  }
  if (add) {
    wireframeIndices.push(face1);
    wireframeIndices.push(face2);
  }
}

function addWireLines(face1, face2, face3, wireframeMap, wireframeIndices) {
  addLine(face1, face2, wireframeMap, wireframeIndices);
  addLine(face1, face3, wireframeMap, wireframeIndices);
  addLine(face2, face3, wireframeMap, wireframeIndices);
}

function findBestTextureSize(vertexSize) {
  let texSize = 512;
  while (texSize * texSize < vertexSize) {
    texSize *= 2;
  }
  return texSize;
}

function createBlendShapeTexture(rawTargets, fbxTree, vertexSize, texSize) {
  function getScaleVal(val) {
    if (val > 100000.0) {
      return 0.0000001;
    } if (val > 10000.0) {
      return 0.000001;
    } if (val > 1000.0) {
      return 0.00001;
    } if (val > 100.0) {
      return 0.0001;
    } if (val > 10.0) {
      return 0.001;
    } if (val > 1.0) {
      return 0.01;
    } if (val > 0.1) {
      return 0.1;
    } if (val > 0.01) {
      return 1.0;
    } if (val > 0.001) {
      return 10.0;
    } if (val > 0.0001) {
      return 100.0;
    } if (val > 0.00001) {
      return 1000.0;
    }
    return 10000.0;
  }

  function getScale(val) {
    if (val > 100000.0) {
      return 255; // 1.0
    } if (val > 10000.0) {
      return 232; // 0.91
    } if (val > 1000.0) {
      return 209; // 0.82
    } if (val > 100.0) {
      return 186; // 0.73
    } if (val > 10.0) {
      return 163; // 0.64
    } if (val > 1.0) {
      return 140; // 0.55
    } if (val > 0.1) {
      return 117; // 0.46
    } if (val > 0.01) {
      return 94; // 0.37
    } if (val > 0.001) {
      return 71; // 0.28
    } if (val > 0.0001) {
      return 48; // 0.18
    } if (val > 0.00001) {
      return 25; // 0.1
    }
    return 2; // 0.18
  }
  let texBytes;
  if (isMobile) {
    texBytes = new Uint8Array(texSize * texSize * 4);
    for (let i = texBytes.length - 1; i >= 0; i -= 1) {
      texBytes[i] = 0;
    }
  } else {
    texBytes = new Float32Array(texSize * texSize * 4);
    for (let i = texBytes.length - 1; i >= 0; i -= 1) {
      texBytes[i] = 0.5;
    }
  }

  let data; let indices; let verts; let st; let veclen; let scaleVal; let
    scaleCode;
  rawTargets.forEach((target, i) => {
    data = fbxTree.get(target.geoID);
    indices = data.indexes;
    verts = data.vertices;
    // console.log(verts);
    st = i * vertexSize * 4;
    if (isMobile) {
      indices.forEach((index, j) => {
        veclen = vec3.length(
          vec3.set(vec3.create(), verts[j * 3], verts[j * 3 + 1], verts[j * 3 + 2]),
        );
        scaleCode = getScale(veclen);
        scaleVal = getScaleVal(veclen);
        texBytes[st + index * 4] = (verts[j * 3] * scaleVal + 0.5) * 255.0;
        texBytes[st + index * 4 + 1] = (verts[j * 3 + 1] * scaleVal + 0.5) * 255.0;
        texBytes[st + index * 4 + 2] = (verts[j * 3 + 2] * scaleVal + 0.5) * 255.0;
        texBytes[st + index * 4 + 3] = scaleCode;
      });
    } else {
      indices.forEach((index, j) => {
        texBytes[st + index * 4] = verts[j * 3] + 0.5;
        texBytes[st + index * 4 + 1] = verts[j * 3 + 1] + 0.5;
        texBytes[st + index * 4 + 2] = verts[j * 3 + 2] + 0.5;
        texBytes[st + index * 4 + 3] = 0.35;
      });
    }
  });
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
  const texOptions = { Wrap: TextureParamType.Clamp_To_Edge, Filter: TextureParamType.Nearest };
  return new Texture2D(
    gl.RGBA, isMobile ? gl.UNSIGNED_BYTE : gl.FLOAT, texSize, texSize, texBytes, texOptions,
  );
  // return new Texture2D(gl.RGBA, gl.UNSIGNED_BYTE, texSize, texSize, texBytes, texOptions);
}

function orthogonalize(nor, tan) {
  let res = vec3.copy(vec3.create(), tan);
  const s = vec3.scale(vec3.create(), nor, vec3.dot(nor, res) / vec3.dot(nor, nor));

  res = vec3.sub(res, res, s);
  return vec3.normalize(res, res);
}

function fromTriangleNormalized(tris, tar, OrigSize) {
  const res = Array(OrigSize).fill(0.0);
  // console.log(`tri.len ${tris.length}, res.len ${res.length}`);
  tris.forEach((tri, i) => {
    res[tri * 3 + 0] += tar[i * 3 + 0];
    res[tri * 3 + 1] += tar[i * 3 + 1];
    res[tri * 3 + 2] += tar[i * 3 + 2];
  });
  const vertCount = OrigSize / 3;
  let v = vec3.create();
  for (let i = 0; i < vertCount; i += 1) {
    v = vec3.set(v, res[i * 3 + 0], res[i * 3 + 1], res[i * 3 + 2]);
    v = vec3.normalize(v, v);
    res[i * 3 + 0] = v[0];
    res[i * 3 + 1] = v[1];
    res[i * 3 + 2] = v[2];
  }
  return res;
}

function toTriangleVec(tris, tar) {
  const res = Array(tris.length * 3).fill(0.0);
  tris.forEach((i1, i) => {
    res[i * 3 + 0] = tar[i1 * 3 + 0];
    res[i * 3 + 1] = tar[i1 * 3 + 1];
    res[i * 3 + 2] = tar[i1 * 3 + 2];
  });
  return res;
}

function calcTangents(vertices, uvs, normals, tris, vertCount) {
  if (!uvs) return [[], []];
  let tangents = Array(vertices.length).fill(0.0);
  let bitangents = Array(vertices.length).fill(0.0);
  let i = 0;
  let i1;
  let i2;
  let i3;
  let v1 = vec3.create();
  let v2 = vec3.create();
  let v3 = vec3.create();
  let w1 = vec2.create();
  let w2 = vec2.create();
  let w3 = vec2.create();
  let r;
  let tangent;
  let bitangent;
  const imax = vertices.length / 9;
  let x1;
  let x2;
  let y1;
  let y2;
  let z1;
  let z2;
  let s1;
  let s2;
  let t1;
  let t2;
  let sdir;
  let tdir;
  let n;
  for (i = 0; i < imax; i += 1) {
    i1 = i * 6 + 0;
    i2 = i * 6 + 2;
    i3 = i * 6 + 4;

    w1 = vec2.set(w1, uvs[i1], uvs[i1 + 1]);
    w2 = vec2.set(w2, uvs[i2], uvs[i2 + 1]);
    w3 = vec2.set(w3, uvs[i3], uvs[i3 + 1]);

    i1 = i * 9 + 0;
    i2 = i * 9 + 3;
    i3 = i * 9 + 6;

    v1 = vec3.set(v1, vertices[i1], vertices[i1 + 1], vertices[i1 + 2]);
    v2 = vec3.set(v2, vertices[i2], vertices[i2 + 1], vertices[i2 + 2]);
    v3 = vec3.set(v3, vertices[i3], vertices[i3 + 1], vertices[i3 + 2]);

    x1 = v2[0] - v1[0];
    x2 = v3[0] - v1[0];
    y1 = v2[1] - v1[1];
    y2 = v3[1] - v1[1];
    z1 = v2[2] - v1[2];
    z2 = v3[2] - v1[2];

    s1 = w2[0] - w1[0];
    s2 = w3[0] - w1[0];
    t1 = w2[1] - w1[1];
    t2 = w3[1] - w1[1];

    r = 1.0 / (s1 * t2 - s2 * t1 + 1e-10);
    sdir = vec3.set(vec3.create(), (t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
    tdir = vec3.set(vec3.create(), (s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

    tangents[i1 + 0] += sdir[0];
    tangents[i1 + 1] += sdir[1];
    tangents[i1 + 2] += sdir[2];
    tangents[i2 + 0] += sdir[0];
    tangents[i2 + 1] += sdir[1];
    tangents[i2 + 2] += sdir[2];
    tangents[i3 + 0] += sdir[0];
    tangents[i3 + 1] += sdir[1];
    tangents[i3 + 2] += sdir[2];

    bitangents[i1 + 0] += tdir[0];
    bitangents[i1 + 1] += tdir[1];
    bitangents[i1 + 2] += tdir[2];
    bitangents[i2 + 0] += tdir[0];
    bitangents[i2 + 1] += tdir[1];
    bitangents[i2 + 2] += tdir[2];
    bitangents[i3 + 0] += tdir[0];
    bitangents[i3 + 1] += tdir[1];
    bitangents[i3 + 2] += tdir[2];
  }

  const vnorms = fromTriangleNormalized(tris, normals, vertCount);
  tangents = fromTriangleNormalized(tris, tangents, vertCount);
  bitangents = fromTriangleNormalized(tris, bitangents, vertCount);

  for (i = 0; i < vertCount; i += 3) {
    n = vec3.set(vec3.create(), vnorms[i], vnorms[i + 1], vnorms[i + 2]);
    tangent = vec3.set(vec3.create(), tangents[i], tangents[i + 1], tangents[i + 2]);
    bitangent = vec3.set(vec3.create(), bitangents[i], bitangents[i + 1], bitangents[i + 2]);

    tangent = orthogonalize(n, tangent);

    if (vec3.dot(vec3.cross(vec3.create(), n, tangent), bitangent) < 0.0) {
      tangent = vec3.scale(tangent, tangent, -1.0);
    }

    tangents[i + 0] = tangent[0];
    tangents[i + 1] = tangent[1];
    tangents[i + 2] = tangent[2];

    bitangents[i + 0] = bitangent[0];
    bitangents[i + 1] = bitangent[1];
    bitangents[i + 2] = bitangent[2];
  }

  return [tangents, bitangents];
}

let blendshapeCount = 0;
export default class Mesh extends Renderable {
  constructor(shaderName, fileData) {
    super(new Material(shaderName));

    // if (fileData === undefined) return undefined;

    this.size = 0;
    this.wireframe_size = 0;
    this.blendShapeSize = 0;
    this.blendShapeNames = undefined;
    this.blendWeights = undefined;
    this.blendVertexTexture = undefined;
    this.shaderName = shaderName;

    if (fileData !== undefined) this.LoadObj(fileData);
    return this;
  }

  LoadMesh(mesh, loadTag = undefined) {
    let loadName = loadTag;
    this.size = 0;
    this.wireframe_size = 0;
    this.aabb = mesh.aabb;
    const { vertices } = mesh;
    // const { normals } = mesh;
    // const { uvs } = mesh;
    const { vertexColors } = mesh;
    const vIndice = mesh.faces;
    const tIndice = mesh.uv_faces;
    const nIndice = mesh.normal_faces;
    const { meshDict } = mesh;
    const { faceVertices } = mesh;
    const { faceUVs } = mesh;
    const { faceNormals } = mesh;
    const { faceTangents } = mesh;
    const { faceBitangents } = mesh;
    // const { faceHairTangents } = mesh;
    if (loadName && (!meshDict || Object.keys(meshDict).length === 0 || !(loadName in meshDict))) {
      loadName = undefined;
    }

    this.meshVertexCount = vertices.length / 3;
    this.meshFaceStart = undefined;
    this.meshFaceCount = vIndice.length / 3;

    const faces = [];
    let vID = -1;
    const wireframeMap = {};
    const wireframeIndices = [];
    const wireframeFaces = [];

    let i;
    let ist;
    let imax;
    if (!loadName) {
      ist = 0;
      imax = vIndice.length;
    } else {
      // console.log(meshDict[loadName]);
      ({ ist, imax } = meshDict[loadName]);
      // ist = meshDict[loadName][0];
      // imax = meshDict[loadName][1];
    }
    // console.log(`ist : ${ist}, imax : ${imax}`);
    for (i = ist; i < imax; i += 3) {
      addWireLines(vIndice[i], vIndice[i + 1], vIndice[i + 2], wireframeMap, wireframeIndices);
    }

    // let v;
    // let v1;
    // let n;
    // const lines = [];
    // let lineSize = 0;
    // for (i = 0; i < normals.length; i += 3) {
    //   // console.log(`${i}, ${i + 1}, ${i + 2}`);
    //   v = vec3.set(vec3.create(), vertices[i], vertices[i + 1], vertices[i + 2]);
    //   n = vec3.set(vec3.create(), normals[i], normals[i + 1], normals[i + 2]);
    //   // n = vec3.scale(n, n, 2.0);
    //   // n = vec3.subtract(n, n, [1.0, 1.0, 1.0]);
    //   // n = vec3.set(n, n[0], n[1], n[2]);
    //   n = vec3.scale(n, n, 0.01);
    //   v1 = vec3.add(vec3.create(), v, n);
    //   lines.push(v[0]);
    //   lines.push(v[1]);
    //   lines.push(v[2]);
    //   lines.push(v1[0]);
    //   lines.push(v1[1]);
    //   lines.push(v1[2]);
    //   lineSize += 2;
    //   // DebugDraw.get()?.addLine(v, v1, [0.0, 0.0, 1.0, 1.0]);
    // }
    // console.log(lines);
    // DebugDraw.get()?.addLines(lines, lineSize, [0.0, 0.0, 1.0, 1.0]);
    let tangents = [];
    let bitangents = [];
    if (!faceTangents || !faceBitangents) {
      [tangents, bitangents] = calcTangents(faceVertices, faceUVs, faceNormals, vIndice, vertices.length);
    }

    const containsTangents = tangents.length > 0 && bitangents.length > 0;
    const containsColors = vertexColors.length > 0;
    const containsNormal = nIndice.length > 0;
    const containsUV = tIndice.length > 0;

    for (i = ist; i < imax; i += 1) {
      vID = vIndice[i];
      this.setFace(faces,
        vID,
        vertices[vID * 3],
        vertices[(vID * 3) + 1],
        vertices[(vID * 3) + 2],
        faceUVs[i * 2],
        faceUVs[i * 2 + 1],
        faceNormals[i * 3],
        faceNormals[i * 3 + 1],
        faceNormals[i * 3 + 2],
        tangents[vID * 3],
        tangents[vID * 3 + 1],
        tangents[vID * 3 + 2],
        bitangents[vID * 3],
        bitangents[vID * 3 + 1],
        bitangents[vID * 3 + 2],
        // faceHairTangents[i * 3],
        // faceHairTangents[i * 3 + 1],
        // faceHairTangents[i * 3 + 2],
        vertexColors[vID * 3],
        vertexColors[vID * 3 + 1],
        vertexColors[vID * 3 + 2]);
    }

    let offset = 0;
    let stride = 3;
    if (faces.length > 0) {
      // vert_id
      stride += 1;
      if (containsColors) stride += 3;
    } else if (containsColors) stride += 3;
    if (containsNormal) stride += 3;
    if (containsUV) stride += 2;
    if (containsTangents) stride += 6;
    // if (faceHairTangents.length > 0) stride += 3;

    let size = 3;
    this.attributeDatas = [];
    this.attributeDatas.push({
      name: 'position', size, offset, stride,
    });
    offset += size;

    if (faces.length > 0) {
      size = 1;
      this.attributeDatas.push({
        name: 'vert_id', size, offset, stride,
      });
      offset += size;

      if (containsUV) {
        size = 2;
        this.attributeDatas.push({
          name: 'texcoord', size, offset, stride,
        });
        offset += size;
      }

      if (containsNormal) {
        size = 3;
        this.attributeDatas.push({
          name: 'normal', size, offset, stride,
        });
        offset += size;
      }

      if (containsTangents) {
        size = 3;
        this.attributeDatas.push({
          name: 'tangent', size, offset, stride,
        });
        offset += size;
        this.attributeDatas.push({
          name: 'bitangent', size, offset, stride,
        });
        offset += size;
      }

      // if (faceHairTangents.length > 0) {
      //   size = 3;
      //   this.attributeDatas.push({
      //     name: 'hair_tangent', size, offset, stride,
      //   });
      //   offset += size;
      // }

      if (containsColors) {
        size = 3;
        this.attributeDatas.push({
          name: 'color', size, offset, stride,
        });
        offset += size;
      }
      this.drawingMode = DrawingMode.Normal;
    } else {
      size = 3;
      this.attributeDatas.push({
        name: 'color', size, offset, stride,
      });
      offset += size;

      this.setPointsFace(faces, vertices, vertexColors);
      this.drawingMode = DrawingMode.Points;
    }

    this.faces = faces;

    this.uploadList(this.faces);

    imax = wireframeIndices.length;
    for (i = 0; i < imax; i += 1) {
      vID = wireframeIndices[i];
      this.setWireframeFace(
        wireframeFaces,
        vertices[vID * 3],
        vertices[(vID * 3) + 1],
        vertices[(vID * 3) + 2],
        vID,
      );
    }
    this.wireframeAttributeDatas = [];
    this.wireframeAttributeDatas.push({
      name: 'position', size: 3, offset: 0, stride: 4,
    });
    this.wireframeAttributeDatas.push({
      name: 'vert_id', size: 1, offset: 3, stride: 4,
    });
    this.uploadListWireframe(wireframeFaces);
    // if (add_vertex_color) {
    //     // this.material.setUniformData('uAlbedo', getDefaultTextures()["black"]);
    // }
    // console.log(this.material);
    if (this.shaderName !== this.material.shaderName) {
      this.material = new Material(this.shaderName);
      this.material.setBlendType(BlendType.NoBlend);
    }
    return this;
  }

  LoadObj(fileData) {
    return this.LoadMesh(ObjLoader(fileData));
  }

  LoadHair(fileData) {
    const mesh = HairLoader(fileData);
    const { vertexBuffer, vertexSize, aabb } = mesh;
    // console.log(mesh);
    // console.log(vertexBuffer);
    // console.log(vertexSize);
    // console.log(aabb);

    this.size = vertexSize;
    this.wireframe_size = 0;
    this.aabb = aabb;
    // console.log(aabb);
    this.attributeDatas = [];
    this.attributeDatas.push({
      name: 'position', size: 3, offset: 0, stride: 6,
    });
    this.attributeDatas.push({
      name: 'tangent', size: 3, offset: 3, stride: 6,
    });

    this.drawingMode = DrawingMode.Lines;
    this.faces = vertexBuffer;
    this.uploadList(this.faces);
    this.material = new Material('hair');
    this.material.setUniformData('color', [0.6, 0.35, 0.1, 1.0]);
    this.material.setBlendType(BlendType.Alpha);
    return this;
  }

  setFace(faces, id, x, y, z, tx, ty, nx, ny, nz, tax, tay, taz, bix, biy, biz, cx, cy, cz) {
    faces.push(parseFloat(x));
    faces.push(parseFloat(y));
    faces.push(parseFloat(z));
    if (id !== undefined) { // need to check for undefined since 0 === !id
      faces.push(id);
    }

    if (tx !== undefined && ty !== undefined) {
      faces.push(parseFloat(tx));
      faces.push(parseFloat(ty));
    }

    if (nx !== undefined && ny !== undefined && nz !== undefined) {
      faces.push(parseFloat(nx));
      faces.push(parseFloat(ny));
      faces.push(parseFloat(nz));
    }

    if (tax !== undefined && tay !== undefined && taz !== undefined && bix !== undefined && biy !== undefined && biz !== undefined) {
      faces.push(parseFloat(tax));
      faces.push(parseFloat(tay));
      faces.push(parseFloat(taz));

      faces.push(parseFloat(bix));
      faces.push(parseFloat(biy));
      faces.push(parseFloat(biz));
    }

    // if (hairtx !== undefined && hairty !== undefined && hairtz !== undefined) {
    //   faces.push(parseFloat(hairtx));
    //   faces.push(parseFloat(hairty));
    //   faces.push(parseFloat(hairtz));
    // }

    if (cx !== undefined && cy !== undefined && cz !== undefined) {
      const checker = parseFloat(cx);
      if (checker < 1.0) {
        faces.push(checker);
        faces.push(parseFloat(cy));
        faces.push(parseFloat(cz));
      } else {
        faces.push(checker / 255.0);
        faces.push(parseFloat(cy) / 255.0);
        faces.push(parseFloat(cz) / 255.0);
      }
    }

    this.size += 1;
    return faces;
  }

  setWireframeFace(faces, x, y, z, id) {
    faces.push(parseFloat(x));
    faces.push(parseFloat(y));
    faces.push(parseFloat(z));
    faces.push(id);

    this.wireframe_size += 1;
    return faces;
  }

  setPointsFace(faces, vertices, vertexColors) {
    if (!vertices || vertices.length === 0 || vertices.length % 3 !== 0) return;

    if (vertexColors.length === vertices.length) {
      const total = vertices.length / 3;
      const checker = vertexColors[0];
      const isuint8 = checker > 1.0;
      for (let i = 0; i < total; i += 1) {
        faces.push(parseFloat(vertices[i * 3 + 0]));
        faces.push(parseFloat(vertices[i * 3 + 1]));
        faces.push(parseFloat(vertices[i * 3 + 2]));
        if (isuint8) {
          faces.push(parseFloat(vertexColors[i * 3 + 0]) / 255.0);
          faces.push(parseFloat(vertexColors[i * 3 + 1]) / 255.0);
          faces.push(parseFloat(vertexColors[i * 3 + 2]) / 255.0);
        } else {
          faces.push(parseFloat(vertexColors[i * 3 + 0]));
          faces.push(parseFloat(vertexColors[i * 3 + 1]));
          faces.push(parseFloat(vertexColors[i * 3 + 2]));
        }
        this.size += 1;
      }
    } else {
      const total = vertices.length / 3;
      for (let i = 0; i < total; i += 1) {
        faces.push(parseFloat(vertices[i * 3 + 0]));
        faces.push(parseFloat(vertices[i * 3 + 1]));
        faces.push(parseFloat(vertices[i * 3 + 2]));
        faces.push(1.0);
        faces.push(1.0);
        faces.push(1.0);
        this.size += 1;
      }
    }
  }

  setLineFace(faces, vertices, vertexColors) {
    if (!vertices || vertices.length === 0 || vertices.length % 3 !== 0) return;

    const total = vertices.length / 3;
    if (vertexColors.length === vertices.length) {
      const checker = vertexColors[0];
      const isuint8 = checker > 1.0;
      if (isuint8) {
        for (let i = 0; i < total; i += 1) {
          faces.push(parseFloat(vertices[i * 3 + 0]));
          faces.push(parseFloat(vertices[i * 3 + 1]));
          faces.push(parseFloat(vertices[i * 3 + 2]));
          faces.push(parseFloat(vertexColors[i * 3 + 0]) / 255.0);
          faces.push(parseFloat(vertexColors[i * 3 + 1]) / 255.0);
          faces.push(parseFloat(vertexColors[i * 3 + 2]) / 255.0);
          this.size += 1;
        }
      } else {
        for (let i = 0; i < total; i += 1) {
          faces.push(parseFloat(vertices[i * 3 + 0]));
          faces.push(parseFloat(vertices[i * 3 + 1]));
          faces.push(parseFloat(vertices[i * 3 + 2]));
          faces.push(parseFloat(vertexColors[i * 3 + 0]));
          faces.push(parseFloat(vertexColors[i * 3 + 1]));
          faces.push(parseFloat(vertexColors[i * 3 + 2]));
          this.size += 1;
        }
      }
    } else {
      faces.push(parseFloat(vertices[0]));
      faces.push(parseFloat(vertices[1]));
      faces.push(parseFloat(vertices[2]));
      faces.push(1.0);
      faces.push(1.0);
      faces.push(1.0);
      this.size += 1;
      for (let i = 0; i < total; i += 1) {
        faces.push(parseFloat(vertices[i * 3 + 0]));
        faces.push(parseFloat(vertices[i * 3 + 1]));
        faces.push(parseFloat(vertices[i * 3 + 2]));
        faces.push(1.0);
        faces.push(1.0);
        faces.push(1.0);
        this.size += 1;
        if (i < total - 1) {
          faces.push(parseFloat(vertices[i * 3 + 0]));
          faces.push(parseFloat(vertices[i * 3 + 1]));
          faces.push(parseFloat(vertices[i * 3 + 2]));
          faces.push(1.0);
          faces.push(1.0);
          faces.push(1.0);
          this.size += 1;
        }
      }
    }
  }

  SetBlendWeight(index, value) {
    this.blendWeights[index] = value;
    this.material.setUniformData('blendWeights', this.blendWeights);
    this.wireframeMaterial.setUniformData('blendWeights', this.blendWeights);
  }

  BlendShapeNameToIndex(name) {
    // console.log(this);
    for (let i = this.blendShapeNames.length - 1; i >= 0; i -= 1) {
      if (this.blendShapeNames[i] === name) return i;
    }
    return -1;
  }

  LoadFBX(mesh, fbxTree, material) {
    this.attributeDatas = [];
    this.wireframeAttributeDatas = [];
    this.size = 0;
    this.wireframe_size = 0;
    this.aabb = [
      mesh.buffers.min, mesh.buffers.max,
    ];
    let normal = mesh.buffers.normal;
    // material = start, count, materialIndex
    if (normal.length === 0) normal = generateNormals(mesh.buffers.meshVertices, mesh.buffers.vertexIndex);
    this.meshVertexCount = mesh.buffers.meshVertices.length / 3;
    if (material) {
      this.meshFaceStart = material.start;
      this.meshFaceCount = material.count;
    } else {
      this.meshFaceStart = undefined;
      this.meshFaceCount = mesh.buffers.vertexIndex.length / 3;
    }

    let [faceTangents, faceBitangents] = calcTangents(mesh.buffers.vertex, mesh.buffers.uvs[0], normal, mesh.buffers.vertexIndex, mesh.buffers.meshVertices.length);

    faceTangents = toTriangleVec(mesh.buffers.vertexIndex, faceTangents);
    faceBitangents = toTriangleVec(mesh.buffers.vertexIndex, faceBitangents);

    const containsTangents = faceTangents.length > 0 && faceBitangents.length > 0;
    const containsColors = mesh.buffers.colors.length > 0;
    const containsNormal = normal.length > 0;
    const containsUV = mesh.buffers.uvs.length > 0 && mesh.buffers.uvs[0].length > 0;

    const faces = [];
    let i = material ? material.start : 0;
    let imax = material ? i + material.count : mesh.buffers.vertexIndex.length;
    for (; i < imax; i += 1) {
      this.setFace(
        faces,
        mesh.buffers.vertexIndex[i],
        mesh.buffers.vertex[i * 3], mesh.buffers.vertex[i * 3 + 1], mesh.buffers.vertex[i * 3 + 2],
        mesh.buffers.uvs[0][i * 2], mesh.buffers.uvs[0][i * 2 + 1],
        normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2],
        faceTangents[i * 3], faceTangents[i * 3 + 1], faceTangents[i * 3 + 2],
        faceBitangents[i * 3], faceBitangents[i * 3 + 1], faceBitangents[i * 3 + 2],
        // undefined, undefined, undefined,
        mesh.buffers.colors[i * 3], mesh.buffers.colors[i * 3 + 1], mesh.buffers.colors[i * 3 + 2],
      );
    }

    let offset = 0;
    // position
    let stride = 3;
    // vert_id
    stride += 1;
    if (containsColors) stride += 3;
    if (containsNormal) stride += 3;
    if (containsUV) stride += 2;
    if (containsTangents) stride += 6;

    let size = 3;
    this.attributeDatas.push({
      name: 'position', size, offset, stride,
    });
    offset += size;
    size = 1;
    this.attributeDatas.push({
      name: 'vert_id', size, offset, stride,
    });
    offset += size;

    if (containsUV) {
      size = 2;
      this.attributeDatas.push({
        name: 'texcoord', size, offset, stride,
      });
      offset += size;
    }
    if (containsNormal) {
      size = 3;
      this.attributeDatas.push({
        name: 'normal', size, offset, stride,
      });
      offset += size;
    }
    if (containsTangents) {
      size = 3;
      this.attributeDatas.push({
        name: 'tangent', size, offset, stride,
      });
      offset += size;
      this.attributeDatas.push({
        name: 'bitangent', size, offset, stride,
      });
      offset += size;
    }
    if (containsColors) {
      size = 3;
      this.attributeDatas.push({
        name: 'color', size, offset, stride,
      });
      offset += size;
    }
    this.faces = faces;

    this.uploadList(this.faces);
    // console.log(imax);
    // console.log(this.faces);
    // console.log(this.attributeDatas);

    const wireframeMap = {};
    const wireframeIndices = [];
    const wireframeFaces = [];
    imax /= 3;
    // console.log(mesh.buffers.vertexIndex);
    for (i = 0; i < imax; i += 1) {
      addWireLines(
        mesh.buffers.vertexIndex[i * 3],
        mesh.buffers.vertexIndex[i * 3 + 1],
        mesh.buffers.vertexIndex[i * 3 + 2],
        wireframeMap,
        wireframeIndices,
      );
    }

    imax = wireframeIndices.length;
    let vID;
    for (i = 0, vID = 0; i < imax; i += 1) {
      vID = wireframeIndices[i];
      this.setWireframeFace(
        wireframeFaces,
        mesh.buffers.meshVertices[vID * 3],
        mesh.buffers.meshVertices[(vID * 3) + 1],
        mesh.buffers.meshVertices[(vID * 3) + 2],
        vID,
      );
    }
    this.wireframeAttributeDatas.push({
      name: 'position', size: 3, offset: 0, stride: 4,
    });
    this.wireframeAttributeDatas.push({
      name: 'vert_id', size: 1, offset: 3, stride: 4,
    });
    this.uploadListWireframe(wireframeFaces);

    if (mesh.morphTargets.length > 0) {
      // console.log(mesh.morphTargets[0].rawTargets);
      // console.log(fbxTree);
      const { rawTargets } = mesh.morphTargets[0];
      this.blendShapeSize = rawTargets.length;
      this.blendWeights = new Float32Array(this.blendShapeSize);
      this.verticeSize = mesh.buffers.meshVertices.length;
      this.blendShapeNames = [];
      for (i = 0; i < this.blendShapeSize; i += 1) {
        this.blendShapeNames.push(rawTargets[i].name);
      }
      const texSize = findBestTextureSize(this.verticeSize * this.blendShapeSize);
      this.blendVertexTexture = createBlendShapeTexture(
        rawTargets, fbxTree, this.verticeSize, texSize,
      );
      this.createBlendShapeMaterial(texSize, true);
    }
    // console.log(this.material);
    return this;
  }

  createBlendShapeMaterial(texSize, doScale) {
    const shader = Shader.FindShaderSource(this.shaderName);
    const replaceDefines = `#define BLENDSHAPE_SIZE ${this.blendShapeSize}\n`
      + `#define BLENDSHAPE_TEX_SIZE ${texSize}.0\n`
      + 'attribute float vert_id;\n';
    let replaceUniforms = 'uniform float verticeSize;\n'
      + 'uniform sampler2D blendVertexTexture;\n'
      + 'uniform float blendWeights[BLENDSHAPE_SIZE];\n'
      + 'vec4 getBlendOffset(sampler2D blendTexture, int blendShapeID) {\n'
      + 'float posInTexture = vert_id + (float(blendShapeID) * verticeSize);\n'
      + 'float xTexel, yTexel;\n'
      + 'xTexel = floor(mod(posInTexture, BLENDSHAPE_TEX_SIZE));\n'
      + 'yTexel = floor(posInTexture / BLENDSHAPE_TEX_SIZE);\n'
      + 'xTexel += 0.5;\n'
      + 'yTexel += 0.5;\n'
      + 'xTexel /= BLENDSHAPE_TEX_SIZE;\n'
      + 'yTexel /= BLENDSHAPE_TEX_SIZE;\n'
      + 'return texture2D(blendTexture, vec2(xTexel, yTexel));\n'
      + '}\n';
    if (doScale) {
      replaceUniforms += 'float getScale(float scaleCode) {\n'
        + 'if (scaleCode > 0.97) return 10000000.0;\n'
        + 'else if (scaleCode > 0.9) return 1000000.0;\n'
        + 'else if (scaleCode > 0.8) return 100000.0;\n'
        + 'else if (scaleCode > 0.7) return 10000.0;\n'
        + 'else if (scaleCode > 0.6) return 1000.0;\n'
        + 'else if (scaleCode > 0.5) return 100.0;\n'
        + 'else if (scaleCode > 0.4) return 10.0;\n'
        + 'else if (scaleCode > 0.3) return 1.0;\n'
        + 'else if (scaleCode > 0.2) return 0.1;\n'
        + 'else if (scaleCode > 0.1) return 0.01;\n'
        + 'else if (scaleCode > 0.01) return 0.001;\n'
        + 'return 0.0;\n'
        + '}\n';
    }
    const replaceGLPositionSrc = 'gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);';
    let replaceGLPosition = 'for (int i = 0; i < BLENDSHAPE_SIZE; i++) {\n'
      + 'vec4 texelVertex = getBlendOffset(blendVertexTexture, i);\n';
    if (doScale) {
      replaceGLPosition += 'if (texelVertex.w > 0.0) finalPos += (texelVertex.xyz - 0.5) * getScale(texelVertex.w) * blendWeights[i];\n';
    } else {
      replaceGLPosition += 'if (texelVertex.w > 0.0) finalPos += (texelVertex.xyz - 0.5) * 10.0 * blendWeights[i];\n';
    }
    replaceGLPosition += '}\n'
      + 'gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);\n';
    blendshapeCount += 1;
    let shaderName = `${this.shaderName}${blendshapeCount}`;
    Shader.ShaderMap()[shaderName] = {
      vs: shader.vs.slice()
        .replace('#pragma DEFINES', replaceDefines)
        .replace('#pragma UNIFORMS', replaceUniforms)
        .replace(replaceGLPositionSrc, replaceGLPosition),
      fs: shader.fs.slice(),
      name: shaderName,
    };
    this.material = new Material(shaderName);
    this.material.setUniformData('verticeSize', this.verticeSize);
    this.material.setUniformData('blendVertexTexture', this.blendVertexTexture);
    this.material.setUniformData('blendWeights', this.blendWeights);
    // console.log(this.material);
    // console.log("loaded");
    shaderName = `wireframe${blendshapeCount}`;
    const debugShader = Shader.FindShaderSource('debug');
    Shader.ShaderMap()[shaderName] = {
      vs: debugShader.vs.slice()
        .replace('#pragma DEFINES', replaceDefines)
        .replace('#pragma UNIFORMS', replaceUniforms)
        .replace(replaceGLPositionSrc, replaceGLPosition),
      fs: debugShader.fs.slice(),
    };
    // console.log(shadersMap);
    // console.log(this.verticeSize);
    this.wireframeMaterial = new Material(shaderName);
    this.wireframeMaterial.setUniformData('verticeSize', this.verticeSize);
    this.wireframeMaterial.setUniformData('blendVertexTexture', this.blendVertexTexture);
    this.wireframeMaterial.setUniformData('blendWeights', this.blendWeights);
    this.wireframeMaterial.setUniformData('color', [0.0, 1.0, 0.0, 1.0]);
  }
}

export { Mesh };
