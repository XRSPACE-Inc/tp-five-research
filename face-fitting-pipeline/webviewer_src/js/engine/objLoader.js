/* eslint-disable max-len */
/* eslint-disable prefer-destructuring */
import { clamp } from './utils.js';
import { vec3 } from '../lib/gl-matrix/index.js';

function calcNormal(v0, v1, v2) {
  const u = vec3.subtract(vec3.create(), v1, v0);
  const v = vec3.subtract(vec3.create(), v2, v0);

  return vec3.normalize(vec3.create(), vec3.set(vec3.create(),
    (u[1] * v[2]) - (u[2] * v[1]),
    (u[2] * v[0]) - (u[0] * v[2]),
    (u[0] * v[1]) - (u[1] * v[0])));
}

function generateNormals(vertices, faces) {
  const normals = Array(vertices.length).fill(0.0);
  const sameIndices = Array(vertices.length).fill(0.0);
  let i;
  let i0;
  let i1;
  let i2;
  let v0;
  let v1;
  let v2;
  let nor;
  const imax = faces.length;
  for (i = 0; i < imax; i += 3) {
    i0 = faces[i];
    i1 = faces[i + 1];
    i2 = faces[i + 2];
    v0 = vec3.set(vec3.create(), vertices[i0 * 3], vertices[(i0 * 3) + 1], vertices[(i0 * 3) + 2]);
    v1 = vec3.set(vec3.create(), vertices[i1 * 3], vertices[(i1 * 3) + 1], vertices[(i1 * 3) + 2]);
    v2 = vec3.set(vec3.create(), vertices[i2 * 3], vertices[(i2 * 3) + 1], vertices[(i2 * 3) + 2]);

    nor = calcNormal(v0, v1, v2);
    sameIndices[(i0 * 3) + 0] += 1.0;
    sameIndices[(i0 * 3) + 1] += 1.0;
    sameIndices[(i0 * 3) + 2] += 1.0;
    sameIndices[(i1 * 3) + 0] += 1.0;
    sameIndices[(i1 * 3) + 1] += 1.0;
    sameIndices[(i1 * 3) + 2] += 1.0;
    sameIndices[(i2 * 3) + 0] += 1.0;
    sameIndices[(i2 * 3) + 1] += 1.0;
    sameIndices[(i2 * 3) + 2] += 1.0;

    normals[(i0 * 3) + 0] += nor[0];
    normals[(i0 * 3) + 1] += nor[1];
    normals[(i0 * 3) + 2] += nor[2];
    normals[(i1 * 3) + 0] += nor[0];
    normals[(i1 * 3) + 1] += nor[1];
    normals[(i1 * 3) + 2] += nor[2];
    normals[(i2 * 3) + 0] += nor[0];
    normals[(i2 * 3) + 1] += nor[1];
    normals[(i2 * 3) + 2] += nor[2];
  }
  for (i = normals.length - 1; i >= 0; i -= 1) {
    normals[i] /= sameIndices[i];
  }
  return normals;
}

export default function ObjLoader(fileData) {
  if (fileData === undefined) return undefined;
  const lines = fileData.split('\n');
  if (lines === undefined || lines.length === 0) return undefined;

  let minX = Number.MAX_VALUE;
  let minY = Number.MAX_VALUE;
  let minZ = Number.MAX_VALUE;
  let maxX = Number.MIN_VALUE;
  let maxY = Number.MIN_VALUE;
  let maxZ = Number.MIN_VALUE;

  let c;
  let max;
  let line;
  let i;
  let valSize;
  let vals;
  const vertices = [];
  const vertexColors = [];
  let normals = [];
  let uvs = [];
  const tangents = [];
  const bitangents = [];
  const vIndices = [];
  let tIndices = [];
  let nIndices = [];
  let vertOnFaceCount = -1;
  let slashCount = 1;
  let addTexcoord = false;
  let addNormal = false;
  const meshDict = {};
  let lastMeshName;

  let splitted = [];
  function setIndices(val, sCount) {
    splitted = val.split(/\//);
    vIndices.push(splitted[0] - 1);
    if (sCount > 1 && addTexcoord) tIndices.push(splitted[1] - 1);
    if (sCount > 1 && addNormal) {
      if (sCount === 3) {
        nIndices.push(splitted[2] - 1);
      } else {
        nIndices.push(splitted[1] - 1);
      }
    }
  }

  for (c = 0, max = lines.length; c < max; c += 1) {
    line = lines[c];
    if (line[0] === 'v') {
      if (line[1] === 'n') {
        vals = line.split(' ');
        vals[1] = parseFloat(vals[1]);
        vals[2] = parseFloat(vals[2]);
        vals[3] = parseFloat(vals[3]);
        normals.push(vals[1]);
        normals.push(vals[2]);
        normals.push(vals[3]);
        addNormal = true;
      } else if (line[1] === 't') {
        vals = line.split(' ');
        vals[1] = parseFloat(vals[1]);
        vals[2] = parseFloat(vals[2]);
        uvs.push(vals[1]);
        uvs.push(vals[2]);
        addTexcoord = true;
      } else {
        vals = line.split(' ');
        vals[1] = parseFloat(vals[1]);
        vals[2] = parseFloat(vals[2]);
        vals[3] = parseFloat(vals[3]);
        // console.log(vals);
        vertices.push(vals[1]);
        vertices.push(vals[2]);
        vertices.push(vals[3]);

        if (vals[1] < minX) minX = vals[1];
        if (vals[2] < minY) minY = vals[2];
        if (vals[3] < minZ) minZ = vals[3];
        if (vals[1] > maxX) maxX = vals[1];
        if (vals[2] > maxY) maxY = vals[2];
        if (vals[3] > maxZ) maxZ = vals[3];

        if (vals.length > 4) {
          vertexColors.push(clamp(parseFloat(vals[4]), 0.0, 1.0));
          vertexColors.push(clamp(parseFloat(vals[5]), 0.0, 1.0));
          vertexColors.push(clamp(parseFloat(vals[6]), 0.0, 1.0));
          // add_vertex_color = true;
        }
      }
    } else if (line[0] === 'f') {
      if (vertOnFaceCount === -1) {
        //                                       - 'f'
        vertOnFaceCount = line.split(' ').length - 1;
        slashCount = 1 + (line.split('/').length - 1) / vertOnFaceCount;
      }
      vals = line.split(/ /);

      valSize = vals.length;
      for (i = 3; i < valSize; i += 1) {
        setIndices(vals[1], slashCount);
        setIndices(vals[i - 1], slashCount);
        setIndices(vals[i], slashCount);
      }
    } else if (line[0] === 'u') {
      if (lastMeshName) {
        meshDict[lastMeshName].imax = vIndices.length;
      }
      lastMeshName = line.split(/ /)[1];
      meshDict[lastMeshName] = { ist: vIndices.length, imax: -1 };
      // console.log(meshDict[lastMeshName]);
    } else if (line[0] === 't' && line[1] === 'a') {
      vals = line.split(' ');
      vals[1] = parseFloat(vals[1]);
      vals[2] = parseFloat(vals[2]);
      vals[3] = parseFloat(vals[3]);
      tangents.push(vals[1]);
      tangents.push(vals[2]);
      tangents.push(vals[3]);
    } else if (line[0] === 'b' && line[1] === 'i') {
      vals = line.split(' ');
      vals[1] = parseFloat(vals[1]);
      vals[2] = parseFloat(vals[2]);
      vals[3] = parseFloat(vals[3]);
      bitangents.push(vals[1]);
      bitangents.push(vals[2]);
      bitangents.push(vals[3]);
    }
  }

  if (lastMeshName) {
    meshDict[lastMeshName].imax = vIndices.length;
  }

  if (tIndices.length === 0) {
    tIndices = vIndices;
  }

  if (nIndices.length === 0) {
    nIndices = vIndices;
  }

  // console.log(vertices.length / 3);
  // console.log(uvs.length / 2);
  if (uvs.length === 0) {
    uvs = Array((vertices.length * 2) / 3).fill(0.5);
  }

  // normals = [];
  if (normals.length === 0) {
    // console.log('generate normals');
    normals = generateNormals(vertices, vIndices);
    nIndices = vIndices;
  }

  // console.log(`vind len: ${vIndices.length}, tind len: ${tIndices.length}, nind len: ${nIndices.length}`);

  const faceVertices = Array(vIndices.length * 3).fill(0.0);
  const faceUVs = Array(vIndices.length * 2).fill(0.0);
  const faceNormals = Array(vIndices.length * 3).fill(0.0);
  const doTangent = tangents.length > 0 && bitangents.length > 0;
  let faceTangents;
  let faceBitangents;
  if (doTangent) {
    faceTangents = Array(vIndices.length * 3).fill(0.0);
    faceBitangents = Array(vIndices.length * 3).fill(0.0);
  }
  const imax = vIndices.length;
  let vID;
  for (i = 0; i < imax; i += 1) {
    vID = vIndices[i];
    faceVertices[i * 3] = vertices[vID * 3];
    faceVertices[i * 3 + 1] = vertices[(vID * 3) + 1];
    faceVertices[i * 3 + 2] = vertices[(vID * 3) + 2];
    if (doTangent) {
      faceTangents[i * 3] = tangents[vID * 3];
      faceTangents[i * 3 + 1] = tangents[(vID * 3) + 1];
      faceTangents[i * 3 + 2] = tangents[(vID * 3) + 2];
      faceBitangents[i * 3] = bitangents[vID * 3];
      faceBitangents[i * 3 + 1] = bitangents[(vID * 3) + 1];
      faceBitangents[i * 3 + 2] = bitangents[(vID * 3) + 2];
    }
    vID = tIndices[i];
    faceUVs[i * 2] = uvs[vID * 2];
    faceUVs[i * 2 + 1] = uvs[(vID * 2) + 1];
    vID = nIndices[i];
    faceNormals[i * 3] = normals[vID * 3];
    faceNormals[i * 3 + 1] = normals[(vID * 3) + 1];
    faceNormals[i * 3 + 2] = normals[(vID * 3) + 2];
  }

  faceUVs.forEach((v, index, arr) => {
    // eslint-disable-next-line no-param-reassign
    if (v === undefined) arr[index] = 0;
  });

  const aabb = [];
  aabb[0] = [minX, minY, minZ];
  aabb[1] = [maxX, maxY, maxZ];
  // console.slog(meshDict);
  // console.slog(Object.keys(meshDict).length);
  // console.log(normals);
  // console.log(vertices);
  return {
    vertices,
    uvs,
    normals,
    faceVertices,
    faceUVs,
    faceNormals,
    faceTangents,
    faceBitangents,
    vertexColors,
    faces: vIndices,
    uv_faces: tIndices,
    normal_faces: nIndices,
    aabb,
    meshDict,
  };
}

export { ObjLoader, generateNormals };
