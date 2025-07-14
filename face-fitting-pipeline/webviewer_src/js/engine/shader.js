/* eslint-disable no-console */
import { gl } from './gl.js';
import { isMobile } from './utils.js';

let programUsing = null;

if (!String.prototype.replaceAll) {
  // eslint-disable-next-line no-extend-native
  String.prototype.replaceAll = function replaceAll(search, replace) {
    return this.split(search).join(replace);
  };
}

function removeShaderText(text) {
  return text.replace(' void ', '\nvoid ')
    .replace(' { ', '\n{\n')
    .replaceAll('; ', ';\n')
    .replaceAll('\r', '');
}

const shadersMap = {
  error: {
    vs: `#pragma DEFINES
attribute vec3 position;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;
#pragma UNIFORMS
void main(void)
{
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}`,
    fs: `void main(void)
{
  gl_FragColor = vec4(0.5, 0.0, 0.7, 1.0);
}`,
  },
  debug: {
    vs: `#pragma DEFINES
attribute vec3 position;

uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

uniform vec4 color;
#pragma UNIFORMS
varying lowp vec4 vColor;

void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vColor = color;
}`,
    fs: `varying lowp vec4 vColor;
void main(void)
{
  gl_FragColor = vColor;
}`,
  },
  default: {
    vs: `#pragma DEFINES
attribute vec3 position;
attribute vec3 color;
attribute vec3 normal;
attribute vec2 texcoord;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

#pragma UNIFORMS

varying highp vec2 vTexcoord;

varying highp vec3 N;
varying highp vec3 v;
void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vTexcoord = texcoord;
  v = (viewMatrix * modelMatrix * vec4(finalPos, 1.0)).xyz;
  N = normalize(normal);
}`,
    fs: `precision mediump float;
#pragma DEFINES
#pragma UNIFORMS

uniform sampler2D uAlbedo;

#ifdef LIGHT_COUNT
uniform vec3 lightDirections[LIGHT_COUNT];
uniform vec3 lightColors[LIGHT_COUNT];
#endif

varying highp vec2 vTexcoord;
varying highp vec3 N;
varying highp vec3 v;
void main(void)
{
  highp vec4 col = texture2D(uAlbedo, vTexcoord);
  highp vec4 Idiff = vec4(0.0, 0.0, 0.0, 1.0);
  highp vec4 Ispec = vec4(0.0, 0.0, 0.0, 1.0);
#ifdef LIGHT_COUNT
  for(int i = 0; i < LIGHT_COUNT; i++) {
    vec3 L = normalize(lightDirections[i] - v);
    vec3 E = normalize(-v);
    vec3 R = normalize(-reflect(L, N));
    Idiff.rgb += lightColors[i] * max(dot(N,L), 0.0);
    Ispec.rgb += lightColors[i] * pow(max(dot(R, E), 0.0), 30.0);
  }
#else
  highp vec3 L = normalize(vec3(0.0, 0.0, -1.0) - v);
  highp vec3 E = normalize(-v);
  highp vec3 R = normalize(-reflect(L, N));
  Idiff = vec4(vec3(0.8, 0.8, 0.78).rgb, 1.0) * max(dot(N,L), 0.0);
  Ispec = vec4(0.8, 0.8, 0.78, 1.0) * pow(max(dot(R, E), 0.0), 30.0);
#endif
  highp vec4 Iamb = vec4(0.3, 0.3, 0.3, 1.0);

  Idiff = clamp(Idiff, 0.0, 1.0);
  Ispec = clamp(Ispec, 0.0, 1.0);
  gl_FragColor = col * (Iamb + Idiff + Ispec);
}`,
  },
  default_orig: {
    vs: `#pragma DEFINES
attribute vec3 position;
attribute vec3 color;
attribute vec3 normal;
attribute vec2 texcoord;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

#pragma UNIFORMS

varying highp vec2 vTexcoord;

varying highp vec3 vLighting;
void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vTexcoord = texcoord;
  highp vec3 ambientLight = vec3(0.3, 0.3, 0.3);
  highp vec3 directionalLightColor = vec3(0.8, 0.8, 0.78);
  highp vec3 directionalVector = normalize(vec3(0.0, 0.0, 1.0));

  highp vec3 transformedNormal = normalize(normalMatrix * normal);

  highp float directional = max(dot(transformedNormal.xyz, directionalVector), 0.0);
  vLighting = ambientLight + (directionalLightColor * directional);
}`,
    fs: `#pragma DEFINES

uniform sampler2D uAlbedo;

#pragma UNIFORMS

varying highp vec2 vTexcoord;
varying highp vec3 vLighting;
void main(void)
{
  highp vec4 col = texture2D(uAlbedo, vTexcoord);
  gl_FragColor = vec4(col.rgb * vLighting, col.a);
}`,
  },
  uv: {
    vs: `#pragma DEFINES
attribute vec3 position;
attribute vec2 texcoord;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

#pragma UNIFORMS

varying highp vec2 vTexcoord;

void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vTexcoord = texcoord;
}`,
    fs: `#pragma DEFINES

#pragma UNIFORMS

varying highp vec2 vTexcoord;
void main(void)
{
  gl_FragColor = vec4(vTexcoord, 0.0, 1.0);
}`,
  },
  normal: {
    vs: `#pragma DEFINES
attribute vec3 position;
attribute vec3 normal;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

#pragma UNIFORMS

varying highp vec3 vNormal;

void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vNormal = normal;
}`,
    fs: `#pragma DEFINES

#pragma UNIFORMS

varying highp vec3 vNormal;
void main(void)
{
  gl_FragColor = vec4(vNormal, 1.0);
}`,
  },
  tangent: {
    vs: `#pragma DEFINES
attribute vec3 position;
attribute vec3 tangent;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

#pragma UNIFORMS

varying highp vec3 vTangent;

void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vTangent = tangent * 0.5 + 0.5;
}`,
    fs: `#pragma DEFINES

#pragma UNIFORMS

varying highp vec3 vTangent;
void main(void)
{
  gl_FragColor = vec4(vTangent, 1.0);
}`,
  },
  texture: {
    vs: `#pragma DEFINES
attribute vec3 position;
attribute vec2 texcoord;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

#pragma UNIFORMS

varying highp vec2 vTexcoord;

void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vTexcoord = texcoord;
}`,
    fs: `#pragma DEFINES

uniform sampler2D uAlbedo;
#pragma UNIFORMS

varying highp vec2 vTexcoord;
void main(void)
{
  gl_FragColor = texture2D(uAlbedo, vTexcoord);
}`,
  },
  sprite: {
    vs: `#pragma DEFINES
attribute vec2 position;
attribute vec2 texcoord;
uniform mat4 modelMatrix;
uniform vec2 halfView;
uniform vec4 widgetRect;
uniform vec4 color;
#pragma UNIFORMS
varying highp vec2 vTexcoord;
varying lowp vec4 vColor;
void main(void)
{
  vec2 pos = vec2(widgetRect.x, widgetRect.y);
  vec2 size = vec2(widgetRect.z, widgetRect.w);
  vec2 vertPos_homogeneous_space = ((position * size + size * 0.5) - halfView + pos) / halfView;
  gl_Position = modelMatrix * vec4(vertPos_homogeneous_space, 0, 1.0);
  vTexcoord = texcoord;
  vColor = color;
}`,
    fs: `#pragma DEFINES
precision highp float;
uniform sampler2D uAlbedo;
#pragma UNIFORMS
varying highp vec2 vTexcoord;
varying lowp vec4 vColor;
void main(void)
{
  gl_FragColor = texture2D(uAlbedo, vTexcoord) * vColor;
}`,
  },
  sprite_circle: {
    vs: `precision highp float;
#pragma DEFINES
attribute vec2 position;
attribute vec2 texcoord;
uniform mat4 modelMatrix;
uniform vec2 halfView;
uniform vec4 widgetRect;
uniform float radius;
uniform vec4 color;
#pragma UNIFORMS
varying highp vec2 vTexcoord;
varying lowp vec4 vColor;
varying highp vec2 radius_data;
void main(void)
{
  vec2 pos = vec2(widgetRect.x, widgetRect.y);
  vec2 size = vec2(widgetRect.z, widgetRect.w);
  vec2 vertPos_homogeneous_space = ((position * size + size * 0.5) - halfView + pos) / halfView;
  gl_Position = modelMatrix * vec4(vertPos_homogeneous_space, 0, 1.0);
  vTexcoord = texcoord;
  vColor = color;
}`,
    fs: `precision highp float;
uniform sampler2D uAlbedo;
varying highp vec2 vTexcoord;
varying lowp vec4 vColor;
varying highp vec2 radius_data;
void main(void)
{
  vec2 dir = vTexcoord * 2.0 - 1.0;

  if ((dir.x * dir.x + dir.y * dir.y) > 1.0) discard;
  gl_FragColor = texture2D(uAlbedo, vTexcoord) * vColor;

}`,
  },
  pointcloud: {
    vs: `#pragma DEFINES
attribute vec3 position;
attribute vec3 color;

uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;
#pragma UNIFORMS
varying highp vec3 vColor;

void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vColor = color;
  gl_PointSize = 3.0;
}`,
    fs: `varying highp vec3 vColor;

void main(void)
{
  gl_FragColor = vec4(vColor, 1.0);
}`,
  },
  brdf: {
    vs: `precision highp float;
#pragma DEFINES
attribute vec3 position;
attribute vec2 texcoord;
#pragma UNIFORMS

varying mediump vec2 vTexcoord;

void main(void) {
  vTexcoord = texcoord;
  gl_Position = vec4(position, 1.0); 
}`,
    fs: `#pragma DEFINES
precision mediump float;
varying vec2 vTexcoord;

#define M_PI2 6.283185307179586
#define SAMPLE_COUNT 1024

#pragma UNIFORMS

float VanDerCorpus(int n, int base) {
  float invBase = 1.0 / float(base);
  float denom = 1.0;
  float result = 0.0;

  for (int i = 0; i < 32; ++i) {
    if (n > 0) {
      denom = mod(float(n), 2.0);
      result += denom * invBase;
      invBase = invBase / 2.0;
      n = int(float(n) / 2.0);
    }
  }

  return result;
}

vec2 Hammersley(int i, int N) {
  return vec2(float(i)/float(N), VanDerCorpus(i, 2));
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
  float a = roughness * roughness;

  float phi = M_PI2 * Xi.x;
  float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  // from spherical coordinates to cartesian coordinates
  vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

  // from tangent-space vector to world-space sample vector
  vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, N));
  vec3 bitangent = cross(N, tangent);

  vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(sampleVec);
}

// Smith's Schlick-GGX with k = a * a / 2
float GeometrySchlickGGX(float NdotV, float roughness) {
  float a = roughness;
  float k = (a * a) / 2.0;

  float nom = NdotV;
  float denom = NdotV * (1.0 - k) + k;

  return nom / denom;
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
  float ggx1 = GeometrySchlickGGX(NdotV, roughness);
  float ggx2 = GeometrySchlickGGX(NdotL, roughness);

  return ggx1 * ggx2;
}

vec2 IntegrateBRDF(float NdotV, float roughness) {
  vec3 V = vec3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
  float A = 0.0;
  float B = 0.0;
  vec3 N = vec3(0.0, 0.0, 1.0);

  for (int i = 0; i < SAMPLE_COUNT; ++i) {
    // generates a sample vector that's biased towards the preferred alignment direction (importance sampling).
    vec2 Xi = Hammersley(i, SAMPLE_COUNT);
    vec3 H = ImportanceSampleGGX(Xi, N, roughness);
    vec3 L = normalize(2.0 * dot(V, H) * H - V);

    float NdotL = max(L.z, 0.0);
    float NdotH = max(H.z, 0.0);
    float VdotH = max(dot(V, H), 0.0);

    if (NdotL > 0.0) {
      float G = GeometrySmith(max(dot(N, V), 0.0), max(dot(N, L), 0.0), roughness);
      float G_Vis = (G * VdotH) / (NdotH * NdotV);
      float Fc = pow(1.0 - VdotH, 5.0);

      A += (1.0 - Fc) * G_Vis;
      B += Fc * G_Vis;
    }
  }

  A /= float(SAMPLE_COUNT);
  B /= float(SAMPLE_COUNT);

  return vec2(A, B);
}

void main(void) {
  vec2 integrateBRDF = IntegrateBRDF(vTexcoord.x, vTexcoord.y);
  gl_FragColor = vec4(integrateBRDF, 0.0, 1.0);
}`,
  },
  hair: {
    vs: `#pragma DEFINES
attribute vec3 position;
attribute vec3 tangent;

uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

#pragma UNIFORMS

varying highp vec3 vWorldPos;
varying highp vec3 vWorldTangent;

void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vWorldPos = (modelMatrix * vec4(finalPos, 1.0)).xyz;
  vWorldTangent = (modelMatrix * vec4(tangent, 0.0)).xyz;
}`,
    fs: `
precision highp float;
#pragma DEFINES

#define M_PI 3.14159265358979323846
#define M_E  2.71828182845904523536

varying highp vec3 vWorldPos;
varying highp vec3 vWorldTangent;
#pragma UNIFORMS

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 uCameraPosition;
uniform vec2 cameraResolution;

uniform vec4 color;

#ifdef LIGHT_COUNT
uniform vec3 lightDirections[LIGHT_COUNT];
uniform vec3 lightColors[LIGHT_COUNT];
uniform mat4 lightMatrices[LIGHT_COUNT];
#if LIGHT_COUNT > 0
uniform sampler2D shadowmap0;
#if LIGHT_COUNT > 1
uniform sampler2D shadowmap1;
#if LIGHT_COUNT > 2
uniform sampler2D shadowmap2;
#if LIGHT_COUNT > 3
uniform sampler2D shadowmap3;
#endif
#endif
#endif
#endif
#endif

vec3 kajiya_kay(vec3 diffuse, vec3 specular, float p, vec3 tangent, vec3 light, vec3 eye) {
  float cosTL = dot(tangent, light);
  float cosTE = dot(tangent, eye);

  float cosTL_squared = cosTL * cosTL;
  float cosTE_squared = cosTE * cosTE;

  float one_minus_cosTL_squared = 1.0 - cosTL_squared;
  float one_minus_cosTE_squared = 1.0 - cosTE_squared;

  float sinTL = sqrt(one_minus_cosTL_squared);
  float sinTE = sqrt(one_minus_cosTE_squared);

  vec3 diffuse_colors = diffuse * sinTL;
  float spec = cosTL * cosTE + sinTL * sinTE;
  vec3 specular_colors = specular * pow(max(spec, 0.0), p);
  return diffuse_colors + specular_colors * 0.5;
}

float approximate_deep_shadow(float shadow_depth, float light_depth, float strand_radius, float strand_alpha) {
    float strand_depth = max(0.0, light_depth - shadow_depth); // depth of the current strand inside geometry.
    float strand_count = strand_depth * strand_radius; // expected number of hair strands occluding the strand.

    if (strand_depth > 1e-10) strand_count += 1.0; // assume we have passed some strand if the depth is above some
    // floating point error threshold (e.g from the given shadow map) and add it to the occluding strand count.

    // We also take into account the transparency of the hair strand to determine how much light might scatter.
    return pow(1.0 - strand_alpha, strand_count); // this gives us "stronger" shadows with deeper hair strand.
}

vec4 tex2Dproj(sampler2D image, vec2 projected_position, vec2 displacement) {
    return texture2D(image, projected_position.st + displacement);
}

float approximate_deep_shadows(sampler2D shadow_map, vec4 light_space_strand, float kernel_width, float smoothing, float strand_radius, float strand_opacity) {
  float visibility = 0.0;
  vec2 shadow_map_size = vec2(1536.0, 1536.0);

  // float kernel_range = (kernel_width - 1.0) / 2.0;
  float sigma_stddev = (kernel_width / 2.0) / 2.4;
  float sigma_squared = sigma_stddev * sigma_stddev;

  float light_depth = (light_space_strand.z / light_space_strand.w) * 0.5 + 0.5;
  vec2 shadow_map_stride = shadow_map_size / smoothing; // stride.

  vec2 projected_position = (light_space_strand.xy / light_space_strand.w) * 0.5 + 0.5;

  float total_weight = 0.0;
  //             -kernel_range <= +kernel_range
  for (float y = -1.0; y <= 1.0; y += 1.0) {
    for (float x = -1.0; x <= 1.0; x += 1.0) {
        float exponent = -1.0 * (x * x + y * y) / 2.0 * sigma_squared; // Gaussian RBDF.
        float local_weight = 1.0 / (2.0 * M_PI * sigma_squared) * pow(M_E, exponent);

        float shadow_depth = tex2Dproj(shadow_map, projected_position, vec2(x, y) / shadow_map_stride).r;
        float shadow = approximate_deep_shadow(shadow_depth, light_depth, strand_radius, strand_opacity);

        visibility += shadow * local_weight;
        total_weight += local_weight;
    }
  }
  return visibility / total_weight;
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
  return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

void main(void)
{
  float hair_alpha = 0.1;
  // === gpaa ===
  vec2 screen_fragment = gl_FragCoord.xy;
  vec4 world_line = vec4(vWorldPos, 1.0);
  mat4 view_projection = projectionMatrix * viewMatrix;
  vec2 resolution = cameraResolution;
  float line_thickness = 2.0;

  vec4 clip_line = view_projection * world_line;
  vec3 ndc_line = (clip_line.xyz / clip_line.w);
  vec2 screen_line = ndc_line.xy;
  screen_line = (screen_line + 1.0) * (resolution / 2.0);
  float d = length(screen_line - screen_fragment);
  float converage = 1.0 - (d / (line_thickness / 2.0));
  converage *= hair_alpha;

  if (converage < 0.001) discard;

  converage *= 1.0 / 0.042;
  vec3 diffuse = pow(color.xyz, vec3(2.1632601288));

  // === occlusion & shadow ===
#ifdef LIGHT_COUNT
  float shadows[4];
  shadows[0] = shadows[1] = shadows[2] = shadows[3] = 1.0;
  #if LIGHT_COUNT > 0
  vec4 shadow_space_fragment = lightMatrices[0] * vec4(vWorldPos, 1.0);
  shadows[0] = approximate_deep_shadows(shadowmap0, shadow_space_fragment, 3.0, 4.0, 15000.0, hair_alpha);
  #if LIGHT_COUNT > 1
  shadow_space_fragment = lightMatrices[1] * vec4(vWorldPos, 1.0);
  shadows[1] = approximate_deep_shadows(shadowmap1, shadow_space_fragment, 3.0, 4.0, 15000.0, hair_alpha);
  #if LIGHT_COUNT > 2
  shadow_space_fragment = lightMatrices[2] * vec4(vWorldPos, 1.0);
  shadows[2] = approximate_deep_shadows(shadowmap2, shadow_space_fragment, 3.0, 4.0, 15000.0, hair_alpha);
  #if LIGHT_COUNT > 3
  shadow_space_fragment = lightMatrices[3] * vec4(vWorldPos, 1.0);
  shadows[3] = approximate_deep_shadows(shadowmap3, shadow_space_fragment, 3.0, 4.0, 15000.0, hair_alpha);
  #endif
  #endif
  #endif
  #endif
#endif
  lowp vec3 Lo = vec3(0.0);
  highp vec3 eye = normalize(vWorldPos - uCameraPosition);
#ifdef LIGHT_COUNT
  // === kajiya kay ===
  for (int i = 0; i < LIGHT_COUNT; i++) {
    //                vec3 diffuse, vec3 specular, float p, vec3 tangent, vec3 light, vec3 eye
    vec3 lighting = kajiya_kay(diffuse * lightColors[i], lightColors[i] * 0.25 * shadows[i], 80.0, vWorldTangent, lightDirections[i], eye);
    lighting *= shadows[i];
    Lo += lighting;
  }
#endif
  float NdotV = max(dot(vWorldTangent, eye), 0.0);
  vec3 F = fresnelSchlickRoughness(NdotV, diffuse, 1.0);
  vec3 kS = F;
  vec3 kD = 1.0 - kS;
  vec3 ambient = kD * diffuse * 0.2;

  vec3 result = ambient + Lo;
  result = result / (result + vec3(1.0));
  result = pow(result, vec3(0.46226525728));
  gl_FragColor = vec4(result, converage);
}`,
  },
  shadow: {
    vs: `precision highp float;
#pragma DEFINES
attribute vec3 position;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;
#pragma UNIFORMS
varying highp vec2 zw;
void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  zw = gl_Position.zw;
}`,
    fs: `precision highp float;
varying highp vec2 zw;
vec3 jH(float v){
    vec4 jI = vec4(1.0, 255.0, 65025.0, 16581375.0) * v;
    jI = fract(jI);
    jI.xyz -= jI.yzw * (1.0 / 255.0);
    return jI.xyz;
}
void main(void)
{
  // gl_FragColor.xyz = jH((zw.x / zw.y) * 0.5 + 0.5);
  gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
}`,
  },
  pbr_specular: {
    vs: `#pragma DEFINES
precision mediump float;
attribute vec3 position;
attribute vec3 color;
attribute vec3 normal;
attribute vec2 texcoord;
attribute vec3 tangent;
attribute vec3 bitangent;

uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

#ifdef LIGHT_COUNT
uniform mat4 lightMatrices[LIGHT_COUNT];
#endif

#pragma UNIFORMS

varying highp vec3 vWorldPos;
varying mediump vec3 vTangent;
varying mediump vec3 vBitangent;
varying mediump vec3 vNormal;
varying mediump vec2 vTexcoord;
varying highp vec3 vGlpos;

void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vTexcoord = texcoord;

#ifdef MH
  vNormal = -normalize(normal * 2.0 - 1.0).xzy;
#else
  vNormal = normal;
#endif

  vTangent = tangent;
  vBitangent = bitangent;
  vWorldPos = (modelMatrix * vec4(finalPos, 1.0)).xyz;

// #ifdef LIGHT_COUNT
// #if LIGHT_COUNT > 0
//   vNormal = (lightMatrices[0] * vec4(vNormal, 1.0)).xyz;
//   vTangent = (lightMatrices[0] * vec4(vTangent, 1.0)).xyz;
//   vBitangent = (lightMatrices[0] * vec4(vBitangent, 1.0)).xyz;
// #endif
// #endif

  vGlpos = gl_Position.xyz / gl_Position.w;
}`,
    fs: `#pragma DEFINES
#extension GL_OES_standard_derivatives : enable

#define M_PI 3.14159265358979323846
#define M_E  2.71828182845904523536

precision mediump float;
uniform sampler2D uAlbedo;
uniform sampler2D uNormal;
uniform sampler2D uSpecular;
uniform sampler2D uGloss;
uniform sampler2D uTranslucency;
uniform sampler2D uEnvir;

uniform vec3 uCameraPosition;

uniform vec4 uTransColor;
uniform vec4 uFresnelColor;
uniform vec3 uSubdermisColor;
uniform float uTransScatter;
uniform float uFresnelOcc;
uniform float uFresnelGlossMask;
uniform float uTransSky;
uniform float uTransIntegral;
uniform float uSkinTransDepth;
uniform float uSkinShadowBlur;
uniform float uNormalSmooth;
uniform vec4 uTexRangeTranslucency;

uniform vec4 uSH9Coeffs[9];
uniform float uHorizonOcclude;
uniform vec3 uFresnel;

#ifdef LIGHT_COUNT
uniform vec3 lightPositions[LIGHT_COUNT];
uniform vec3 lightDirections[LIGHT_COUNT];
uniform vec3 lightColors[LIGHT_COUNT];
#if LIGHT_COUNT > 0
uniform sampler2D shadowmap0;
#if LIGHT_COUNT > 1
uniform sampler2D shadowmap1;
#if LIGHT_COUNT > 2
uniform sampler2D shadowmap2;
#if LIGHT_COUNT > 3
uniform sampler2D shadowmap3;
#endif
#endif
#endif
#endif
uniform mat4 lightMatrices[LIGHT_COUNT];

struct ShadowData { // ev & eD
  float val[LIGHT_COUNT]; // eL
  float val2[LIGHT_COUNT]; // eE
};
#endif

#pragma UNIFORMS

varying highp vec3 vWorldPos; // dv
varying mediump vec3 vTangent; // dA
varying mediump vec3 vBitangent; // dB
varying mediump vec3 vNormal; // dC
varying mediump vec2 vTexcoord; // d
varying highp vec3 vGlpos; // wp

struct SkinData {
  vec3 subdermisColor; // hX
  vec3 hY,hZ,ic;
  vec3 extra; // he
  vec3 smoothedSH; // ec
  vec3 SHColor; // eh
  vec3 id;
  vec3 normal; // ie
  vec3 fresnelColor; // ih
  vec3 fresnelIntensity; // ii
  vec3 transColor; // ij
  float transScatter; // ik
  float il;
  float bias; // im
  float shadowBlur; // eC
};

vec3 SqrV3(vec3 v) { return v * v; }

vec3 UnpackNormal(vec3 normalmap) {
  vec3 tan = vTangent;
  vec3 bit = vBitangent;
  vec3 nor = gl_FrontFacing ? vNormal: -vNormal;
  normalmap = normalmap * 2.0 - vec3(1.0);
  vec3 res = tan * normalmap.x + bit * normalmap.y + nor * normalmap.z;
  return normalize(res);
}

vec3 UnpackNormalLOD(sampler2D normal,vec2 texcoord, float bias) {
    return UnpackNormal(texture2D(normal, texcoord, bias * 2.5).xyz);
}

vec3 SmootshadowPosH9(vec3 L0, vec3 L1, vec3 L2, vec3 fP, vec3 subdermisColor) { // fQ
    vec3 fS = mix(vec3(1.0), fP.yyy, subdermisColor);
    vec3 fT = mix(vec3(1.0), fP.zzz, subdermisColor);
    return (L0 + L1 * fS) + L2 * fT;
}

void SH9_SKIN(inout vec3 L0, inout vec3 L1, inout vec3 L2, vec3 normal) { // fK
    L0 = uSH9Coeffs[0].xyz;
    L1 = uSH9Coeffs[1].xyz * normal.y;
    L1 += uSH9Coeffs[2].xyz * normal.z;
    L1 += uSH9Coeffs[3].xyz * normal.x;
    vec3 swz=normal.yyz*normal.xzx;
    L2 = uSH9Coeffs[4].xyz * swz.x;
    L2 += uSH9Coeffs[5].xyz * swz.y;
    L2 += uSH9Coeffs[7].xyz * swz.z;
    vec3 sqr=normal*normal;
    L2 += uSH9Coeffs[6].xyz * (3.0 * sqr.z - 1.0);
    L2 += uSH9Coeffs[8].xyz * (sqr.x - sqr.y);
}

vec3 environmentmap(vec3 halfView, float gloss) { //em
    halfView /= dot(vec3(1.0, 1.0, 1.0), abs(halfView));
    vec2 fU = abs(halfView.zx) - vec2(1.0, 1.0);
    vec2 fV = vec2(halfView.x < 0.0 ? fU.x : -fU.x , halfView.z < 0.0 ? fU.y : -fU.y);
    vec2 texcoord = (halfView.y < 0.0) ? fV : halfView.xz;
    texcoord = vec2(0.5 * (254.0 / 256.0), 0.125 * 0.5 * (254.0 / 256.0)) * texcoord + vec2(0.5, 0.125 * 0.5);
    float fX = fract(gloss);
    texcoord.y += 0.125 * (gloss - fX);
    vec4 fZ = texture2D(uEnvir, texcoord);
    vec3 r = fZ.xyz * (7.0 * fZ.w);
    return r * r;
}

float SpecularIntensity(float intensity) {
  intensity = clamp(1.0 + uHorizonOcclude * intensity, 0.0, 1.0);
  return intensity * intensity;
}

vec3 SubdermisColor(float ldotsn, vec3 subdermisColor) {
  vec3 fn = vec3(ldotsn, ldotsn, ldotsn);
    fn = 0.8 * fn + vec3(0.2);
    vec3 fo = cos(fn * 3.14159);
    vec3 fu = cos(fn * 3.14159 * 0.5);
    fu *= fu;
    fu *= fu;
    fu *= fu;
    fn = fn + 0.05 * fo * fu * subdermisColor;
    fu *= fu;
    fu *= fu;
    fu *= fu;
    fn = fn + 0.1 * fo * fu * subdermisColor;
    fn = clamp(fn, 0.0, 1.0);
    fn *= fn * 1.2;
    return fn;
}

void SkinLighting(inout SkinData s, float iE,float iF, vec3 lightDir, vec3 nor, vec3 lightColor) {
  float ldotn = dot(lightDir, nor); // fk
  float ldotsn = dot(lightDir, s.normal); // fl

  float degree_ldotn = clamp((1.0 / 3.1415926) * ldotn, 0.0, 1.0); // eN
  float hi = iE * iE;
  hi *= hi;
  hi = clamp(6.0 * hi, 0.0, 1.0);

  vec3 iG = SubdermisColor(ldotsn, s.subdermisColor);

  float fi = s.fresnelIntensity.z;
  float extra = clamp(-ldotsn * fi + ldotsn + fi, 0.0, 1.0); // iH

  vec3 ic = vec3(extra * extra);
  highp vec3 iL = (0.995 * s.subdermisColor) + vec3(0.005, 0.005, 0.005);
  highp vec3 iM = vec3(1.0) - iL;
  iL = mix(iL, iM, iE);
  float iN = sqrt(iE);
  vec3 iO = 2.0 * vec3(1.0 - iN);
  iN = 1.0 - iN;
  iN = (1.0 - iN * iN);
  vec3 iI = clamp(pow(iL*iN,iO), 0.0, 1.0);
  highp float iP = 0.35 / (uSkinTransDepth + 0.001);
  highp float iQ = clamp(iF * iP, 0.0, 1.0);
  iQ = clamp(1.0 - iQ, 0.0, 1.0);
  iQ *= iQ;
  highp vec3 iR = vec3((-3.0 * iQ) + 3.15);
  highp vec3 iS = (0.9975 * s.transColor) + vec3(0.0025, 0.0025, 0.0025);
  highp float io = clamp(10.0 * dot(iS, iS), 0.0, 1.0);
  vec3 iK = pow(iS * iQ, iR) * io;

  vec3 hk = mix(vec3(1.0), iI, uFresnelOcc);
  vec3 extraColor = clamp(ldotsn * fi + ldotsn + fi, 0.0, 1.0) * hk; // * hj

  iG *= iI;
  degree_ldotn *= hi;
  ic *= iK;

  s.extra = extraColor * lightColor + s.extra;
  s.ic = ic * lightColor + s.ic;
  s.hZ = iG * lightColor + s.hZ;
  s.hY = degree_ldotn * lightColor + s.hY;
}

float approximate_deep_shadow(float shadow_depth, float light_depth, float strand_radius, float strand_alpha) {
  float depth = max(0.0, light_depth - shadow_depth); // depth of the current strand inside geometry.

  float strand_count = depth * strand_radius;

  if (depth > 1e-10) strand_count += 1.0;

  return pow(1.0 - strand_alpha, strand_count);
}

float tex2Dproj(sampler2D image, vec2 projected_position, vec2 displacement) {
  return texture2D(image, projected_position.st + displacement).r;
}

float approximate_deep_shadows(sampler2D shadow_map, vec4 light_space_vertex, float smoothing, float strand_radius, float strand_opacity, float blur) {
  float visibility = 0.0;
  vec2 shadow_map_size = vec2(1536.0, 1536.0);

  float kernel_width = 100.0 * blur;
  // float kernel_range = (kernel_width - 1.0) / 2.0;
  float sigma_stddev = (kernel_width * 0.5) / 2.4;
  float sigma_squared = sigma_stddev * sigma_stddev;

  float light_depth = (light_space_vertex.z / light_space_vertex.w) * 0.5 + 0.5;
  vec2 shadow_map_stride = smoothing / shadow_map_size; // stride.
  vec2 projected_position = (light_space_vertex.xy / light_space_vertex.w) * 0.5 + 0.5;
  vec2 l = vec2(0.19134182940220767, 0.46193971935850725) * blur;

  float total_weight = 0.0;
  //             -kernel_range <= +kernel_range
  for (float y = -0.4; y <= 0.4; y += 0.2) {
    for (float x = -0.4; x <= 0.4; x += 0.2) {
        float exponent = -1.0 * (x * x + y * y) / 2.0 * sigma_squared; // Gaussian RBDF.
        float local_weight = 1.0 / (2.0 * M_PI * sigma_squared) * pow(M_E, exponent);

        // float shadow = shadowmapBlur(shadow_map, projected_position, vec2(x, y) * shadow_map_stride, blur, light_depth);
        float shadow_depth = tex2Dproj(shadow_map, projected_position, vec2(x, y) * shadow_map_stride + l);
        float shadow = approximate_deep_shadow(shadow_depth, light_depth, strand_radius, strand_opacity);
        // float shadow = (light_depth < shadow_depth ? 1.0 : 0.0);

        visibility += shadow * local_weight;
        total_weight += local_weight;
    }
  }

  return visibility / total_weight;
}

#ifdef LIGHT_COUNT
void ShadowLighting(out ShadowData shadows, float blur) { // eG & eB
#ifdef SKIN
  for (int i = 0; i < LIGHT_COUNT; ++i) {
    shadows.val2[i] = 1.0;
  }
#endif
  vec3 worldPos = vWorldPos + blur * (gl_FrontFacing ? vNormal: -vNormal);
  vec4 pos = vec4(worldPos, 1.0);
#if LIGHT_COUNT > 0
  vec4 shadow_space_fragment = lightMatrices[0] * pos;
  //  float approximate_deep_shadows(sampler2D shadow_map, vec4 light_space_vertex, float smoothing, float strand_radius, float strand_opacity, float blur)
  shadows.val[0] = approximate_deep_shadows(shadowmap0, shadow_space_fragment, 7.5, 6000.0, 0.08, blur);
#if LIGHT_COUNT > 1
  shadow_space_fragment = lightMatrices[1] * pos;
  shadows.val[1] = approximate_deep_shadows(shadowmap1, shadow_space_fragment, 7.5, 6000.0, 0.08, blur);
#if LIGHT_COUNT > 2
  shadow_space_fragment = lightMatrices[2] * pos;
  shadows.val[2] = approximate_deep_shadows(shadowmap2, shadow_space_fragment, 7.5, 6000.0, 0.08, blur);
#if LIGHT_COUNT > 3
  shadow_space_fragment = lightMatrices[3] * pos;
  shadows.val[3] = approximate_deep_shadows(shadowmap3, shadow_space_fragment, 7.5, 6000.0, 0.08, blur);
#endif
#endif
#endif
#endif
}
#endif

vec3 Lighting(inout SkinData s, vec3 nor, vec3 camDir, float gloss) {
  vec3 res = vec3(1.0, 1.0, 1.0);
#ifdef LIGHT_COUNT
  res = vec3(0.0, 0.0, 0.0);
  highp float eo = 10.0 / log2(gloss * 0.968 + 0.03);
  eo = eo * eo;
  float eu = eo * 0.0397887 + 0.159155;
  eu = min(eu, 1000.0);

  ShadowData shadows;
  float shadow_kernel = 4.0 / 1536.0;
  float shadowblur = shadow_kernel + shadow_kernel * s.shadowBlur;
  shadowblur = 0.0005 + 0.5 * shadowblur;
  ShadowLighting(shadows, shadowblur);

  for (int i = 0; i < LIGHT_COUNT; ++i) {
    // vec3 lightDir = lightPositions[i] - vWorldPos;
    vec3 lightDir = lightDirections[i];
    float eI = inversesqrt(dot(lightDir, lightDir));
    lightDir *= eI;
    vec3 lightColor = lightColors[i];
#ifdef SKIN
    SkinLighting(s, shadows.val[i], shadows.val2[i], lightDir, nor, lightColor);
#else
    float intensity = clamp((1.0 / 3.1415926) * dot(lightDir, nor), 0.0, 1.0);
    intensity *= shadows.val[i];
    s.extra += intensity * lightColor;
#endif
    vec3 halfView = normalize(lightDir + camDir);
    float eP = eu * pow(clamp(dot(halfView, nor), 0.0, 1.0), eo);
    res += eP * lightColor;
  }
#endif

  return res;
}

vec2 Fresnel(vec2 intensity, vec2 fresnelIntensity) {
    intensity = 1.0 - intensity;
    vec2 sqr = intensity * intensity;
    intensity = mix(sqr * sqr, intensity * 0.4, fresnelIntensity);
    return intensity;
}

void Skin(out vec3 skinColor, out vec3 skinExtra, inout SkinData s, vec3 camDir, vec3 nor, float gloss) { //eQ
  float l = dot(camDir, nor); // fH

  s.extra *= s.SHColor * Fresnel(vec2(l, l), s.fresnelIntensity.xy).x;
  s.extra *= s.fresnelColor;
  float glossMask = clamp(1.0 + (-uFresnelGlossMask) * gloss, 0.0, 1.0); // hm
  s.extra *= glossMask * glossMask;
  s.ic = s.ic * uTransIntegral;

  s.hZ = (s.hZ) + s.smoothedSH;
  skinColor = mix(s.hY, s.hZ, s.bias);

  skinColor += s.ic * s.il;

  skinExtra = s.extra * s.bias;
}

vec3 Specular(vec3 camDir, vec3 nor, vec3 spec, float gloss) {
    float C = 1.0 - clamp(dot(camDir, nor), 0.0, 1.0);
    float cSqr = C * C;
    C *= cSqr * cSqr;
    C *= gloss * gloss;
    return (spec - C * spec) + C * uFresnel;
}

vec3 iY(vec3 c) {
    vec3 iZ = sqrt(c);
    return (iZ - iZ * c) + c * (0.4672 * c + vec3(0.5328));
}

vec3 roughnessFromNormal(vec3 worldNormal) {
  vec3 du = dFdx(worldNormal);
  vec3 dv = dFdy(worldNormal);

  float variance = 1.0 * (dot(du, du) + dot(dv, dv));
  float kernelRoughness = clamp(min(2.0 * variance, 1.0), 0.0, 1.0);
  kernelRoughness = sqrt(sqrt(kernelRoughness));
  return vec3(kernelRoughness, kernelRoughness, kernelRoughness);
}

vec3 SH9(vec3 fJ) {
  #define c(n) uSH9Coeffs[n].xyz
  vec3 G = (c(0) +
          fJ.y * ((c(1) + c(4) * fJ.x) + c(5) * fJ.z)) +
          fJ.x * (c(3) + c(7) * fJ.z) + c(2) * fJ.z;
  #undef c
  vec3 sqr = fJ * fJ;
  G += uSH9Coeffs[6].xyz * (3.0 * sqr.z - 1.0);
  G += uSH9Coeffs[8].xyz * (sqr.x - sqr.y);
  return G;
}

void main(void)
{
  vec4 alb = texture2D(uAlbedo, vTexcoord); // m
  vec3 col = SqrV3(alb.rgb); // dF
  float alpha = alb.a; // e

#ifdef MH
  vec4 NRRT = texture2D(uSpecular, vTexcoord);
  vec3 nor = -UnpackNormal(normalize(vec3(NRRT.a, NRRT.g, -1.0)));
#else
  vec3 nor = UnpackNormal(texture2D(uNormal, vTexcoord).xyz); // dI
#endif
  vec3 camDir = normalize(uCameraPosition - vWorldPos); // dO

  SkinData s; // dY
#ifdef SKIN
  vec4 m;
  s.subdermisColor = uSubdermisColor; // hX
  s.bias = 1.0;
  s.transColor = uTransColor.rgb; // ij
  s.transScatter = uTransScatter; // ik
  s.il = max(max(s.transColor.r, s.transColor.g), s.transColor.b) * uTransColor.a;
  float io = max(s.subdermisColor.r, max(s.subdermisColor.g, s.subdermisColor.b));
  io = 1.0 - io;
  io *= io;
  io *= io;
  io *= io;
  io = 1.0 - (io * io);
  s.bias *= io;
  s.shadowBlur = uSkinShadowBlur * s.bias * dot(s.subdermisColor.rgb, vec3(0.333, 0.334, 0.333));
  s.transColor *= SqrV3(texture2D(uTranslucency, fract(vTexcoord) * uTexRangeTranslucency.xy + uTexRangeTranslucency.zw).rgb);
  s.hZ = s.ic = s.extra = vec3(0.0);
#else
  s.extra = SH9(nor);
  // s.extra = vec3(0.1, 0.1, 0.1);
#endif

#ifdef MH
  vec2 snor = texture2D(uSpecular, vTexcoord, uNormalSmooth * s.bias * 2.5).ag;
  s.normal = -UnpackNormal(normalize(vec3(snor.r, snor.g, -1.0)));

  vec3 spec = vec3(NRRT.r, NRRT.r, NRRT.r) * 0.05;
  float gloss = NRRT.b;
#else
  s.normal = UnpackNormalLOD(uNormal, vTexcoord, uNormalSmooth * s.bias); // ie

  vec3 spec = SqrV3(texture2D(uSpecular, vTexcoord).xyz); // dP
  // vec3 spec = SqrV3(roughnessFromNormal(s.normal)) * 0.3;

  float gloss = texture2D(uGloss, vTexcoord).r; // dQ
#endif

#ifdef SKIN
  vec3 L0, L1, L2;
  SH9_SKIN(L0, L1, L2, s.normal);
  s.SHColor = s.hY = L0 + L1 + L2;
  // s.SHColor = s.hY = vec3(0.05, 0.05, 0.05);

  s.smoothedSH = SmootshadowPosH9(L0, L1, L2, vec3(1.0, 0.6667, 0.25), s.subdermisColor * 0.2 + vec3(0.1));
  // s.smoothedSH = vec3(0.05, 0.05, 0.05);
  s.id = vec3(0.0);
  s.subdermisColor *= 0.5;
  s.transScatter *= 0.5;
  s.fresnelColor = uFresnelColor.rgb;
  s.fresnelIntensity = uFresnelColor.aaa * vec3(1.0, 0.5, 0.25);
#endif

  vec3 halfView = reflect(-camDir, nor); // ek
  vec3 specular = environmentmap(halfView, 7.0 * gloss); // el

  float specularIntensity = SpecularIntensity(dot(halfView, gl_FrontFacing ? vNormal: -vNormal));
  specular *= specularIntensity;

  vec3 lighting = Lighting(s, nor, camDir, gloss); // el2

#ifdef SKIN
  vec3 color, skinExtra;
  Skin(color, skinExtra, s, camDir, nor, gloss);
#else
  vec3 color = s.extra;
#endif

  vec3 si = Specular(camDir, nor, spec, gloss);
  specular *= si;
  lighting *= si;

  vec3 res = (color * col);
  res += lighting + specular;
#ifdef SKIN
  res += skinExtra;
#endif

  gl_FragColor.xyz = iY(res.xyz);
  gl_FragColor.a = alpha;
}`,
  },
  pbr: {
    vs: `#pragma DEFINES
precision mediump float;
attribute vec3 position;
attribute vec3 color;
attribute vec3 normal;
attribute vec2 texcoord;
attribute vec3 tangent;
attribute vec3 bitangent;

uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

#pragma UNIFORMS

varying highp vec3 vWorldPos;
varying mediump vec3 vTangent;
varying mediump vec3 vBitangent;
varying mediump vec3 vNormal;
varying mediump vec2 vTexcoord;

void main(void)
{
  vec3 finalPos = position;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(finalPos, 1.0);
  vTexcoord = texcoord;
  vNormal = normal;
  vTangent = tangent;
  vBitangent = bitangent;
  vWorldPos = (modelMatrix * vec4(finalPos, 1.0)).xyz;
}`,
    fs: `#pragma DEFINES
#extension GL_OES_standard_derivatives : enable

#define M_PI 3.14159265358979323846
#define M_E  2.71828182845904523536

precision mediump float;
uniform sampler2D uAlbedo;
uniform sampler2D uNormal;

#ifdef CAVITY_TEX
uniform sampler2D uCavity;
#endif

#ifdef TRANSLUCENCY_TEX
uniform sampler2D uTranslucency;
#endif

#ifdef ENVIR_TEX
uniform sampler2D uEnvir;
#endif

#ifdef METALLIC_TEX
uniform sampler2D uMetallic;
#endif

#ifdef AO_TEX
uniform sampler2D uAO;
#endif

#ifdef ROUGHNESS_TEX
uniform sampler2D uRoughness;
#else
#ifdef SPECULAR_TEX
uniform sampler2D uSpecular;
#ifdef GLOSS_TEX
uniform sampler2D uGloss;
#endif
#endif
#endif

#ifdef DIRECTION_TEX
uniform sampler2D uDirection;
#endif

uniform sampler2D uBRDF;

uniform vec3 uCameraPosition;

#ifdef LIGHT_COUNT
uniform vec3 lightPositions[LIGHT_COUNT];
uniform vec3 lightDirections[LIGHT_COUNT];
uniform vec3 lightColors[LIGHT_COUNT];
#if LIGHT_COUNT > 0
uniform sampler2D shadowmap0;
#if LIGHT_COUNT > 1
uniform sampler2D shadowmap1;
#if LIGHT_COUNT > 2
uniform sampler2D shadowmap2;
#if LIGHT_COUNT > 3
uniform sampler2D shadowmap3;
#endif
#endif
#endif
#endif
uniform mat4 lightMatrices[LIGHT_COUNT];

struct ShadowData {
  float val[LIGHT_COUNT];
};
#endif

#pragma UNIFORMS

varying highp vec3 vWorldPos; // dv
varying mediump vec3 vTangent; // dA
varying mediump vec3 vBitangent; // dB
varying mediump vec3 vNormal; // dC
varying mediump vec2 vTexcoord; // d

vec3 UnpackNormal(vec3 normalmap) {
  vec3 tan = vTangent;
  vec3 bit = vBitangent;
  vec3 nor = gl_FrontFacing ? vNormal: -vNormal;
  normalmap = normalmap * 2.0 - vec3(1.0);
  vec3 res = tan * normalmap.x + bit * normalmap.y + nor * normalmap.z;
  return normalize(res);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
  float a = roughness*roughness;
  float a2 = a*a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH*NdotH;

  float nom   = a2;
  float denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = M_PI * denom * denom;

  return nom / max(denom, 0.0000001); // prevent divide by zero for roughness=0.0 and NdotH=1.0
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float roughnessFromNormal(vec3 worldNormal) {
  vec3 du = dFdx(worldNormal);
  vec3 dv = dFdy(worldNormal);

  float variance = 1.0 * (dot(du, du) + dot(dv, dv));
  float kernelRoughness = clamp(min(2.0 * variance, 1.0), 0.0, 1.0);
  kernelRoughness = sqrt(sqrt(kernelRoughness));
  return kernelRoughness;
}

float approximate_deep_shadow(float shadow_depth, float light_depth, float strand_radius, float strand_alpha) {
  float depth = max(0.0, light_depth - shadow_depth); // depth of the current strand inside geometry.

  float strand_count = depth * strand_radius;

  if (depth > 1e-10) strand_count += 1.0;

  return pow(1.0 - strand_alpha, strand_count);
}

float tex2Dproj(sampler2D image, vec2 projected_position, vec2 displacement) {
  return texture2D(image, projected_position.st + displacement).r;
}

float approximate_deep_shadows(sampler2D shadow_map, vec4 light_space_vertex, float smoothing, float strand_radius, float strand_opacity, float blur) {
  float visibility = 0.0;
  vec2 shadow_map_size = vec2(1536.0, 1536.0);

  float kernel_width = 100.0 * blur;
  // float kernel_range = (kernel_width - 1.0) / 2.0;
  float sigma_stddev = (kernel_width * 0.5) / 2.4;
  float sigma_squared = sigma_stddev * sigma_stddev;

  float light_depth = (light_space_vertex.z / light_space_vertex.w) * 0.5 + 0.5;
  vec2 shadow_map_stride = smoothing / shadow_map_size; // stride.
  vec2 projected_position = (light_space_vertex.xy / light_space_vertex.w) * 0.5 + 0.5;
  vec2 l = vec2(0.19134182940220767, 0.46193971935850725) * blur;

  float total_weight = 0.0;
  //             -kernel_range <= +kernel_range
  for (float y = -0.4; y <= 0.4; y += 0.2) {
    for (float x = -0.4; x <= 0.4; x += 0.2) {
        float exponent = -1.0 * (x * x + y * y) / 2.0 * sigma_squared; // Gaussian RBDF.
        float local_weight = 1.0 / (2.0 * M_PI * sigma_squared) * pow(M_E, exponent);

        // float shadow = shadowmapBlur(shadow_map, projected_position, vec2(x, y) * shadow_map_stride, blur, light_depth);
        float shadow_depth = tex2Dproj(shadow_map, projected_position, vec2(x, y) * shadow_map_stride + l);
        float shadow = approximate_deep_shadow(shadow_depth, light_depth, strand_radius, strand_opacity);
        // float shadow = (light_depth < shadow_depth ? 1.0 : 0.0);

        visibility += shadow * local_weight;
        total_weight += local_weight;
    }
  }

  return visibility / total_weight;
}

#ifdef LIGHT_COUNT
void ShadowLighting(out ShadowData shadows, float blur) { // eG & eB
  vec3 worldPos = vWorldPos + blur * (gl_FrontFacing ? vNormal: -vNormal);
  vec4 pos = vec4(worldPos, 1.0);
#if LIGHT_COUNT > 0
  vec4 shadow_space_fragment = lightMatrices[0] * pos;
  //  float approximate_deep_shadows(sampler2D shadow_map, vec4 light_space_vertex, float smoothing, float strand_radius, float strand_opacity, float blur)
  shadows.val[0] = approximate_deep_shadows(shadowmap0, shadow_space_fragment, 7.5, 6000.0, 0.08, blur);
#if LIGHT_COUNT > 1
  shadow_space_fragment = lightMatrices[1] * pos;
  shadows.val[1] = approximate_deep_shadows(shadowmap1, shadow_space_fragment, 7.5, 6000.0, 0.08, blur);
#if LIGHT_COUNT > 2
  shadow_space_fragment = lightMatrices[2] * pos;
  shadows.val[2] = approximate_deep_shadows(shadowmap2, shadow_space_fragment, 7.5, 6000.0, 0.08, blur);
#if LIGHT_COUNT > 3
  shadow_space_fragment = lightMatrices[3] * pos;
  shadows.val[3] = approximate_deep_shadows(shadowmap3, shadow_space_fragment, 7.5, 6000.0, 0.08, blur);
#endif
#endif
#endif
#endif
}
#endif

#ifdef ENVIR_TEX
vec3 environmentmap(vec3 halfView, float gloss) {
  halfView /= dot(vec3(1.0, 1.0, 1.0), abs(halfView));
  vec2 fU = abs(halfView.zx) - vec2(1.0, 1.0);
  vec2 fV = vec2(halfView.x < 0.0 ? fU.x : -fU.x , halfView.z < 0.0 ? fU.y : -fU.y);
  vec2 texcoord = (halfView.y < 0.0) ? fV : halfView.xz;
  texcoord = vec2(0.5 * (254.0 / 256.0), 0.125 * 0.5 * (254.0 / 256.0)) * texcoord + vec2(0.5, 0.125 * 0.5);
  gloss = gloss * 7.0;
  float fX = fract(gloss);
  texcoord.y += 0.125 * (gloss - fX);
  vec4 fZ = texture2D(uEnvir, texcoord);
  vec3 r = fZ.xyz * (7.0 * fZ.w);
  return r * r;
}
#endif

vec3 Translucency(float translucencyTex, vec3 view, vec3 norm, vec3 lightDir, vec3 transColor) {
  float distortion = 0.3;
  float ltscale = 0.3;
  float power = 8.5;
  vec3 ltlight = lightDir + norm * distortion;
  float intensity = pow(clamp(dot(view, -ltlight), 0.0, 1.0), power) * ltscale * translucencyTex;
  return transColor * intensity;
}

float CalSpecular(float NdotV, float spec, float gloss) {
  float C = 1.0 - NdotV;
  float cSqr = C * C;
  C *= cSqr * cSqr;
  C *= gloss * gloss;
  return (spec - C * spec);
}

vec3 kajiya_kay(vec3 diffuse, vec3 specular, float p, vec3 tangent, vec3 light, vec3 eye) {
  float cosTL = dot(tangent, light);
  float cosTE = dot(tangent, eye);

  float cosTL_squared = cosTL * cosTL;
  float cosTE_squared = cosTE * cosTE;

  float one_minus_cosTL_squared = 1.0 - cosTL_squared;
  float one_minus_cosTE_squared = 1.0 - cosTE_squared;

  float sinTL = sqrt(one_minus_cosTL_squared);
  float sinTE = sqrt(one_minus_cosTE_squared);

  vec3 diffuse_colors = diffuse * sinTL;
  float spec = cosTL * cosTE + sinTL * sinTE;
  vec3 specular_colors = specular * pow(max(spec, 0.0), p);
  return diffuse_colors + specular_colors * 0.5;
}

void main(void)
{
  vec4 albedoTex = texture2D(uAlbedo, vTexcoord);
  vec3 albedo = pow(albedoTex.rgb, vec3(2.1632601288));

#ifdef CAVITY_TEX
  // https://help.quixel.com/hc/en-us/articles/115000612165-What-maps-are-included-and-how-do-I-use-them-
  float cavity = texture2D(uCavity, vTexcoord).r;
#else
  float cavity = 0.2;
#endif

#ifdef METALLIC_TEX
  float metallic = texture2D(uMetallic, vTexcoord).r;
#else
  float metallic = 0.5;
#ifdef SKIN
  metallic = 0.0;
#endif
#endif
  vec3 N = UnpackNormal(texture2D(uNormal, vTexcoord).rgb);
#ifdef ROUGHNESS_TEX
  float roughness = texture2D(uRoughness, vTexcoord).r;
#else
#ifdef SPECULAR_TEX
  float roughness = 1.0 - texture2D(uSpecular, vTexcoord).r;
#ifdef GLOSS_TEX
  float gloss = texture2D(uGloss, vTexcoord).r;
#endif
#else
  float roughness = 1.0 - roughnessFromNormal(N);
#ifndef CAVITY_TEX
  roughness *= 0.87;
#endif
  roughness *= roughness;
#endif
#endif
#ifdef AO_TEX
  float ao = texture2D(uAO, vTexcoord).r;
#else
  float ao = 1.0;
#endif
  vec3 view = normalize(uCameraPosition - vWorldPos);

#ifdef TRANSLUCENCY_TEX
  float translucencyTex = texture2D(uTranslucency, vTexcoord).r;
#endif

#ifdef DIRECTION_TEX
  vec3 dirmap = UnpackNormal(texture2D(uDirection, vTexcoord).rgb);
  vec3 tan = mix(dirmap, N, 0.0);
#endif

  vec3 F0 = vec3(0.04);
  F0 = mix(F0, albedo, metallic);
  vec3 Lo = vec3(0.0);

  float NdotV = max(dot(N, view), 0.0);

#ifdef LIGHT_COUNT
  ShadowData shadows;
  float shadow_kernel = 4.0 / 1536.0;
  float shadowblur = shadow_kernel + shadow_kernel;
  shadowblur = 0.0005 + 0.5 * shadowblur;
  ShadowLighting(shadows, shadowblur);

  for(int i = 0; i < LIGHT_COUNT; i++) {
#ifdef DIRECTION_TEX
    vec3 lightDir = lightDirections[i];

    float NdotL = max(dot(N, lightDir), 0.0);

    vec3 L = kajiya_kay(albedo * lightColors[i], vec3(1.0 - roughness), 30000.0, tan, lightDir, -view);

    L *= NdotL * shadows.val[i];
    Lo += L;
#else
    // vec3 lightDir = lightPositions[i] - vWorldPos;
    vec3 lightDir = lightDirections[i];
    vec3 H = normalize(view + lightDir);
    vec3 radiance = lightColors[i];
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, view, lightDir, roughness);
    vec3 F = fresnelSchlick(max(dot(H, view), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * max(dot(N, lightDir), 0.0) + 0.001;
    vec3 specular = cavity * numerator / denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    float NdotL = max(dot(N, lightDir), 0.0);

    vec3 L = (kD * albedo / M_PI + specular) * radiance;

#ifdef TRANSLUCENCY_TEX
    vec3 translucency = Translucency(translucencyTex, view, N, lightDir, albedo * radiance);
    L += translucency;
#endif

    L *= NdotL * shadows.val[i];
    Lo += L;
#endif
  }
#endif
  vec3 F = fresnelSchlickRoughness(NdotV, F0, roughness);
  vec3 kS = F;
  vec3 kD = 1.0 - kS;
  kD *= 1.0 - metallic;

  vec3 irradiance = vec3(1.0, 1.0, 1.0);
  vec3 diffuse = irradiance * albedo;

#ifdef ENVIR_TEX
  vec3 R = reflect(-view, N);
  vec3 envir = environmentmap(R, (1.0 - roughness)).rgb;
#else
  vec3 envir = vec3(0.0, 0.0, 0.0);
#endif
  vec2 brdf = texture2D(uBRDF, vec2(NdotV, roughness)).rg;
  vec3 specular = (envir * (F * brdf.x + brdf.y)) * cavity;

  vec3 ambient = (kD * diffuse + specular) * ao;
#ifdef SKIN
  ambient *= 0.2;
#endif
  vec3 color = ambient + Lo;

  color = color / (color + vec3(1.0));
  color = pow(color, vec3(0.46226525728));
#ifdef DIRECTION_TEX
  if (albedoTex.a < 0.5) discard;
#else
  if (albedoTex.a < 0.01) discard;
#endif
  gl_FragColor = vec4(color, albedoTex.a);
}`,
  },
};

const keys = Object.keys(shadersMap);
for (let i = keys.length - 1; i >= 0; i -= 1) {
  const k = keys[i];
  shadersMap[k].name = k;
  shadersMap[k].vs = removeShaderText(shadersMap[k].vs);
  shadersMap[k].fs = removeShaderText(shadersMap[k].fs);
}

function loadShader(type, name, s) {
  const source = s.replace('#pragma DEFINES', '').replace('#pragma UNIFORMS', '');
  const shader = gl.createShader(type);

  // Send the source to the shader object
  gl.shaderSource(shader, source);

  // Compile the shader program
  gl.compileShader(shader);

  // See if it compiled successfully
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(`An error occurred compiling the shader "${name}" :\n${gl.getShaderInfoLog(shader)}`);
    if (isMobile) {
      // eslint-disable-next-line no-alert
      alert(`shader(${name}) : ${source}`);
    } else {
      console.error(`shader(${name}) : ${source}`);
    }
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

function initShaderProgram(shaderSource) {
  const vertexShader = loadShader(gl.VERTEX_SHADER, shaderSource.name, shaderSource.vs);
  const fragmentShader = loadShader(gl.FRAGMENT_SHADER, shaderSource.name, shaderSource.fs);

  // Create the shader program
  const shaderProgram = gl.createProgram();
  if (vertexShader !== null && fragmentShader !== null) {
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);
  }
  // If creating the shader program failed, alerts
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    console.error(`Unable to initialize the shader program: ${gl.getProgramInfoLog(shaderProgram)}`);
    return null;
  }

  return shaderProgram;
}

export default class Shader {
  constructor(shaderSource) {
    this.shaderSource = shaderSource;

    this.program = initShaderProgram(shaderSource);

    this.attrib_cache = {};
    this.uniform_cache = {};
    this.value_cache = {};

    return this;
  }

  get getProgram() { return this.program; }

  attribLocation(name) {
    let location = this.attrib_cache[name];
    if (location === undefined) {
      location = gl.getAttribLocation(this.program, name);
      this.attrib_cache[name] = location;
    }
    if (location >= 0) {
      gl.enableVertexAttribArray(location);
    }
    return location;
  }

  bind() {
    if (this !== programUsing) {
      programUsing = this;
      gl.useProgram(this.program);
    }
    return this;
  }

  unbind() {
    if (programUsing === this) {
      programUsing = null;
      gl.useProgram(null);
    }
    return this;
  }

  getUniformLocation(name) {
    let location = this.uniform_cache[name];
    if (location === undefined) {
      location = gl.getUniformLocation(this.program, name);
      this.uniform_cache[name] = location;
    }
    if (location === undefined) console.error(`name : ${name} is not found in shader ${this.shaderSource.name}`);
    return location;
  }

  setInt(name, value, debug = false) {
    if (this.value_cache[name] !== value) {
      this.value_cache[name] = value;
      const uniLoc = this.getUniformLocation(name);
      if (uniLoc) gl.uniform1i(uniLoc, value);
      else console.log(`setInt failed for ${name}`);
    }
    if (debug) {
      console.log(`setInt for ${name} : ${value}`);
    }
    return this;
  }

  setIv(name, values) {
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniform1iv(uniLoc, values);
    return this;
  }

  setFloat(name, value) {
    if (this.value_cache[name] !== value) {
      this.value_cache[name] = value;
      const uniLoc = this.getUniformLocation(name);
      if (uniLoc) gl.uniform1f(uniLoc, value);
    }
    return this;
  }

  setFv(name, values) {
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniform1fv(uniLoc, values);
    return this;
  }

  setVal2(name, x, y) {
    const cached = this.value_cache[name];
    if (cached) {
      if (cached[0] !== x || cached[1] !== y) {
        cached[0] = x;
        cached[1] = y;
        this.value_cache[name] = cached;
      } else return this;
    } else {
      this.value_cache[name] = [x, y];
    }
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniform2f(uniLoc, x, y);
    // else console.log(`Cannot find uniLoc for name: ${name}`);
    return this;
  }

  setVal3(name, x, y, z) {
    const cached = this.value_cache[name];
    if (cached) {
      if (cached[0] !== x || cached[1] !== y || cached[2] !== z) {
        cached[0] = x;
        cached[1] = y;
        cached[2] = z;
        this.value_cache[name] = cached;
      } else return this;
    } else {
      this.value_cache[name] = [x, y, z];
    }
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniform3f(uniLoc, x, y, z);
    // else console.log(`Cannot find uniLoc for name: ${name}`);
    return this;
  }

  setVal4(name, x, y, z, w) {
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniform4f(uniLoc, x, y, z, w);
    // else console.log(`Cannot find uniLoc for name: ${name}`);
    return this;
  }

  setVec2(name, vec) {
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniform2fv(uniLoc, vec);
    return this;
  }

  setVec3(name, vec) {
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniform3fv(uniLoc, vec);
    // else console.log(`Cannot find uniLoc for name: ${name}`);
    return this;
  }

  setVec4(name, vec) {
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniform4fv(uniLoc, vec);
    return this;
  }

  setMatrix2(name, matrix) {
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniformMatrix2fv(uniLoc, false, matrix);
    return this;
  }

  setMatrix3(name, matrix) {
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniformMatrix3fv(uniLoc, false, matrix);
    return this;
  }

  setMatrix4(name, matrix) {
    const uniLoc = this.getUniformLocation(name);
    if (uniLoc) gl.uniformMatrix4fv(uniLoc, false, matrix);
    return this;
  }

  static FindShaderSource(name) {
    if (typeof name === 'string') {
      if (name in shadersMap) {
        return shadersMap[name];
      }
      // eslint-disable-next-line no-console
      console.error(`Cannot find shader ${name}`);
      return null;
    }
    return name;
  }

  static ShaderMap() {
    return shadersMap;
  }
}

export { Shader };
