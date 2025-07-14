#ifndef MATCAP_INC
#define MATCAP_INC
#ifndef UNIVERSAL_PIPELINE_CORE_INCLUDED
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
#endif  // UNIVERSAL_PIPELINE_CORE_INCLUDED
#ifndef BUILTIN_TARGET_API  // fix: redefinition of 'PackHeightmap'
#ifndef UNIVERSAL_LIGHTING_INCLUDED
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
#endif  // UNIVERSAL_LIGHTING_INCLUDED
#endif  // BUILTIN_TARGET_API

struct Attributes
{
    float4 positionOS : POSITION;
    float3 normalOS : NORMAL;
    float2 texcoord : TEXCOORD0;
    UNITY_VERTEX_INPUT_INSTANCE_ID
};

struct Veryings
{
    float4 positionCS : SV_POSITION;
    float2 uv : TEXCOORD0;
    half2 vN : TEXCOORD1;
    half3 SHLighting : TEXCOORD2;
    UNITY_VERTEX_INPUT_INSTANCE_ID
    UNITY_VERTEX_OUTPUT_STEREO
};


struct Attributes2
{
    float4 positionOS : POSITION;
    float3 normalOS : NORMAL;
    float2 texcoord : TEXCOORD0;
    float4 texcoord1 : TEXCOORD1;
    float4 texcoord2 : TEXCOORD2;
    UNITY_VERTEX_INPUT_INSTANCE_ID
};

struct Veryings2
{
    float4 positionCS : SV_POSITION;
    float2 uv : TEXCOORD0;
    half2 vN : TEXCOORD1;
    half3 SHLighting : TEXCOORD2;
    half4 values : TEXCOORD3;
    UNITY_VERTEX_INPUT_INSTANCE_ID
    UNITY_VERTEX_OUTPUT_STEREO
};


half lum(half3 col)
{
    return col.r * 0.2 + col.g * 0.7 + col.b * 0.1;
}

half2 MatcapUV(float4 vvertex, float3 vnormal)
{
    float3 viewSpacePosition = normalize(mul(UNITY_MATRIX_MV, vvertex).xyz);
    float3 viewSpaceNormal = normalize(mul((float3x3)UNITY_MATRIX_IT_MV, vnormal));
    float3 viewDir = normalize(-viewSpacePosition);
    float3 x = normalize(float3(viewDir.z, 0.0, -viewDir.x));
    float3 y = normalize(cross(viewDir, x));
    half2 matcapUV = half2(dot(x, viewSpaceNormal), dot(y, viewSpaceNormal));

    return matcapUV * 0.49h + 0.5h;
}

#ifdef BUILTIN_TARGET_API  // fix: redefinition of 'PackHeightmap'
#define INITIALIZE_MATCAP(v, o) o.positionCS = TransformObjectToHClip(v.positionOS.xyz); \
    o.uv = TRANSFORM_TEX(v.texcoord, _MainTex); \
    o.vN = MatcapUV(v.positionOS, v.normalOS); \
    o.SHLighting = half3(0.1h, 0.1h, 0.1h);
#define INITIALIZE_MATCAP2(v, o) o.positionCS = TransformObjectToHClip(v.positionOS.xyz); \
    o.uv = TRANSFORM_TEX(v.texcoord.xy, _MainTex); \
    o.vN = MatcapUV(v.positionOS, v.normalOS); \
    o.SHLighting = half3(0.1h, 0.1h, 0.1h); \
    o.values = v.texcoord2; \
    if (v.texcoord1.z > 0) o.vN = (o.vN * v.texcoord1.zw) + v.texcoord1.xy;
#else
#define INITIALIZE_MATCAP(v, o) o.positionCS = TransformObjectToHClip(v.positionOS.xyz); \
    o.uv = TRANSFORM_TEX(v.texcoord, _MainTex); \
    o.vN = MatcapUV(v.positionOS, v.normalOS); \
    VertexNormalInputs normalInputs = GetVertexNormalInputs(v.normalOS.xyz); \
    o.SHLighting = SampleSHVertex(normalInputs.normalWS);
#define INITIALIZE_MATCAP2(v, o) o.positionCS = TransformObjectToHClip(v.positionOS.xyz); \
    o.uv = TRANSFORM_TEX(v.texcoord.xy, _MainTex); \
    o.vN = MatcapUV(v.positionOS, v.normalOS); \
    VertexNormalInputs normalInputs = GetVertexNormalInputs(v.normalOS.xyz); \
    o.SHLighting = SampleSHVertex(normalInputs.normalWS); \
    o.values = v.texcoord2; \
    if (v.texcoord1.z > 0) o.vN = (o.vN * v.texcoord1.zw) + v.texcoord1.xy;
#endif  // BUILTIN_TARGET_API
#endif  // MATCAP_INC
