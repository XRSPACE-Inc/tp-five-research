// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Render Splats1"
{
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite Off
            Blend OneMinusDstAlpha One
            Cull Off
            
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma require compute
#pragma use_dxc

// #include "GaussianSplatting.hlsl"

float4x4 _MatrixObjectToWorld;
float4x4 _MatrixVP;
float4x4 _MatrixMV;
float4x4 _MatrixP;
float4 _VecScreenParams;

StructuredBuffer<uint> _OrderBuffer;

struct v2f
{
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    float4 vertex : SV_POSITION;
};
struct SplatData2
{
    float3 pos;
    float4 rotation;
    float3 scale;
};

StructuredBuffer<SplatData2> _SplatViewData;
Texture2D _SplatColor;

static const uint kTexWidth = 2048;
static const uint kTexHeight = 16;

uint2 DecodeMorton2D_16x16(uint t)      // --------EAFBGCHD
{
    t = (t & 0xFF) | ((t & 0xFE) << 7); // -EAFBGCHEAFBGCHD
    t &= 0x5555;                        // -E-F-G-H-A-B-C-D
    t = (t ^ (t >> 1)) & 0x3333;        // --EF--GH--AB--CD
    t = (t ^ (t >> 2)) & 0x0f0f;        // ----EFGH----ABCD
    return uint2(t & 0xF, t >> 8);      // --------EFGHABCD
}

uint3 SplatIndexToPixelIndex(uint idx)
{
    uint3 res;

    uint2 xy = DecodeMorton2D_16x16(idx);
    uint width = kTexWidth / 16;
    idx >>= 8;
    res.x = (idx % width) * 16 + xy.x;
    res.y = (idx / width) * 16 + xy.y;
    res.z = 0;
    return res;
}

float3x3 CalcMatrixFromRotationScale(float4 rot, float3 scale)
{
    float3x3 ms = float3x3(
        scale.x, 0, 0,
        0, scale.y, 0,
        0, 0, scale.z
    );
    float x = rot.x;
    float y = rot.y;
    float z = rot.z;
    float w = rot.w;
    float3x3 mr = float3x3(
        1-2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
          2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x),
          2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)
    );
    return mul(mr, ms);
}

void CalcCovariance3D(float3x3 rotMat, out float3 sigma0, out float3 sigma1)
{
    float3x3 sig = mul(rotMat, transpose(rotMat));
    sigma0 = float3(sig._m00, sig._m01, sig._m02);
    sigma1 = float3(sig._m11, sig._m12, sig._m22);
}

void DecomposeCovariance(float3 cov2d, out float2 v1, out float2 v2)
{
    // same as in antimatter15/splat
    float diag1 = cov2d.x, diag2 = cov2d.z, offDiag = cov2d.y;
    float mid = 0.5f * (diag1 + diag2);
    float radius = length(float2((diag1 - diag2) / 2.0, offDiag));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    float2 diagVec = normalize(float2(offDiag, lambda1 - diag1));
    diagVec.y = -diagVec.y;
    float maxSize = 4096.0;
    v1 = min(sqrt(2.0 * lambda1), maxSize) * diagVec;
    v2 = min(sqrt(2.0 * lambda2), maxSize) * float2(diagVec.y, -diagVec.x);
}

// from "EWA Splatting" (Zwicker et al 2002) eq. 31
float3 CalcCovariance2D(float3 worldPos, float3 cov3d0, float3 cov3d1, float4x4 matrixV, float4x4 matrixP, float4 screenParams)
{
    float4x4 viewMatrix = matrixV;
    float3 viewPos = mul(viewMatrix, float4(worldPos, 1)).xyz;

    // this is needed in order for splats that are visible in view but clipped "quite a lot" to work
    float aspect = matrixP._m00 / matrixP._m11;
    float tanFovX = rcp(matrixP._m00);
    float tanFovY = rcp(matrixP._m11 * aspect);
    float limX = 1.3 * tanFovX;
    float limY = 1.3 * tanFovY;
    viewPos.x = clamp(viewPos.x / viewPos.z, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y / viewPos.z, -limY, limY) * viewPos.z;

    float focal = screenParams.x * matrixP._m00 / 2;

    float3x3 J = float3x3(
        focal / viewPos.z, 0, -(focal * viewPos.x) / (viewPos.z * viewPos.z),
        0, focal / viewPos.z, -(focal * viewPos.y) / (viewPos.z * viewPos.z),
        0, 0, 0
    );
    float3x3 W = (float3x3)viewMatrix;
    float3x3 T = mul(J, W);
    float3x3 V = float3x3(
        cov3d0.x, cov3d0.y, cov3d0.z,
        cov3d0.y, cov3d1.x, cov3d1.y,
        cov3d0.z, cov3d1.y, cov3d1.z
    );
    float3x3 cov = mul(T, mul(V, transpose(T)));

    // Low pass filter to make each splat at least 1px size.
    cov._m00 += 0.3;
    cov._m11 += 0.3;
    return float3(cov._m00, cov._m01, cov._m11);
}

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o = (v2f)0;
    instID = _OrderBuffer[instID];
    SplatData2 view = _SplatViewData[instID];
    float3 centerWorldPos = mul(_MatrixObjectToWorld, float4(view.pos,1)).xyz;
    float4 centerClipPos = mul(_MatrixVP, float4(centerWorldPos, 1));
    // float4 centerClipPos = view.pos;
    bool behindCam = centerClipPos.w <= 0;
    if (behindCam)
    {
        o.vertex = asfloat(0x7fc00000); // NaN discards the primitive
    }
    else
    {
        float4 boxRot = normalize(view.rotation.yzwx);
        float3 boxSize = exp(view.scale);

        float3x3 splatRotScaleMat = CalcMatrixFromRotationScale(boxRot, boxSize);

        float3 cov3d0, cov3d1;
        CalcCovariance3D(splatRotScaleMat, cov3d0, cov3d1);
        float3 cov2d = CalcCovariance2D(view.pos, cov3d0, cov3d1, _MatrixMV, _MatrixP, _VecScreenParams);
        float2 axis1, axis2;
        DecomposeCovariance(cov2d, axis1, axis2);

        uint3 coord = uint3(instID % kTexWidth, instID / kTexWidth, 0);
        o.col = _SplatColor.Load(coord);

        uint idx = vtxID;
        float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
        quadPos *= 2;

        o.pos = quadPos;

        float2 deltaScreenPos = (quadPos.x * axis1 + quadPos.y * axis2) * 2 / _ScreenParams.xy;
        o.vertex = centerClipPos;
        o.vertex.xy += deltaScreenPos * centerClipPos.w;
    }
    return o;
}

half4 frag (v2f i) : SV_Target
{
    float power = -dot(i.pos, i.pos);
    half alpha = exp(power);
    if (i.col.a >= 0)
    {
        alpha = saturate(alpha * i.col.a);
    }
    else
    {
        // "selected" splat: magenta outline, increase opacity, magenta tint
        half3 selectedColor = half3(1,0,1);
        if (alpha > 7.0/255.0)
        {
            if (alpha < 10.0/255.0)
            {
                alpha = 1;
                i.col.rgb = selectedColor;
            }
            alpha = saturate(alpha + 0.3);
        }
        i.col.rgb = lerp(i.col.rgb, selectedColor, 0.5);
    }
    
    if (alpha < 1.0/255.0)
        discard;

    half4 res = half4(i.col.rgb * alpha, alpha);
    return res;
}
ENDCG
        }
    }
}
