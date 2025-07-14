Shader "Custom/MatCap Grayscale As Specular"
{
    Properties
    {
        [MainColor] _Color ("Main Color", Color) = (0.5,0.5,0.5,1)
        [MainTexture] _MainTex ("Texture (RGB)", 2D) = "white" {}
        _MatCap ("MatCap (RGB)", 2D) = "black" {}
        _Val ("Main color intensity", Range(-10.0 , 10.0)) = 0
        _Val2 ("Specular intenstiy", Range(-10.0 , 10.0)) = 0
    }

    Subshader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }
        Pass
        {
            Name "MatCap_GrayScaleSpecular"
            Tags { "LightMode"="UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vertex
            #pragma fragment frag
            #pragma fragmentoption ARB_precision_hint_fastest

            #include "MatCap.hlsl"

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            CBUFFER_START(UnityPerMaterial)
            sampler2D _MatCap;
            uniform half3 _Color;
            uniform float4 _MatCap_ST;
            uniform float4 _MainTex_ST;
            uniform half _Val;
            uniform half _Val2;
            CBUFFER_END

            Veryings vertex(Attributes v)
            {
                Veryings o;
                INITIALIZE_MATCAP(v, o)
                return o;
            }

            half4 frag (Veryings i) : COLOR
            {
                half3 matcap = tex2D(_MatCap, i.vN).rgb;
                half4 color = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv);
                half mask = lum(color.rgb);
                color.rgb = color.rgb * _Color * matcap * pow(2, _Val) + lerp(0, i.SHLighting * matcap * pow(2, _Val2), mask);
                return color;
            }
            ENDHLSL
        }
    }
}