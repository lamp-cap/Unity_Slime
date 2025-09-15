  Shader "Instanced/Particle3D" {
    Properties {
        _Color ("Color", Color) = (1, 1, 1, 1)
        _Size ("Size", float) = 0.035
    }

    SubShader {
        Pass 
        {
            Tags { 
                "RenderPipeline" = "UniversalRenderPipeline" 
                "Queue" = "Transparent" 
                "IgnoreProjector" = "True" 
                "RenderType" = "Transparent" 
                "PreviewType" = "Plane"
            }
            
            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            #pragma enable_d3d11_debug_symbols
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // matches the structure of our data on the CPU side
            struct Particle {
                float3 x;
                int ID;
            };

            struct a2v {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f {
                float4 pos : SV_POSITION;
                float4 uv : TEXCOORD0;
                float3 normal : TEXCOORD1;
            };

            float _Size;
            float4 _Color;
            int _Aniso;

            StructuredBuffer<Particle> _ParticleBuffer;
            StructuredBuffer<float4x4> _CovarianceBuffer;

            v2f vert (a2v v, uint id : SV_InstanceID) 
            {
                v2f o;
                float3x3 covMatrix = _Aniso > 0 ? (float3x3)_CovarianceBuffer[id] : float3x3(1,0,0,0,1,0,0,0,1);
                float3 anisoPos = mul(covMatrix, v.vertex.xyz);
                float3 worldPosition = (_ParticleBuffer[id].x * 0.1) + anisoPos * _Size;
                // project into camera space
                o.pos = TransformWorldToHClip(worldPosition);
                o.normal = TransformObjectToWorldNormal(v.normal);
                o.uv = float4(_ParticleBuffer[id].x, _ParticleBuffer[id].ID);
                
                return o;
            }
            static const float4 colors[8] = {
                float4(1, 0, 0, 1),
                float4(1, 0.5, 0, 1),
                float4(1, 1, 0, 1),
                float4(0.5, 1, 0, 1),
                float4(0, 1, 0, 1),
                float4(0, 1, 0.5, 1),
                float4(0, 1, 1, 1),
                float4(0, 0, 1, 1),
            };

            float4 frag (v2f i) : SV_Target 
            {
                float NoL = max(0, dot(GetMainLight().direction, i.normal));
                float3 sh = SampleSH(i.normal);
                int id = round(i.uv.w);
                float4 color = id >-0.5 ?  colors[((uint)id) & 7] : _Color;
                return float4(color.rgb*(NoL.xxx + sh), 1);
            }

            ENDHLSL
        }
    }
}
