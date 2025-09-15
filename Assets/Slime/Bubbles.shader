  Shader "Instanced/Bubbles" {
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
            }
            ZWrite Off
            ZTest On
            Blend SrcAlpha OneMinusSrcAlpha
            
            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            #pragma enable_d3d11_debug_symbols
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // matches the structure of our data on the CPU side
            struct Bubble
            {
                float3 Pos;
                float Radius;
                float3 Vel;
                float LifeTime;
            };

            struct a2v {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 normal : TEXCOORD1;
            };

            float _Size;
            float4 _Color;

            StructuredBuffer<Bubble> _BubblesBuffer;

            v2f vert (a2v v, uint id : SV_InstanceID) 
            {
                v2f o;
                Bubble b = _BubblesBuffer[id];
                float3 worldPosition = (b.Pos * 0.1) + (v.vertex.xyz - float3(0, 0.2, 0)) * (_Size * b.Radius);
                o.worldPos = worldPosition;
                // project into camera space
                o.pos = TransformWorldToHClip(worldPosition);
                o.normal = TransformObjectToWorldNormal(v.normal);
                
                return o;
            }

            float4 frag (v2f i) : SV_Target 
            {
                float3 L = GetMainLight().direction;
                float3 V = normalize(_WorldSpaceCameraPos - i.worldPos);
                float NoL = max(0, dot(L, i.normal));
                float NoH = max(0, dot(i.normal, normalize(V + L)));
                float NoV = dot(i.normal, V);
                float specular = pow(NoH, 100);
                float fresnell = (1 - NoV)*(1- NoV);
                float3 sh = SampleSH(i.normal);
                return float4(_Color.rgb + specular, min(1, fresnell + specular));
            }

            ENDHLSL
        }
    }
}
