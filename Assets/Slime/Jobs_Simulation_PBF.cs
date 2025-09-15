using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Slime
{
    public struct Particle 
    {
        public float3 Position;
        public int ID;
    }
    
    public struct ParticleController
    {
        public float3 Center;
        public float Radius;
        public float3 Velocity;
        public float Concentration;
    }
    
    public struct MyBoxCollider
    {
        public float3 Center;
        public float3 Extent;
    }

    public static class PBF_Utils
    {
        public const int Width = 16;
        public const int Num = Width * Width * Width / 2;
        public const int BubblesCount = 2048;
        public const float PredictStep = 0.02f;
        public const float DeltaTime = 0.02f;
        public const float TargetDensity = 1.5f;
        public const int GridSize = 4 * 4 * 4;
        public const int GridNum = 768;

        public const float h = 1.0f;
        public const float h2 = h * h;
        public const float CellSize = 0.5f * h;
        public const float Mass = 1.0f;
        public const float Scale = 0.1f;
        public const float InvScale = 10f;
    
        public struct Int2Comparer : IComparer<int2>
        {
            public int Compare(int2 lhs, int2 rhs) => lhs.x - rhs.x;
        }
        public struct BlockComparer : IComparer<int4>
        {
            public int Compare(int4 lhs, int4 rhs) => lhs.w - rhs.w;
        }

        public static int GetKey(int3 coord)
        {
            unchecked
            {
                int key = coord.x & 1023;
                key = (key << 10) | (coord.y & 1023);
                key = (key << 10) | (coord.z & 1023);
                return key;
            }
        }

        public static int3 GetCoord(float3 pos)
        {
            return (int3)math.floor(pos / h);
        }
    
        private const float KernelPoly6 = 315 / (64 * math.PI * h2 * h2 * h2 * h2 * h);

        public static float SmoothingKernelPoly6(float r2)
        {
            if (r2 < h2)
            {
                float v = h2 - r2;
                return v * v * v * KernelPoly6;
            }
            return 0;
        }
    
        public static float SmoothingKernelPoly6(float dst, float radius)
        {
            if (dst < radius)
            {
                float scale = 315 / (64 * math.PI * math.pow(radius, 9));
                float v = radius * radius - dst * dst;
                return v * v * v * scale;
            }
            return 0;
        }
    
        private const float Spiky3 = 15 / (h2*h2*h2 * math.PI);
        public static float DerivativeSpikyPow3(float r)
        {
            if (r <= h)
            {
                float v = h - r;
                return -v * v * 3 * Spiky3;
            }
            return 0;
        }
        public static float DerivativeSpikyPow3(float dst, float radius)
        {
            if (dst <= radius)
            {
                float scale = 45 / (math.pow(radius, 6) * math.PI);
                float v = radius - dst;
                return -v * v * scale;
            }
            return 0;
        }
        public static float SpikyKernelPow3(float r)
        {
            if (r < h)
            {
                float v = h - r;
                return v * v * v * Spiky3;
            }
            return 0;
        }
        private static float SpikyKernelPow3(float dst, float radius)
        {
            if (dst < radius)
            {
                float scale = 15 / (math.PI * math.pow(radius, 6));
                float v = radius - dst;
                return v * v * v * scale;
            }
            return 0;
        }
    }

    public static class Simulation_PBF
    {
        [BurstCompile]
        public struct HashJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Particle> Ps;
            [WriteOnly] public NativeArray<int2> Hashes;

            public void Execute(int i)
            {
                int3 gridPos = PBF_Utils.GetCoord(Ps[i].Position);
                int hash = PBF_Utils.GetKey(gridPos);
                Hashes[i] = math.int2(hash, i);
            }
        }

        [BurstCompile]
        public struct BuildLutJob : IJob
        {
            [ReadOnly] public NativeArray<int2> Hashes;
            public NativeHashMap<int, int2> Lut;

            public void Execute()
            {
                int currentKey = Hashes[0].x;
                int start = 0;
                for (int i = 1; i < Hashes.Length; ++i)
                {
                    if (Hashes[i].x == currentKey) continue;
                    Lut.TryAdd(currentKey, new int2(start, i));
                    currentKey = Hashes[i].x;
                    start = i;
                }

                Lut.TryAdd(currentKey, new int2(start, Hashes.Length));
            }
        }

        [BurstCompile]
        public struct ShuffleJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> Hashes;
            [ReadOnly] public NativeArray<Particle> PsRaw;
            [ReadOnly] public NativeArray<Particle> PsNew;
            [ReadOnly] public NativeArray<float3> Velocity;

            [WriteOnly] public NativeArray<float3> PosOld;
            [WriteOnly] public NativeArray<float3> PosPredict;
            [WriteOnly] public NativeArray<float3> VelocityOut;

            public void Execute(int i)
            {
                int id = Hashes[i].y;
                PosPredict[i] = PsNew[id].Position;
                PosOld[i] = PsRaw[id].Position;
                VelocityOut[i] = Velocity[id];
            }
        }

        [BurstCompile]
        public struct ApplyForceJob : IJobParallelFor
        {
            [ReadOnly] public NativeList<ParticleController> Controllers;
            [ReadOnly] public NativeArray<Particle> Ps;
            [WriteOnly] public NativeArray<Particle> PsNew;
            public NativeArray<float3> Velocity;
            public float3 Gravity;

            public void Execute(int i)
            {
                Particle p = Ps[i];

                var velocity = Velocity[i] * 0.99f + Gravity * PBF_Utils.DeltaTime;
                if (p.ID >= 0 && p.ID < Controllers.Length)
                {
                    ParticleController cl = Controllers[p.ID];
                    // float3 toCenter = cl.Center + new float3(0, cl.Radius * 0.3f, 0) - p.Position;
                    float3 toCenter = cl.Center + new float3(0, cl.Radius * 0.05f, 0) - p.Position;
                    float len = math.length(toCenter);

                    if (len < cl.Radius)
                    {
                        // velocity += PBF_Utils.DeltaTime * 0.5f * math.max(0, 1 - len*len*0.01f) * MoveDir;
                        velocity = math.lerp(cl.Velocity, velocity, math.lerp(1, len * 0.1f, cl.Concentration * 0.002f));
                        velocity += cl.Concentration * PBF_Utils.DeltaTime * math.min(1, len) *
                                    math.normalizesafe(toCenter);
                    }
                }

                p.Position += velocity * PBF_Utils.PredictStep;
                PsNew[i] = p;
                Velocity[i] = velocity;
            }
        }

        [BurstCompile]
        public struct ComputeLambdaJob : IJobParallelFor
        {
            [ReadOnly] public NativeHashMap<int, int2> Lut;
            [ReadOnly] public NativeArray<float3> PosPredict;
            [WriteOnly] public NativeArray<float> Lambda;

            public void Execute(int i)
            {
                float3 pos = PosPredict[i];
                int3 coord = PBF_Utils.GetCoord(pos);
                float rho = 0.0f;
                float3 grad_i = float3.zero;
                float sigmaGrad = 0.0f;
                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int key = PBF_Utils.GetKey(coord + math.int3(dx, dy, dz));
                    if (!Lut.ContainsKey(key)) continue;
                    int2 range = Lut[key];
                    for (int j = range.x; j < range.y; j++)
                    {
                        if (i == j)
                            continue;

                        float3 dir = pos - PosPredict[j];
                        float r2 = math.lengthsq(dir);
                        if (r2 > PBF_Utils.h2) continue;

                        float r = math.sqrt(r2);
                        rho += PBF_Utils.SmoothingKernelPoly6(r2) / PBF_Utils.TargetDensity;
                        float3 grad_j = PBF_Utils.DerivativeSpikyPow3(r) / PBF_Utils.TargetDensity * math.normalize(dir);
                        sigmaGrad += math.lengthsq(grad_j);
                        grad_i += grad_j;
                    }
                }

                sigmaGrad += math.dot(grad_i, grad_i);
                float c = math.max(-0.2f, rho / PBF_Utils.TargetDensity - 1.0f);
                Lambda[i] = -c / (sigmaGrad + 1e-5f);
            }
        }

        [BurstCompile]
        public struct ComputeDeltaPosJob : IJobParallelFor
        {
            [ReadOnly] public NativeHashMap<int, int2> Lut;
            [ReadOnly] public NativeArray<float3> PosPredict;
            [ReadOnly] public NativeArray<float> Lambda;
            [WriteOnly] public NativeArray<Particle> PsNew;
            private const float TensileDq = 0.25f * PBF_Utils.h;
            private const float TensileK = 0.1f;
            // private const int TensileN = 4;

            public void Execute(int i)
            {
                float3 position = PosPredict[i];
                float3 dp = float3.zero;
                float W_dp = PBF_Utils.SmoothingKernelPoly6(TensileDq * TensileDq);

                float lambda = Lambda[i];
                int3 coord = PBF_Utils.GetCoord(position);
                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int key = PBF_Utils.GetKey(coord + math.int3(dx, dy, dz));
                    if (!Lut.ContainsKey(key)) continue;
                    int2 range = Lut[key];
                    for (int j = range.x; j < range.y; j++)
                    {
                        if (i == j)
                            continue;

                        float3 dir = position - PosPredict[j];
                        float r2 = math.dot(dir, dir);
                        if (r2 >= PBF_Utils.h2) continue;

                        float r = math.sqrt(r2);
                        float3 w_spiky = PBF_Utils.SpikyKernelPow3(r) * math.normalize(dir);
                        float corr = PBF_Utils.SmoothingKernelPoly6(r2) / W_dp;
                        float s_corr = -TensileK * corr * corr * corr * corr;
                        dp += (lambda + Lambda[j] + s_corr) * w_spiky;
                    }
                }

                dp /= PBF_Utils.TargetDensity;

                PsNew[i] = new Particle
                {
                    Position = position - dp,
                    ID = 0,
                };
            }
        }

        // [BurstCompile]
        public struct UpdateJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<MyBoxCollider> Colliders;
            [ReadOnly] public NativeArray<float3> PosOld;
            [WriteOnly] public NativeArray<float3> Velocity;
            public NativeArray<Particle> Ps;

            public void Execute(int i)
            {
                Particle p = Ps[i];

                // float3 relativePos = p.Position - BoundsCenter;
                // float3 toBoundary = math.abs(relativePos);
                // float3 sign = math.sign(relativePos);
                // relativePos = math.select(relativePos, sign * BoundsExtent, toBoundary > BoundsExtent);
                // p.Position = relativePos + BoundsCenter;
                p.Position.y = math.max(1f, p.Position.y);
                foreach (var box in Colliders)
                {
                    float3 dir = p.Position - box.Center;
                    float3 vec = math.abs(dir);
                    if (math.all(vec < box.Extent))
                    {
                        float3 remain = box.Extent - vec;
                        bool3 pushAxis = new bool3(false, false, false);
                        int axis = 0;
                        if (remain.y < remain[axis]) axis = 1;
                        if (remain.z < remain[axis]) axis = 2;
                        pushAxis[axis] = true;
                        p.Position = math.select(p.Position, box.Center + math.sign(dir) * box.Extent, pushAxis);
                    }
                }
                float3 vel = (p.Position - PosOld[i]) / PBF_Utils.DeltaTime;
                Velocity[i] = math.min(30, math.length(vel)) * math.normalizesafe(vel);
                Ps[i] = p;
            }
        }

        [BurstCompile]
        public struct ApplyViscosityJob : IJobParallelFor
        {
            [ReadOnly] public NativeHashMap<int, int2> Lut;
            [ReadOnly] public NativeArray<float3> PosPredict;
            [ReadOnly] public NativeArray<float3> VelocityR;
            [WriteOnly] public NativeArray<float3> VelocityW;
            public float ViscosityStrength;

            public void Execute(int i)
            {
                float3 pos = PosPredict[i];
                int3 coord = PBF_Utils.GetCoord(pos);
                float3 viscosityForce = float3.zero;
                float3 vel = VelocityR[i];
                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int key = PBF_Utils.GetKey(coord + math.int3(dx, dy, dz));
                    if (!Lut.ContainsKey(key)) continue;
                    int2 range = Lut[key];
                    for (int j = range.x; j < range.y; j++)
                    {
                        if (i == j)
                            continue;

                        float3 dir = pos - PosPredict[j];
                        float r2 = math.lengthsq(dir);
                        if (r2 > PBF_Utils.h2) continue;

                        viscosityForce += (VelocityR[j] - vel) * PBF_Utils.SmoothingKernelPoly6(r2);
                    }
                }

                VelocityW[i] = vel + viscosityForce / PBF_Utils.TargetDensity * ViscosityStrength * PBF_Utils.DeltaTime;
            }
        }
    }
}