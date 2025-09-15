using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Slime
{
    public class Effects
    {
        public struct Component
        {
            public int ID;
            public float3 Center;
            public float3 BoundsMin;
            public float3 BoundsMax;
            public int CellCount;

            public Component(int id)
            {
                ID = id;
                Center = float3.zero;
                BoundsMin = new float3(float.MaxValue, float.MaxValue, float.MaxValue);
                BoundsMax = new float3(float.MinValue, float.MinValue, float.MinValue);
                CellCount = 0;
            }
        }

        public struct Bubble
        {
            public float3 Pos;
            public float Radius;
            public float3 Vel;
            public float LifeTime;
        }

        [BurstCompile]
        public struct ConnectComponentBlockJob : IJob
        {
            [ReadOnly] public NativeArray<int3> Keys;
            [ReadOnly] public NativeHashMap<int3, int> GridLut;
            [ReadOnly] public NativeArray<float> Grid;
            [WriteOnly] public NativeArray<int> GridID;
            public NativeList<Component> Components;
            public float Threshold;
            
            public void Execute()
            {
                var visited = new NativeArray<bool>(Grid.Length, Allocator.Temp);
                var stack = new NativeList<int3>(256, Allocator.Temp);
                int2 off = new int2(1, 0);

                for (int i = 0; i < Keys.Length; i++)
                {
                    int3 key = Keys[i];
                    for (int j = 0; j < 64; j++)
                    {
                        var coord = key * 4 + GetLocalCoord(j);
                        stack.Add(coord);

                        float3 posSum = float3.zero;
                        int id = Components.Length;
                        Component c = new Component(id);

                        while (stack.Length > 0)
                        {
                            int3 cur = stack[0];
                            stack.RemoveAtSwapBack(0);
                            if (!IsValid(cur, visited, out int index))
                                continue;
                            GridID[index] = id;
                            c.CellCount++;
                            float3 cellCenter = (float3)cur + 0.5f;
                            posSum += cellCenter;
                            c.BoundsMin = math.min(c.BoundsMin, cellCenter);
                            c.BoundsMax = math.max(c.BoundsMax, cellCenter);

                            stack.Add(cur + off.xyy);
                            stack.Add(cur - off.xyy);
                            stack.Add(cur + off.yxy);
                            stack.Add(cur - off.yxy);
                            stack.Add(cur + off.yyx);
                            stack.Add(cur - off.yyx);
                        }

                        if (c.CellCount > 0)
                        {
                            c.Center = (c.BoundsMin + c.BoundsMax) * 0.5f;
                            Components.Add(c);
                        }
                    }
                }

                stack.Dispose();
                visited.Dispose();
            }

            private bool IsValid(int3 coord, NativeArray<bool> visited, out int index)
            {
                int3 key = coord >> 2;
                index = 0;
                if (!GridLut.ContainsKey(key)) return false;
                int offset = GridLut[key];
                int localIndex = GetLocalIndex(coord & 3);
                index = localIndex + offset;
                return IsIdxValid(index, visited);
            }

            private bool IsIdxValid(int idx, NativeArray<bool> visited)
            {
                if (visited[idx]) return false;
                visited[idx] = true;
                if (Grid[idx] < Threshold) return false;
                return true;
            }

            private int3 GetLocalCoord(int i)
            {
                return new int3(i & 3, (i >> 2) & 3, (i >> 4) & 3);
            }

            private int GetLocalIndex(int3 coord)
            {
                return coord.x + 4 * (coord.y + 4 * coord.z);
            }
        }

        [BurstCompile]
        public struct ParticleIDJob : IJobParallelFor
        {
            [ReadOnly] public NativeHashMap<int3, int> GridLut;
            [ReadOnly] public NativeArray<int> GridID;
            public NativeArray<Particle> Particles;
            public float3 MinPos;

            public void Execute(int index)
            {
                Particle p = Particles[index];
                int3 coord = (int3)math.floor((p.Position - MinPos) / PBF_Utils.CellSize);
                int3 key = coord >> 2;
                if (GridLut.ContainsKey(key))
                {
                    int offset = GridLut[key];
                    p.ID = GridID[offset + GetLocalIndex(coord & 3)];
                }
                else
                {
                    p.ID = 0;
                }
                Particles[index] = p;
            }
            
            private int GetLocalIndex(int3 coord)
            {
                return coord.x + 4 * (coord.y + 4 * coord.z);
            }
        }
        
        public struct ConnectComponentArrayJob : IJob
        {
            [ReadOnly] public NativeArray<int3> Keys;
            [ReadOnly] public NativeHashMap<int3, int> GridLut;
            [ReadOnly] public NativeArray<float> Grid;
            public NativeList<Component> Components;
            public float Threshold;
            
            public void Execute()
            {
                var visited = new NativeArray<bool>(Grid.Length, Allocator.Temp);
                var stack = new NativeList<int3>(256, Allocator.Temp);
                int2 off = new int2(1, 0);

                for (int i = 0; i < Keys.Length; i++)
                {
                    int3 key = Keys[i];
                    for (int j = 0; j < 64; j++)
                    {
                        var coord = key * 4 + GetLocalCoord(j);
                        stack.Add(coord);

                        float3 posSum = float3.zero;
                        Component c = new Component(Components.Length);

                        while (stack.Length > 0)
                        {
                            int3 cur = stack[0];
                            stack.RemoveAtSwapBack(0);
                            if (!IsValid(cur, visited))
                                continue;
                            c.CellCount++;
                            float3 cellCenter = (float3)cur + 0.5f;
                            posSum += cellCenter;
                            c.BoundsMin = math.min(c.BoundsMin, cellCenter);
                            c.BoundsMax = math.max(c.BoundsMax, cellCenter);

                            stack.Add(cur + off.xyy);
                            stack.Add(cur - off.xyy);
                            stack.Add(cur + off.yxy);
                            stack.Add(cur - off.yxy);
                            stack.Add(cur + off.yyx);
                            stack.Add(cur - off.yyx);
                        }

                        if (c.CellCount > 0)
                        {
                            c.Center = posSum / c.CellCount;
                            Components.Add(c);
                        }
                    }
                }

                stack.Dispose();
                visited.Dispose();
            }

            private bool IsValid(int3 coord, NativeArray<bool> visited)
            {
                int3 key = coord >> 2;
                if (!GridLut.ContainsKey(key)) return false;
                int offset = GridLut[key];
                int localIndex = GetLocalIndex(coord & 3);
                return IsIdxValid(offset + localIndex, visited);
            }

            private bool IsIdxValid(int idx, NativeArray<bool> visited)
            {
                if (visited[idx]) return false;
                visited[idx] = true;
                if (Grid[idx] < Threshold) return false;
                return true;
            }

            private int3 GetLocalCoord(int i)
            {
                return new int3(i & 3, (i >> 2) & 3, (i >> 4) & 3);
            }

            private int GetLocalIndex(int3 coord)
            {
                return coord.x + 4 * (coord.y + 4 * coord.z);
            }
        }
        
        
        [BurstCompile]
        public struct RayInsectJob : IJob
        {
            [ReadOnly] public NativeHashMap<int3, int> GridLut;
            [ReadOnly] public NativeArray<float> Grid;
            public NativeArray<float3> Result;
            public float Threshold;
            public float3 Pos;
            public float3 Dir;
            public float3 MinPos;
            
            public void Execute()
            {
                float3 cur = Pos;
                float oldValue = 0;
                int3 oldCoord = int3.zero;
                for (int i = 0; i < 20; i++)
                {
                    int3 coord = (int3)math.floor((cur - MinPos) / PBF_Utils.CellSize);
                    if (math.all(coord == oldCoord)) continue;
                    oldCoord = coord;
                    int3 key = coord >> 2;
                    if (GridLut.ContainsKey(key))
                    {
                        int offset = GridLut[key];
                        float curValue = Grid[offset + GetLocalIndex(coord & 3)];
                        if (curValue < Threshold)
                        {
                            cur = math.lerp(cur - Dir * 0.5f, cur, (Threshold - oldValue) / (curValue - oldValue));
                            break;
                        }
                        oldValue = curValue;
                    }
                    else
                        break;
                    
                    cur += Dir * 0.5f;
                }
                Result[0] = cur;
            }

            private int GetLocalIndex(int3 coord)
            {
                return coord.x + 4 * (coord.y + 4 * coord.z);
            }
        }

        [BurstCompile]
        public struct GenerateBubblesJobs : IJob
        {
            [ReadOnly] public NativeArray<int4> Keys;
            [ReadOnly] public NativeHashMap<int3, int> GridLut;
            [ReadOnly] public NativeArray<float> Grid;

            [WriteOnly] public NativeArray<Bubble> BubblesBuffer;
            public NativeList<int> BubblesStack;
            public float Threshold;
            public float Speed;
            public int BlockCount;
            public float3 MinPos;
            public uint Seed;
            
            public void Execute()
            {
                var rnd = new Random(Seed);
                for (int i = 0; i < BlockCount; i++)
                {
                    int3 key = Keys[i].xyz;
                    int offset = GridLut[key];
                    for (int j = 0; j < 64; j++)
                    {
                        float data = Grid[offset + j];
                        if (data < Threshold) continue;

                        if (BubblesStack.Length < 1) 
                            return;
                        
                        var coord = key * 4 + GetLocalCoord(j);
                        
                        if (coord.y > 2 || rnd.NextFloat() > Speed) continue;
                        
                        int id = BubblesStack[0];
                        BubblesStack.RemoveAtSwapBack(0);
                        float radius = (rnd.NextFloat() * 0.7f + 0.3f) * PBF_Utils.CellSize;
                        Bubble bubble = new Bubble()
                        {
                            Pos = MinPos + PBF_Utils.CellSize * (coord + new float3(rnd.NextFloat(), rnd.NextFloat() - 1, rnd.NextFloat())),
                            Radius = radius,
                            Vel = new float3(0, radius * 2, 0),
                            LifeTime = 1,
                        };
                        
                        BubblesBuffer[id] = bubble;
                    }
                }
            }
            private int3 GetLocalCoord(int i)
            {
                return new int3(i & 3, (i >> 2) & 3, (i >> 4) & 3);
            }
        }

        [BurstCompile]
        public struct BubblesViscosityJob : IJobParallelFor
        {
            [ReadOnly] public NativeHashMap<int, int2> Lut;
            [ReadOnly] public NativeArray<Particle> Particles;
            [ReadOnly] public NativeArray<float3> VelocityR;
            [ReadOnly] public NativeList<ParticleController> Controllers;
            public NativeArray<Bubble> BubblesBuffer;
            public float ViscosityStrength;

            public void Execute(int i)
            {
                Bubble bubble = BubblesBuffer[i];
                if (bubble.LifeTime < 0) 
                    return;
                
                float3 vel = bubble.Vel;
                float3 pos = bubble.Pos;

                // for (int j = 0; j < Controllers.Length; j++)
                // {
                //     ParticleController cl = Controllers[j];
                //     float3 toCenter = cl.Center + new float3(0, cl.Radius * 0.05f, 0) - pos;
                //     float len = math.length(toCenter);
                //     if (len < cl.Radius)
                //     {
                //         vel += cl.Velocity * (cl.Concentration * 0.1f * PBF_Utils.DeltaTime);
                //         // vel += 0.5f * cl.Concentration * PBF_Utils.DeltaTime * math.min(1, len * 0.1f) * math.normalizesafe(toCenter);
                //     }
                // }
                
                int3 coord = PBF_Utils.GetCoord(pos);
                float3 viscosityVel = float3.zero;
                float rho = 0;
                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int key = PBF_Utils.GetKey(coord + math.int3(dx, dy, dz));
                    if (!Lut.ContainsKey(key)) continue;
                    int2 range = Lut[key];
                    for (int j = range.x; j < range.y; j++)
                    {
                        float3 dir = pos - Particles[j].Position;
                        float r2 = math.lengthsq(dir);
                        if (r2 > PBF_Utils.h2) continue;

                        float weight =  PBF_Utils.SmoothingKernelPoly6(r2);
                        viscosityVel += VelocityR[j] * weight;
                        rho += weight;
                    }
                }

                viscosityVel = rho > 1e-5f ? viscosityVel / rho : float3.zero;

                float y = math.min(4, pos.y) * 0.25f;
                bubble.Vel = math.lerp(vel, viscosityVel, new float3(y, ViscosityStrength, y));
                
                BubblesBuffer[i] = bubble;
            }
        }

        [BurstCompile]
        public struct UpdateBubblesJob : IJob
        {
            [ReadOnly] public NativeHashMap<int3, int> GridLut;
            [ReadOnly] public NativeArray<float> Grid;

            public NativeArray<Bubble> BubblesBuffer;
            public NativeList<int> BubblesStack;
            
            public float3 MinPos;
            public float Threshold;
            
            public void Execute()
            {
                for (int i = 0; i < BubblesBuffer.Length; i++)
                {
                    Bubble b = BubblesBuffer[i];
                    if (b.LifeTime < 0)
                        continue;
                    b.Vel.y += b.Radius * b.Radius * PBF_Utils.DeltaTime * 10;
                    b.Pos += b.Vel * PBF_Utils.DeltaTime * 2;
                    b.LifeTime += PBF_Utils.DeltaTime;
                    float scale = 1.002f;
                    b.Radius *= scale;

                    int3 coord = (int3)math.floor((b.Pos - MinPos) / PBF_Utils.CellSize);
                    int3 key = coord >> 2;
                    if (!GridLut.ContainsKey(key))
                    {
                        b.LifeTime = -1;
                        b.Radius = 0;
                        BubblesStack.Add(i);
                    }
                    else
                    {
                        int offset = GridLut[key];
                        float data = Grid[offset + GetLocalIndex(coord & 3)];
                        if (data < Threshold)
                        {
                            b.LifeTime = -1;
                            b.Radius = 0;
                            BubblesStack.Add(i);
                        }
                    }

                    BubblesBuffer[i] = b;
                }
            }
            
            private int GetLocalIndex(int3 coord)
            {
                return coord.x + 4 * (coord.y + 4 * coord.z);
            }
        }
    }
}
