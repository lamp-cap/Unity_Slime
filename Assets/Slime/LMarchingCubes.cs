#if UNITY_EDITOR
using System.Linq;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Slime
{
    public class LMarchingCubes : System.IDisposable
    {
        private NativeArray<int> _triangleConnectionTable;
        private NativeArray<int> _triangleVertCountTable;
        private NativeArray<int> _cubeEdgeFlags;
        private NativeArray<int2> _edgeVert;
        private NativeArray<int3> _vertPos;
        private Mesh _mesh;
        private int _lastVertexCount;

        public LMarchingCubes()
        {
            _triangleConnectionTable = new NativeArray<int>(MarchingCubesTables.TriangleConnectionTable.SelectMany(x => x).ToArray(), Allocator.Persistent);
            _triangleVertCountTable = new NativeArray<int>(MarchingCubesTables.TriangleVertCountTable, Allocator.Persistent);
            _cubeEdgeFlags = new NativeArray<int>(MarchingCubesTables.CubeEdgeFlags, Allocator.Persistent);
            _edgeVert = new NativeArray<int2>(MarchingCubesTables.EdgeVert.Select(x => new int2(x[0], x[1])).ToArray(), Allocator.Persistent);
            _vertPos = new NativeArray<int3>(MarchingCubesTables.VertPos, Allocator.Persistent);
            _mesh = new Mesh();
            _mesh.indexFormat = IndexFormat.UInt32;
        }

        public void Dispose()
        {
            _triangleConnectionTable.Dispose();
            _triangleVertCountTable.Dispose();
            _cubeEdgeFlags.Dispose();
            _edgeVert.Dispose();
            _vertPos.Dispose();
            Object.Destroy(_mesh);
        }
        
        public Mesh MarchingCubes(NativeArray<int3> keys, NativeHashMap<int3, int> gridLut, NativeArray<float> grid, float threshold, float scale = 1)
        {
            var vertices = new NativeList<float3>(_lastVertexCount + 512, Allocator.TempJob);
            var normals = new NativeList<float3>(_lastVertexCount + 512, Allocator.TempJob);
            var triangles = new NativeList<int>(_lastVertexCount + 512, Allocator.TempJob);
            new MarchingCubesJobs
            {
                TriangleConnectionTable = _triangleConnectionTable,
                EdgeVert = _edgeVert,
                VertPos = _vertPos,
                Keys = keys,
                GridLut = gridLut,
                Grid = grid,
                Vertices = vertices,
                Normals = normals,
                Triangles = triangles,
                Threshold = threshold,
                Scale = scale,
            }.Schedule().Complete();
            
            var mesh = _mesh;
            if (vertices.Length > _lastVertexCount)
            {
                mesh.SetVertices(vertices.AsArray());
                mesh.SetNormals(normals.AsArray());
                mesh.SetIndices(triangles.AsArray(), MeshTopology.Triangles, 0);
            }
            else
            {
                mesh.SetIndices(triangles.AsArray(), MeshTopology.Triangles, 0);
                mesh.SetVertices(vertices.AsArray());
                mesh.SetNormals(normals.AsArray());
            }
            mesh.RecalculateBounds();
            mesh.UploadMeshData(false);
            
            _lastVertexCount = vertices.Length;
            vertices.Dispose();
            normals.Dispose();
            triangles.Dispose();
            
            return mesh;
        }

        public Mesh MarchingCubesParallel(NativeArray<int3> keys, NativeHashMap<int3, int> gridLut,
            NativeArray<float> grid, float threshold, float scale = 1)
        {
            var blockVertCount = new NativeArray<int>(keys.Length + 1, Allocator.TempJob);
            var handle = new TrianglesCounterJobs
            {
                TriangleVertCountTable = _triangleVertCountTable,
                VertPos = _vertPos,
                Keys = keys,
                GridLut = gridLut,
                Grid = grid,
                BlockVertCount = blockVertCount,
                Threshold = threshold
            }.Schedule(keys.Length, 32);

            new PrefixSum()
            {
                BlockVertCount = blockVertCount
            }.Schedule(handle).Complete();

            var totalVertCount = blockVertCount[keys.Length];
            var vertices = new NativeArray<float3>(totalVertCount, Allocator.TempJob);
            var normals = new NativeArray<float3>(totalVertCount, Allocator.TempJob);
            var triangles = new NativeArray<int>(totalVertCount, Allocator.TempJob);
            
            new MarchingCubesParallelJobs
            {
                TriangleConnectionTable = _triangleConnectionTable,
                EdgeVert = _edgeVert,
                VertPos = _vertPos,
                Keys = keys,
                GridLut = gridLut,
                Grid = grid,
                BlockVertPrefixSum = blockVertCount,
                Vertices = vertices,
                Normals = normals,
                Triangles = triangles,
                Threshold = threshold,
                Scale = scale,
            }.Schedule(keys.Length, 32).Complete();
            
            var mesh = _mesh;
            if (vertices.Length > _lastVertexCount)
            {
                mesh.SetVertices(vertices);
                mesh.SetNormals(normals);
                mesh.SetIndices(triangles, MeshTopology.Triangles, 0);
            }
            else
            {
                mesh.SetIndices(triangles, MeshTopology.Triangles, 0);
                mesh.SetVertices(vertices);
                mesh.SetNormals(normals);
            }
            mesh.RecalculateBounds();
            mesh.UploadMeshData(false);
            
            _lastVertexCount = vertices.Length;
            vertices.Dispose();
            normals.Dispose();
            triangles.Dispose();
            blockVertCount.Dispose();
            
            return mesh;
        }

        [BurstCompile]
        private unsafe struct TrianglesCounterJobs : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int> TriangleVertCountTable;
            [ReadOnly] public NativeArray<int3> VertPos;
            
            [ReadOnly] public NativeArray<int3> Keys;
            [ReadOnly] public NativeHashMap<int3, int> GridLut;
            [ReadOnly] public NativeArray<float> Grid;
            
            [WriteOnly] public NativeArray<int> BlockVertCount;
            
            public float Threshold;

            public void Execute(int index)
            {
                float* block = stackalloc float[8 * 8 * 8];

                int3 key = Keys[index];
                
                for (int z = 0; z < 8; z++)
                for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    block[GetBlockIndex(new int3(x, y, z))] = 0;

                int3 minCoord = (key << 2) - 2;
                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int3 nKey = key + new int3(dx, dy, dz);
                    if (!GridLut.ContainsKey(nKey)) continue;

                    int nOff = GridLut[nKey];
                    for (int j = 0; j < 64; j++)
                    {
                        int3 coord = (nKey * 4) + GetLocalCoord(j) - minCoord;
                        if (math.any(coord < 0) || math.any(coord >= 8))
                            continue;
                        block[GetBlockIndex(coord)] = Grid[nOff + j];
                    }
                }

                int vertCount = 0;
                for (int z = 1; z < 6; z++)
                for (int y = 1; y < 6; y++)
                for (int x = 1; x < 6; x++)
                {
                    var result = 0;
                    int3 localCoord = new int3(x, y, z);
                    int3 coord = localCoord + minCoord;
                    int3 nKey = coord >> 2;
                    if ((math.any(localCoord < 2) || math.any(localCoord > 5)) && GridLut.ContainsKey(nKey))
                        continue;

                    for (int i = 0; i < 8; i++)
                    {
                        if (ReadGrid(localCoord + VertPos[i], block) > Threshold)
                            result |= 1 << i;
                    }
                    
                    vertCount += TriangleVertCountTable[result];
                }
                
                BlockVertCount[index] = vertCount;
            }
            
            private static float ReadGrid(int3 coord, float* block)
            {
                coord = math.clamp(coord, 0, 7);
                return block[GetBlockIndex(coord)];
            }

            private static int3 GetLocalCoord(int index)
            {
                return new int3(index & 3, (index >> 2) & 3, (index >> 4) & 3);
            }
            
            private static int GetBlockIndex(int3 coord)
            {
                return coord.x + 8 * (coord.y + 8 * coord.z);
            }
        }

        [BurstCompile]
        private struct PrefixSum : IJob
        {
            public NativeArray<int> BlockVertCount;
            
            public void Execute()
            {
                int sum = 0;
                for (int i = 0; i < BlockVertCount.Length; i++)
                {
                    var count = BlockVertCount[i];
                    BlockVertCount[i] = sum;
                    sum += count;
                }
            }
        }
        
        [BurstCompile]
        private unsafe struct MarchingCubesJobs : IJob
        {
            [ReadOnly] public NativeArray<int> TriangleConnectionTable;
            [ReadOnly] public NativeArray<int2> EdgeVert;
            [ReadOnly] public NativeArray<int3> VertPos;
            
            [ReadOnly] public NativeArray<int3> Keys;
            [ReadOnly] public NativeHashMap<int3, int> GridLut;
            [ReadOnly] public NativeArray<float> Grid;
            
            public NativeList<float3> Vertices;
            public NativeList<float3> Normals;
            public NativeList<int> Triangles;
            
            public float Threshold;
            public float Scale;

            public void Execute()
            {
                float* weights = stackalloc float[8];
                int* indices = stackalloc int[3];
                float* block = stackalloc float[8 * 8 * 8];

                foreach (var key in Keys)
                {
                    UnsafeUtility.MemClear(block, 8 * 8 * 8 * sizeof(float));

                    int3 minCoord = (key << 2) - 2;
                    for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int3 nKey = key + new int3(dx, dy, dz);
                        if (!GridLut.ContainsKey(nKey)) continue;

                        int nOff = GridLut[nKey];
                        for (int j = 0; j < 64; j++)
                        {
                            int3 coord = (nKey * 4) + GetLocalCoord(j) - minCoord;
                            if (math.any(coord < 0) || math.any(coord >= 8))
                                continue;

                            block[GetBlockIndex(coord)] = Grid[nOff + j];
                        }
                    }

                    for (int z = 1; z < 6; z++)
                    for (int y = 1; y < 6; y++)
                    for (int x = 1; x < 6; x++)
                    {
                        var result = 0;
                        int3 localCoord = new int3(x, y, z);
                        int3 coord = localCoord + minCoord;
                        int3 nKey = coord >> 2;
                        if ((math.any(localCoord < 2) || math.any(localCoord > 5)) && GridLut.ContainsKey(nKey))
                            continue;

                        for (int i = 0; i < 8; i++)
                        {
                            weights[i] = ReadGrid(localCoord + VertPos[i], block);
                            if (weights[i] > Threshold) result |= 1 << i;
                        }

                        if (result < 1) continue;

                        var line = new NativeSlice<int>(TriangleConnectionTable, result * 16, 16);
                        if (line[0] < 0) continue;
                        for (int i = 0; line[i] > -1 && i < 15; i += 3)
                        {
                            indices[0] = line[i];
                            indices[1] = line[i + 2];
                            indices[2] = line[i + 1];

                            for (int j = 0; j < 3; j++)
                            {
                                int2 ev = EdgeVert[indices[j]];
                                int3 v0 = VertPos[ev.x] + coord;
                                int3 v1 = VertPos[ev.y] + coord;
                                float weight = (Threshold - weights[ev.x]) / (weights[ev.y] - weights[ev.x]);
                                int id = Vertices.Length;
                                float3 pos = math.lerp(v0, v1, weight);
                                float3 normal = CalcNormal(pos - minCoord, block);
                                Vertices.Add(pos * Scale);
                                Normals.Add(normal);
                                Triangles.Add(id);
                            }
                        }
                    }
                }
            }
            
            private static float ReadGrid(int3 coord, float* block)
            {
                coord = math.clamp(coord, 0, 7);
                return block[GetBlockIndex(coord)];
            }

            private static int3 GetLocalCoord(int index)
            {
                return new int3(index & 3, (index >> 2) & 3, (index >> 4) & 3);
            }
            
            private static int GetBlockIndex(int3 coord)
            {
                return coord.x + 8 * (coord.y + 8 * coord.z);
            }

            private float3 CalcNormal(float3 pos, float* block)
            {
                float3 offset = new float3(1f, 0, 0);
                float nx = ReadGridBilinear(pos - offset.xyz, block) - ReadGridBilinear(pos + offset.xyz, block);
                float ny = ReadGridBilinear(pos - offset.yxz, block) - ReadGridBilinear(pos + offset.yxz, block);
                float nz = ReadGridBilinear(pos - offset.yzx, block) - ReadGridBilinear(pos + offset.yzx, block);
                return math.normalize(new float3(nx, ny, nz));
            }

            private float ReadGridBilinear(float3 uvw, float* block)
            {
                int3 p000 = (int3)math.floor(uvw);
                int3 p111 = p000 + 1;
                float3 f = uvw - p000;
                float c000 = ReadGrid(p000, block);
                float c100 = ReadGrid(new int3(p111.x, p000.y, p000.z), block);
                float c010 = ReadGrid(new int3(p000.x, p111.y, p000.z), block);
                float c110 = ReadGrid(new int3(p111.x, p111.y, p000.z), block);
                float c001 = ReadGrid(new int3(p000.x, p000.y, p111.z), block);
                float c101 = ReadGrid(new int3(p111.x, p000.y, p111.z), block);
                float c011 = ReadGrid(new int3(p000.x, p111.y, p111.z), block);
                float c111 = ReadGrid(p111, block);
                float c00 = math.lerp(c000, c100, f.x);
                float c10 = math.lerp(c010, c110, f.x);
                float c01 = math.lerp(c001, c101, f.x);
                float c11 = math.lerp(c011, c111, f.x);
                float c0 = math.lerp(c00, c10, f.y);
                float c1 = math.lerp(c01, c11, f.y);
                return math.lerp(c0, c1, f.z);
            }
        }
        
        [BurstCompile]
        private unsafe struct MarchingCubesParallelJobs : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int> TriangleConnectionTable;
            [ReadOnly] public NativeArray<int2> EdgeVert;
            [ReadOnly] public NativeArray<int3> VertPos;
            
            [ReadOnly] public NativeArray<int3> Keys;
            [ReadOnly] public NativeHashMap<int3, int> GridLut;
            [ReadOnly] public NativeArray<float> Grid;
            [ReadOnly] public NativeArray<int> BlockVertPrefixSum;
            
            [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float3> Vertices;
            [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float3> Normals;
            [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<int> Triangles;
            
            public float Threshold;
            public float Scale;

            public void Execute(int index)
            {
                var key = Keys[index];
                    
                float* weights = stackalloc float[8];
                int* indices = stackalloc int[3];
                float* block = stackalloc float[8 * 8 * 8];
                
                UnsafeUtility.MemClear(block, 8 * 8 * 8 * sizeof(float));

                int3 minCoord = (key << 2) - 2;
                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int3 nKey = key + new int3(dx, dy, dz);
                    if (!GridLut.ContainsKey(nKey)) continue;

                    int nOff = GridLut[nKey];
                    for (int j = 0; j < 64; j++)
                    {
                        int3 coord = (nKey * 4) + GetLocalCoord(j) - minCoord;
                        if (math.any(coord < 0) || math.any(coord >= 8))
                            continue;

                        block[GetBlockIndex(coord)] = Grid[nOff + j];
                    }
                }

                int offset = BlockVertPrefixSum[index];
                for (int z = 1; z < 6; z++)
                for (int y = 1; y < 6; y++)
                for (int x = 1; x < 6; x++)
                {
                    var result = 0;
                    int3 localCoord = new int3(x, y, z);
                    int3 coord = localCoord + minCoord;
                    int3 nKey = coord >> 2;
                    if ((math.any(localCoord < 2) || math.any(localCoord > 5)) && GridLut.ContainsKey(nKey))
                        continue;

                    for (int i = 0; i < 8; i++)
                    {
                        weights[i] = ReadGrid(localCoord + VertPos[i], block);
                        if (weights[i] > Threshold) result |= 1 << i;
                    }
                    
                    var line = TriangleConnectionTable.Slice(result * 16, 16);
                    
                    if (line[0] < 0) continue;
                    
                    for (int i = 0; line[i] > -1 && i < 15; i += 3)
                    {
                        indices[0] = line[i];
                        indices[1] = line[i + 2];
                        indices[2] = line[i + 1];

                        for (int j = 0; j < 3; j++)
                        {
                            int2 ev = EdgeVert[indices[j]];
                            int3 v0 = VertPos[ev.x] + coord;
                            int3 v1 = VertPos[ev.y] + coord;
                            float weight = (Threshold - weights[ev.x]) / (weights[ev.y] - weights[ev.x]);
                            float3 pos = math.lerp(v0, v1, weight);
                            float3 normal = CalcNormal(pos - minCoord, block);
                            Vertices[offset] = (pos * Scale);
                            Normals[offset] = (normal);
                            Triangles[offset] = offset;
                            offset++;
                        }
                    }
                }
            }
            
            private static float ReadGrid(int3 coord, float* block)
            {
                coord = math.clamp(coord, 0, 7);
                return block[GetBlockIndex(coord)];
            }

            private static int3 GetLocalCoord(int index)
            {
                return new int3(index & 3, (index >> 2) & 3, (index >> 4) & 3);
            }
            
            private static int GetBlockIndex(int3 coord)
            {
                return coord.x + 8 * (coord.y + 8 * coord.z);
            }

            private float3 CalcNormal(float3 pos, float* block)
            {
                float3 offset = new float3(1f, 0, 0);
                float nx = ReadGridBilinear(pos - offset.xyz, block) - ReadGridBilinear(pos + offset.xyz, block);
                float ny = ReadGridBilinear(pos - offset.yxz, block) - ReadGridBilinear(pos + offset.yxz, block);
                float nz = ReadGridBilinear(pos - offset.yzx, block) - ReadGridBilinear(pos + offset.yzx, block);
                return math.normalize(new float3(nx, ny, nz));
            }

            private float ReadGridBilinear(float3 uvw, float* block)
            {
                int3 p000 = (int3)math.floor(uvw);
                int3 p111 = p000 + 1;
                float3 f = uvw - p000;
                float c000 = ReadGrid(p000, block);
                float c100 = ReadGrid(new int3(p111.x, p000.y, p000.z), block);
                float c010 = ReadGrid(new int3(p000.x, p111.y, p000.z), block);
                float c110 = ReadGrid(new int3(p111.x, p111.y, p000.z), block);
                float c001 = ReadGrid(new int3(p000.x, p000.y, p111.z), block);
                float c101 = ReadGrid(new int3(p111.x, p000.y, p111.z), block);
                float c011 = ReadGrid(new int3(p000.x, p111.y, p111.z), block);
                float c111 = ReadGrid(p111, block);
                float c00 = math.lerp(c000, c100, f.x);
                float c10 = math.lerp(c010, c110, f.x);
                float c01 = math.lerp(c001, c101, f.x);
                float c11 = math.lerp(c011, c111, f.x);
                float c0 = math.lerp(c00, c10, f.y);
                float c1 = math.lerp(c01, c11, f.y);
                return math.lerp(c0, c1, f.z);
            }
        }

    }
    public static class MarchingCubesTables
    {
        public static readonly float3[] EdgePos = new[]
        {
            new float3(0.5f, 0, 0),
            new float3(1, 0.5f, 0),
            new float3(0.5f, 1, 0),
            new float3(0, 0.5f , 0),
            new float3(0.5f, 0, 1),
            new float3(1, 0.5f, 1),
            new float3(0.5f, 1, 1),
            new float3(0, 0.5f, 1),
            new float3(0, 0, 0.5f),
            new float3(1, 0, 0.5f),
            new float3(1, 1, 0.5f),
            new float3(0, 1, 0.5f)
        };
        public static readonly int[][] EdgeVert = new[]
        {
            new [] {0, 1},
            new [] {1, 2},
            new [] {2, 3},
            new [] {0, 3},
            new [] {4, 5},
            new [] {5, 6},
            new [] {6, 7},
            new [] {4, 7},
            new [] {0, 4},
            new [] {1, 5},
            new [] {2, 6},
            new [] {3, 7},
        };
        public static readonly int3[] VertPos = new[]
        {
            new int3(0, 0, 0),
            new int3(1, 0, 0),
            new int3(1, 1, 0),
            new int3(0, 1 , 0),
            new int3(0, 0, 1),
            new int3(1, 0, 1),
            new int3(1, 1, 1),
            new int3(0, 1, 1),
        };
        // For any edge, if one vertex is inside of the surface and the other is outside of the surface
        // then the edge intersects the surface
        // For each of the 8 vertices of the cube can be two possible states : either inside or outside of the surface
        // For any cube the are 2^8=256 possible sets of vertex states
        // This table lists the edges intersected by the surface for all 256 possible vertex states
        // There are 12 edges.  For each entry in the table, if edge #n is intersected, then bit #n is set to 1
        // cubeEdgeFlags[256]
        public static readonly int[] CubeEdgeFlags = new int[]
        {
            0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
            0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
            0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
            0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
            0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
            0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
            0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
            0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
            0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
            0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
            0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
            0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
            0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
            0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
            0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
            0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
        };

        //  For each of the possible vertex states listed in cubeEdgeFlags there is a specific triangulation
        //  of the edge intersection points.  triangleConnectionTable lists all of them in the form of
        //  0-5 edge triples with the list terminated by the invalid value -1.
        //  For example: triangleConnectionTable[3] list the 2 triangles formed when corner[0] 
        //  and corner[1] are inside of the surface, but the rest of the cube is not.
        //  triangleConnectionTable[256][16]
        public static readonly int[][] TriangleConnectionTable = new int[][]
        {
            new [] {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
            new [] {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
            new [] {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
            new [] {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
            new [] {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
            new [] {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
            new [] {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
            new [] {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
            new [] {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
            new [] {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
            new [] {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
            new [] {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
            new [] {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
            new [] {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
            new [] {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
            new [] {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
            new [] {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
            new [] {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
            new [] {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
            new [] {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
            new [] {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
            new [] {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
            new [] {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
            new [] {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
            new [] {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
            new [] {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
            new [] {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
            new [] {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
            new [] {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
            new [] {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
            new [] {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
            new [] {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
            new [] {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
            new [] {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
            new [] {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
            new [] {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
            new [] {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
            new [] {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
            new [] {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
            new [] {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
            new [] {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
            new [] {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
            new [] {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
            new [] {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
            new [] {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
            new [] {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
            new [] {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
            new [] {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
            new [] {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
            new [] {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
            new [] {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
            new [] {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
            new [] {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
            new [] {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
            new [] {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
            new [] {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
            new [] {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
            new [] {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
            new [] {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
            new [] {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
            new [] {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
            new [] {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
            new [] {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
            new [] {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
            new [] {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
            new [] {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
            new [] {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
            new [] {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
            new [] {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
            new [] {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
            new [] {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
            new [] {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
            new [] {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
            new [] {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
            new [] {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
            new [] {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
            new [] {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
            new [] {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
            new [] {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
            new [] {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
            new [] {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
            new [] {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
            new [] {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
            new [] {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
            new [] {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
            new [] {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
            new [] {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
            new [] {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
            new [] {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
            new [] {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
            new [] {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
            new [] {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
            new [] {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
            new [] {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
            new [] {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
            new [] {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
            new [] {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
            new [] {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
            new [] {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
            new [] {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
            new [] {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
            new [] {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
            new [] {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
            new [] {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
            new [] {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
            new [] {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
            new [] {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
            new [] {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
            new [] {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
            new [] {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
            new [] {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
            new [] {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
            new [] {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
            new [] {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
            new [] {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
            new [] {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
            new [] {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
            new [] {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
            new [] {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
            new [] {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
            new [] {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
            new [] {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
            new [] {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
            new [] {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
            new [] {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
            new [] {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
            new [] {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
            new [] {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
            new [] {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
            new [] {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
            new [] {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
            new [] {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
            new [] {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
            new [] {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
            new [] {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
            new [] {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
            new [] {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            new [] {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
        };

        public static readonly int[] TriangleVertCountTable = new int[]
        {
            0, 3, 3, 6, 3, 6, 6, 9, 3, 6, 6, 9, 6, 9, 9, 6,
            3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
            3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
            6, 9, 9, 6, 9, 12, 12, 9, 9, 12, 12, 9, 12, 15, 15, 6,
            3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
            6, 9, 9, 12, 9, 12, 12, 15, 9, 12, 12, 15, 12, 15, 15, 12,
            6, 9, 9, 12, 9, 12, 6, 9, 9, 12, 12, 15, 12, 15, 9, 6,
            9, 12, 12, 9, 12, 15, 9, 6, 12, 15, 15, 12, 15, 6, 12, 3,
            3, 6, 6, 9, 6, 9, 9, 12, 6, 9, 9, 12, 9, 12, 12, 9,
            6, 9, 9, 12, 9, 12, 12, 15, 9, 6, 12, 9, 12, 9, 15, 6,
            6, 9, 9, 12, 9, 12, 12, 15, 9, 12, 12, 15, 12, 15, 15, 12,
            9, 12, 12, 9, 12, 15, 15, 12, 12, 9, 15, 6, 15, 12, 6, 3,
            6, 9, 9, 12, 9, 12, 12, 15, 9, 12, 12, 15, 6, 9, 9, 6,
            9, 12, 12, 15, 12, 15, 15, 6, 12, 9, 15, 12, 9, 6, 12, 3,
            9, 12, 12, 15, 12, 15, 9, 12, 12, 15, 15, 6, 9, 12, 6, 3,
            6, 9, 9, 6, 9, 12, 6, 3, 9, 6, 12, 3, 6, 3, 3, 0
        };
    }
}
#endif
