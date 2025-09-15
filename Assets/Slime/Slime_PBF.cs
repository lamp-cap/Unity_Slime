using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;

namespace Slime
{
    public class Slime_PBF : MonoBehaviour
    {
        [System.Serializable]
        public enum RenderMode
        {
            Particles,
            Surface,
        }
        
        private struct SlimeInstance
        {
            public bool Active;
            public float3 Center;
            public Vector3 Pos;
            public Vector3 Dir;
            public float Radius;
            public int ControllerID;
        }
        

        [SerializeField, Range(0, 1)] private float bubbleSpeed = 0.2f;
        [SerializeField, Range(0, 100)] private float viscosityStrength = 1.0f;
        [SerializeField, Range(0.1f, 100)] private float concentration = 10f;
        [SerializeField, Range(-10, 10)] private float gravity = -5f;
        [SerializeField, Range(0, 5)] private float threshold = 1f;
        [SerializeField] private bool useAnisotropic = true;
        
        [SerializeField] private Mesh faceMesh;
        [SerializeField] private Material faceMat;
        [SerializeField] private Material mat;
        [SerializeField] private Mesh particleMesh;
        [SerializeField] private Material particleMat;
        [SerializeField] private Material bubblesMat;
        
        public Transform trans;
        public RenderMode renderMode = RenderMode.Surface;
        public int blockNum;
        public int bubblesNum;
        public float3 minPos;
        public float3 maxPos;

        public bool gridDebug;
        public bool componentDebug;

        #region Buffers
        
        private NativeArray<Particle> _particles;
        private NativeArray<Particle> _particlesTemp;
        private NativeArray<float3> _posPredict;
        private NativeArray<float3> _posOld;
        private NativeArray<float> _lambdaBuffer;
        private NativeArray<float3> _velocityBuffer;
        private NativeArray<float3> _velocityTempBuffer;
        private NativeHashMap<int, int2> _lut;
        private NativeArray<int2> _hashes;
        private NativeArray<float4x4> _covBuffer;
        private NativeArray<MyBoxCollider> _colliderBuffer;
        
        private NativeArray<float3> _boundsBuffer;
        private NativeArray<float> _gridBuffer;
        private NativeArray<float> _gridTempBuffer;
        private NativeHashMap<int3, int> _gridLut;
        private NativeArray<int4> _blockBuffer;
        private NativeArray<int> _blockColorBuffer;
        
        private NativeArray<Effects.Bubble> _bubblesBuffer;
        private NativeList<int> _bubblesPoolBuffer;
        
        private NativeList<Effects.Component> _componentsBuffer;
        private NativeArray<int> _gridIDBuffer;
        private NativeList<ParticleController> _controllerBuffer;
        private NativeList<ParticleController> _lastControllerBuffer;
        
        private ComputeBuffer _particlePosBuffer;
        private ComputeBuffer _particleCovBuffer;
        private ComputeBuffer _bubblesDataBuffer;
        
        #endregion
        
        private float3 _lastMousePos;
        private bool _mouseDown;
        private float3 _velocityY = float3.zero;
        private Bounds _bounds;
        private Vector3 _velocity = Vector3.zero;

        private LMarchingCubes _marchingCubes;
        private Mesh _mesh;
        
        private int batchCount = 64;
        private bool _connect;
        private NativeList<SlimeInstance> _slimeInstances;
        private int _controlledInstance;
        private Stack<int> _instancePool;

        void Awake()
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 60;
        }

        void Start()
        {
            _particles = new NativeArray<Particle>(PBF_Utils.Num, Allocator.Persistent);
            float half = PBF_Utils.Width / 2.0f;
            for (int i = 0; i < PBF_Utils.Width / 2; i++)
            for (int j = 0; j < PBF_Utils.Width; j++)
            for (int k = 0; k < PBF_Utils.Width; k++)
            {
                var idx = i * PBF_Utils.Width * PBF_Utils.Width + j * PBF_Utils.Width + k;
                _particles[idx] = new Particle
                {
                    Position = new float3(k - half, j, i - half) * 0.5f,
                    ID = 0,
                };
            }

            int particleNum = PBF_Utils.Num;
            _particlesTemp = new NativeArray<Particle>(particleNum, Allocator.Persistent);
            _posPredict = new NativeArray<float3>(particleNum, Allocator.Persistent);
            _posOld = new NativeArray<float3>(particleNum, Allocator.Persistent);
            _lambdaBuffer = new NativeArray<float>(particleNum, Allocator.Persistent);
            _velocityBuffer = new NativeArray<float3>(particleNum, Allocator.Persistent);
            _velocityTempBuffer = new NativeArray<float3>(particleNum, Allocator.Persistent);
            _boundsBuffer = new NativeArray<float3>(2, Allocator.Persistent);
            _gridBuffer = new NativeArray<float>(PBF_Utils.GridSize * PBF_Utils.GridNum, Allocator.Persistent);
            _gridTempBuffer = new NativeArray<float>(PBF_Utils.GridSize * PBF_Utils.GridNum, Allocator.Persistent);
            _gridLut = new NativeHashMap<int3, int>(PBF_Utils.GridNum, Allocator.Persistent);
            _covBuffer = new NativeArray<float4x4>(particleNum, Allocator.Persistent);
            _blockBuffer = new NativeArray<int4>(PBF_Utils.GridNum, Allocator.Persistent);
            _blockColorBuffer = new NativeArray<int>(9, Allocator.Persistent);
            
            _bubblesBuffer  = new NativeArray<Effects.Bubble>(PBF_Utils.BubblesCount, Allocator.Persistent);
            _bubblesPoolBuffer = new NativeList<int>(PBF_Utils.BubblesCount, Allocator.Persistent);
            for (int i = 0; i < PBF_Utils.BubblesCount; ++i)
            {
                _bubblesBuffer[i] = new Effects.Bubble()
                {
                    LifeTime = -1,
                };
                _bubblesPoolBuffer.Add(i);
            }

            _lut = new NativeHashMap<int, int2>(particleNum, Allocator.Persistent);
            _hashes = new NativeArray<int2>(particleNum, Allocator.Persistent);
            
            _componentsBuffer = new NativeList<Effects.Component>(16, Allocator.Persistent);
            _gridIDBuffer = new NativeArray<int>(PBF_Utils.GridSize * PBF_Utils.GridNum, Allocator.Persistent);
            _controllerBuffer = new NativeList<ParticleController>(16, Allocator.Persistent);
            _controllerBuffer.Add(new ParticleController
            {
                Center = float3.zero,
                Radius = PBF_Utils.InvScale,
                Velocity = float3.zero,
                Concentration = concentration,
            });
            _lastControllerBuffer = new NativeList<ParticleController>(16, Allocator.Persistent);

            _marchingCubes = new LMarchingCubes();

            _particlePosBuffer = new ComputeBuffer(particleNum, sizeof(float) * 4);
            _particleCovBuffer = new ComputeBuffer(particleNum, sizeof(float) * 16);
            _bubblesDataBuffer  = new ComputeBuffer(PBF_Utils.BubblesCount, sizeof(float) * 8);
            particleMat.SetBuffer("_ParticleBuffer", _particlePosBuffer);
            particleMat.SetBuffer("_CovarianceBuffer", _particleCovBuffer);
            bubblesMat.SetBuffer("_BubblesBuffer", _bubblesDataBuffer);

            _slimeInstances = new NativeList<SlimeInstance>(16,  Allocator.Persistent);
            _slimeInstances.Add(new SlimeInstance()
            {
                Center = Vector3.zero,
                Pos = Vector3.zero,
                Dir = Vector3.right,
                Radius = 1
            });
            _instancePool = new Stack<int>();
            var colliders = GetComponentsInChildren<BoxCollider>();
            _colliderBuffer = new NativeArray<MyBoxCollider>(colliders.Length, Allocator.Persistent);
            for (int i = 0; i < colliders.Length; ++i)
            {
                _colliderBuffer[i] = new MyBoxCollider()
                {
                    Center = colliders[i].bounds.center * PBF_Utils.InvScale,
                    Extent = colliders[i].bounds.extents * PBF_Utils.InvScale + Vector3.one,
                };
            }
        }

        private void OnDestroy()
        {
            if (_particles.IsCreated) _particles.Dispose();
            if (_particlesTemp.IsCreated) _particlesTemp.Dispose();
            if (_lut.IsCreated) _lut.Dispose();
            if (_hashes.IsCreated) _hashes.Dispose();
            if (_posPredict.IsCreated) _posPredict.Dispose();
            if (_posOld.IsCreated) _posOld.Dispose();
            if (_lambdaBuffer.IsCreated) _lambdaBuffer.Dispose();
            if (_velocityBuffer.IsCreated) _velocityBuffer.Dispose();
            if (_velocityTempBuffer.IsCreated) _velocityTempBuffer.Dispose();
            if (_boundsBuffer.IsCreated) _boundsBuffer.Dispose();
            if (_gridBuffer.IsCreated) _gridBuffer.Dispose();
            if (_gridTempBuffer.IsCreated) _gridTempBuffer.Dispose();
            if (_covBuffer.IsCreated) _covBuffer.Dispose();
            if (_gridLut.IsCreated) _gridLut.Dispose();
            if (_blockBuffer.IsCreated) _blockBuffer.Dispose();
            if (_blockColorBuffer.IsCreated) _blockColorBuffer.Dispose();
            if (_bubblesBuffer.IsCreated) _bubblesBuffer.Dispose();
            if (_bubblesPoolBuffer.IsCreated) _bubblesPoolBuffer.Dispose();
            if (_componentsBuffer.IsCreated) _componentsBuffer.Dispose();
            if (_gridIDBuffer.IsCreated) _gridIDBuffer.Dispose();
            if (_controllerBuffer.IsCreated) _controllerBuffer.Dispose();
            if (_lastControllerBuffer.IsCreated) _lastControllerBuffer.Dispose();
            if (_slimeInstances.IsCreated) _slimeInstances.Dispose();
            if (_colliderBuffer.IsCreated)  _colliderBuffer.Dispose();

            _marchingCubes.Dispose();
            _particlePosBuffer.Release();
            _particleCovBuffer.Release();
            _bubblesDataBuffer.Release();

        }

        void Update()
        {
            HandleMouseInteraction();

            if (renderMode == RenderMode.Particles)
            {
                Graphics.DrawMeshInstancedProcedural(particleMesh, 0, particleMat, _bounds, PBF_Utils.Num);
            }
            else if (renderMode == RenderMode.Surface)
            {
                if (_mesh != null)
                    Graphics.DrawMesh(_mesh, Matrix4x4.TRS(_bounds.min, Quaternion.identity, Vector3.one), mat, 0);

                Graphics.DrawMeshInstancedProcedural(particleMesh, 0, bubblesMat, _bounds, PBF_Utils.BubblesCount);
            }

            if (concentration > 5)
            {
                foreach (var slime in _slimeInstances)
                {
                    if (!slime.Active) continue;

                    Graphics.DrawMesh(faceMesh, Matrix4x4.TRS(slime.Pos * PBF_Utils.Scale,
                        Quaternion.LookRotation(-slime.Dir),
                        0.2f * math.sqrt(slime.Radius * PBF_Utils.Scale) * Vector3.one), faceMat, 0);
                }
            }
        }

        private void FixedUpdate()
        {
            for (int i = 0; i < 2; i++)
            {
                Profiler.BeginSample("Simulate");
                Simulate();
                Profiler.EndSample();
            }

            Surface();
            
            Control();
            
            Bubbles();
            
            bubblesNum = PBF_Utils.BubblesCount - _bubblesPoolBuffer.Length;
            
            if (renderMode == RenderMode.Particles)
            {
                _particlePosBuffer.SetData(_particles);
                particleMat.SetInt("_Aniso", 0);
            }
            else
                _bubblesDataBuffer.SetData(_bubblesBuffer);
            
            _bounds = new Bounds()
            {
                min = minPos * PBF_Utils.Scale,
                max = maxPos * PBF_Utils.Scale
            };
        }

        private void Surface()
        {
            Profiler.BeginSample("Render");

            var handle = new Reconstruction.ComputeMeanPosJob
            {
                Lut = _lut,
                Ps = _particles,
                MeanPos = _particlesTemp,
            }.Schedule(_particles.Length, batchCount);

            if (useAnisotropic)
            {
                handle = new Reconstruction.ComputeCovarianceJob
                {
                    Lut = _lut,
                    Ps = _particles,
                    MeanPos = _particlesTemp,
                    GMatrix = _covBuffer,
                }.Schedule(_particles.Length, batchCount, handle);
            }

            new Reconstruction.CalcBoundsJob()
            {
                Ps = _particles,
                Bounds = _boundsBuffer,
            }.Schedule(handle).Complete();

            Profiler.EndSample();

            _gridLut.Clear();
            float blockSize = PBF_Utils.CellSize * 4;
            minPos = math.floor(_boundsBuffer[0] / blockSize) * blockSize;
            maxPos = math.ceil(_boundsBuffer[1] / blockSize) * blockSize;

            Profiler.BeginSample("Allocate");
            handle = new Reconstruction.ClearGridJob
            {
                Grid = _gridBuffer,
                GridID = _gridIDBuffer,
            }.Schedule(_gridBuffer.Length, batchCount);

            handle = new Reconstruction.AllocateBlockJob()
            {
                Ps = _particlesTemp,
                GridLut = _gridLut,
                MinPos = minPos,
            }.Schedule(handle);
            handle.Complete();

            var keys = _gridLut.GetKeyArray(Allocator.TempJob);
            blockNum = keys.Length;

            new Reconstruction.ColorBlockJob()
            {
                Keys = keys,
                Blocks = _blockBuffer,
                BlockColors = _blockColorBuffer,
            }.Schedule().Complete();

            Profiler.EndSample();

            Profiler.BeginSample("Splat");

#if USE_SPLAT_SINGLE_THREAD
            handle = new Reconstruction.DensityProjectionJob()
            {
                Ps = _particlesTemp,
                GMatrix = _covBuffer,
                Grid = _gridBuffer,
                GridLut = _gridLut,
                MinPos = minPos,
                UseAnisotropic = useAnisotropic,
            }.Schedule();
#elif USE_SPLAT_COLOR8
            for (int i = 0; i < 8; i++)
            {
                int2 slice = new int2(_blockColorBuffer[i], _blockColorBuffer[i + 1]);
                int count = slice.y - slice.x;
                handle = new Reconstruction.DensitySplatColoredJob()
                {
                    ParticleLut = _lut,
                    ColorKeys = _blockBuffer.Slice(slice.x, count),
                    Ps = _particlesTemp,
                    GMatrix = _covBuffer,
                    Grid = _gridBuffer,
                    GridLut = _gridLut,
                    MinPos = minPos,
                    UseAnisotropic = useAnisotropic,
                }.Schedule(count, count, handle);
            }
#else
            handle = new Reconstruction.DensityProjectionParallelJob()
            {
                Ps = _particlesTemp,
                GMatrix = _covBuffer,
                GridLut = _gridLut,
                Grid = _gridBuffer,
                ParticleLut = _lut,
                Keys = keys,
                UseAnisotropic = useAnisotropic,
                MinPos = minPos,
            }.Schedule(keys.Length, batchCount);
#endif
            handle.Complete();
            Profiler.EndSample();

            Profiler.BeginSample("Blur");

            new Reconstruction.GridBlurJob()
            {
                Keys = keys,
                GridLut = _gridLut,
                GridRead = _gridBuffer,
                GridWrite = _gridTempBuffer,
            }.Schedule(keys.Length, batchCount, handle).Complete();

            Profiler.EndSample();

            Profiler.BeginSample("Marching cubes");
            _mesh = _marchingCubes.MarchingCubesParallel(keys, _gridLut, _gridTempBuffer, threshold, PBF_Utils.Scale * PBF_Utils.CellSize);
            Profiler.EndSample();
            
            Profiler.BeginSample("CCA");
            _componentsBuffer.Clear();
            handle = new Effects.ConnectComponentBlockJob()
            {
                Keys = keys,
                Grid = _gridBuffer,
                GridLut = _gridLut,
                Components = _componentsBuffer,
                GridID = _gridIDBuffer,
                Threshold = 1e-4f,
            }.Schedule();
            
            handle = new Effects.ParticleIDJob()
            {
                GridLut = _gridLut,
                GridID = _gridIDBuffer,
                Particles = _particles,
                MinPos = minPos,
            }.Schedule(_particles.Length, batchCount, handle);
            
            handle.Complete();
            Profiler.EndSample();

            keys.Dispose();
        }

        private void Simulate()
        {
            _lut.Clear();
            new Simulation_PBF.ApplyForceJob
            {
                Ps = _particles,
                Velocity = _velocityBuffer,
                PsNew = _particlesTemp,
                Controllers = _controllerBuffer,
                Gravity = new float3(0, gravity, 0),
            }.Schedule(_particles.Length, batchCount).Complete();

            new Simulation_PBF.HashJob
            {
                Ps = _particlesTemp,
                Hashes = _hashes,
            }.Schedule(_particles.Length, batchCount).Complete();

            _hashes.SortJob(new PBF_Utils.Int2Comparer()).Schedule().Complete();

            new Simulation_PBF.BuildLutJob
            {
                Hashes = _hashes,
                Lut = _lut
            }.Schedule().Complete();

            new Simulation_PBF.ShuffleJob
            {
                Hashes = _hashes,
                PsRaw = _particles,
                PsNew = _particlesTemp,
                Velocity = _velocityBuffer,
                PosOld = _posOld,
                PosPredict = _posPredict,
                VelocityOut = _velocityTempBuffer,
            }.Schedule(_particles.Length, batchCount).Complete();

            new Simulation_PBF.ComputeLambdaJob
            {
                Lut = _lut,
                PosPredict = _posPredict,
                Lambda = _lambdaBuffer,
            }.Schedule(_particles.Length, batchCount).Complete();

            new Simulation_PBF.ComputeDeltaPosJob
            {
                Lut = _lut,
                PosPredict = _posPredict,
                Lambda = _lambdaBuffer,
                PsNew = _particles,
            }.Schedule(_particles.Length, batchCount).Complete();

            new Simulation_PBF.UpdateJob
            {
                Ps = _particles,
                PosOld = _posOld,
                Colliders = _colliderBuffer,
                Velocity = _velocityTempBuffer,
            }.Schedule(_particles.Length, batchCount).Complete();

            new Simulation_PBF.ApplyViscosityJob
            {
                Lut = _lut,
                PosPredict = _posPredict,
                VelocityR = _velocityTempBuffer,
                VelocityW = _velocityBuffer,
                ViscosityStrength = viscosityStrength,
            }.Schedule(_particles.Length, batchCount).Complete();
        }

        private void Control()
        {
            _controllerBuffer.Clear();

            for (int i = 0; i < _componentsBuffer.Length; i++)
            {
                var component = _componentsBuffer[i];
                float3 extent = component.BoundsMax - component.Center;
                // float radius = math.max(extent.x , math.max(extent.y, extent.z)) * PBF_Utils.CellSize * 1.5f;
                float radius = math.max(1, (extent.x + extent.y + extent.z) * PBF_Utils.CellSize * 0.6f);
                float3 center = minPos + component.Center * PBF_Utils.CellSize;
                if (extent.y < 3)
                    center.y += extent.y * PBF_Utils.Scale * PBF_Utils.CellSize;
                float3 toMain = 5 * math.normalize((float3)trans.position * PBF_Utils.InvScale - center);
                _controllerBuffer.Add(new ParticleController()
                {
                    Center = center,
                    Radius = radius,
                    Velocity = _connect ? toMain : float3.zero,
                    Concentration = concentration,
                });
            }
            if (_controllerBuffer.Length == 1) _connect = false;
            
            RearrangeInstances();
        }

        private void RearrangeInstances()
        {
            if (_slimeInstances.Length - _instancePool.Count > _controllerBuffer.Length)
            {
                var used = new NativeArray<bool>(_slimeInstances.Length, Allocator.Temp);
                for (int controllerID = 0; controllerID < _controllerBuffer.Length; controllerID++)
                {
                    var controller = _controllerBuffer[controllerID];
                    var center = controller.Center;
                    int instanceID = -1;
                    float minDst = float.MaxValue;
                    for (int j = 0; j < _slimeInstances.Length; j++)
                    {
                        var slime = _slimeInstances[j];
                        if (used[j] || !slime.Active) continue;
                        var pos = slime.Center;
                        float dst = math.lengthsq(center - pos);
                        if (dst < minDst)
                        {
                            minDst = dst;
                            instanceID = j;
                        }
                    }
                    
                    used[instanceID] = true;
                    UpdateInstanceController(instanceID, controllerID);
                }

                for (int i = 0; i < _slimeInstances.Length; i++)
                {
                    var slime = _slimeInstances[i];
                    if (used[i] || !slime.Active) continue;
                    slime.Active = false;
                    _slimeInstances[i] = slime;
                    _instancePool.Push(i);
                }
                used.Dispose();

                if (!_slimeInstances[_controlledInstance].Active)
                {
                    float3 pos = trans.position * PBF_Utils.InvScale;
                    float minDst = float.MaxValue;
                    for (int i = 0; i < _slimeInstances.Length; i++)
                    {
                        var slime = _slimeInstances[i];
                        if (!slime.Active) continue;

                        float dst = math.lengthsq(pos - slime.Center);
                        if (dst < minDst)
                        {
                            minDst = dst;
                            _controlledInstance = i;
                        }
                    }

                    int controllerID = _slimeInstances[_controlledInstance].ControllerID;
                    UpdateInstanceController(_controlledInstance, controllerID);
                }
            }
            else
            {
                var used = new NativeArray<bool>(_controllerBuffer.Length, Allocator.Temp);
                for (int instanceID = 0; instanceID < _slimeInstances.Length; instanceID++)
                {
                    var slime = _slimeInstances[instanceID];
                    if (!slime.Active)  continue;
                    var pos = slime.Center;
                    int controllerID = -1;
                    float minDst = float.MaxValue;
                    for (int j = 0; j < _controllerBuffer.Length; j++)
                    {
                        if (used[j]) continue;
                        var cl = _controllerBuffer[j];
                        var center = cl.Center;
                        float dst = math.lengthsq(center - pos);
                        if (dst < minDst)
                        {
                            minDst = dst;
                            controllerID = j;
                        }
                    }
                    used[controllerID] = true;
                    UpdateInstanceController(instanceID, controllerID);
                }
                
                for (int i = 0; i < _controllerBuffer.Length; i++)
                {
                    if (used[i]) continue;
                    var controller = _controllerBuffer[i];
                    float3 dir = math.normalizesafe(
                        math.lengthsq(controller.Velocity) < 1e-3f
                            ? (float3)trans.position - controller.Center
                            : controller.Velocity,
                        new float3(1, 0, 0));
                    new Effects.RayInsectJob
                    {
                        GridLut = _gridLut,
                        Grid = _gridBuffer,
                        Result = _boundsBuffer,
                        Threshold = threshold,
                        Pos = controller.Center,
                        Dir = dir,
                        MinPos = minPos,
                    }.Schedule().Complete();
                    
                    float3 newPos = _boundsBuffer[0];
                    if (!math.all(math.isfinite(newPos)))
                        newPos = controller.Center + dir * controller.Radius * 0.5f;
                    
                    SlimeInstance slime = new SlimeInstance()
                    {
                        Active = true,
                        Center =  controller.Center,
                        Radius = controller.Radius,
                        Dir = dir,
                        Pos = newPos,
                        ControllerID = i,
                    };
                    if (_instancePool.Count > 0)
                        _slimeInstances[_instancePool.Pop()] = slime;
                    else
                        _slimeInstances.Add(slime);
                }
                used.Dispose();
            }
        }

        private void UpdateInstanceController(int instanceID, int controllerID)
        {
            var slime = _slimeInstances[instanceID];
            var controller = _controllerBuffer[controllerID];
            
            if (instanceID == _controlledInstance)
                controller.Velocity = _velocity * PBF_Utils.InvScale;

            slime.ControllerID = controllerID;
            float speed = 0.1f;
            slime.Radius = math.lerp(slime.Radius, controller.Radius, speed);
            slime.Center = math.lerp(slime.Center, controller.Center, speed);
            Vector3 vec = controller.Velocity;
            if (vec.sqrMagnitude > 1e-4f)
            {
                var newDir = Vector3.Slerp(slime.Dir, vec.normalized, speed);
                newDir.y = math.clamp(newDir.y, -0.2f, 0.5f);
                slime.Dir = newDir.normalized;
            }
            else
                slime.Dir = Vector3.Slerp(slime.Dir, new Vector3(slime.Dir.x, 0, slime.Dir.z), speed);
            
            new Effects.RayInsectJob
            {
                GridLut = _gridLut,
                Grid = _gridBuffer,
                Result = _boundsBuffer,
                Threshold = threshold,
                Pos = controller.Center,
                Dir = slime.Dir,
                MinPos = minPos,
            }.Schedule().Complete();
            
            float3 newPos = _boundsBuffer[0];
            if (math.all(math.isfinite(newPos)))
                slime.Pos = Vector3.Lerp(slime.Pos + vec * PBF_Utils.DeltaTime, newPos, 0.1f);
            else
                slime.Pos = controller.Center;
            
            _slimeInstances[instanceID] = slime;
            
            if (instanceID == _controlledInstance)
            {
                controller.Center = trans.position * PBF_Utils.InvScale;
                _controllerBuffer[controllerID] = controller;
            }
        }

        private void Bubbles()
        {
            var handle = new Effects.GenerateBubblesJobs()
            {
                GridLut = _gridLut,
                Keys = _blockBuffer,
                Grid = _gridBuffer,
                BubblesStack = _bubblesPoolBuffer,
                BubblesBuffer = _bubblesBuffer,
                Speed = 0.01f * bubbleSpeed,
                Threshold = threshold * 1.2f,
                BlockCount = blockNum,
                MinPos = minPos,
                Seed = (uint)Time.frameCount,
            }.Schedule();

            handle = new Effects.BubblesViscosityJob()
            {
                Lut = _lut,
                Particles = _particles,
                VelocityR = _velocityBuffer,
                BubblesBuffer = _bubblesBuffer,
                Controllers = _controllerBuffer,
                ViscosityStrength = viscosityStrength / 50,
            }.Schedule(_bubblesBuffer.Length, batchCount, handle);

            handle = new Effects.UpdateBubblesJob()
            {
                GridLut = _gridLut,
                Grid = _gridBuffer,
                BubblesStack = _bubblesPoolBuffer,
                BubblesBuffer = _bubblesBuffer,
                Threshold = threshold * 1.2f,
                MinPos = minPos,
            }.Schedule(handle);
            
            handle.Complete();
        }

        void HandleMouseInteraction()
        {
            if (Input.GetKeyDown(KeyCode.P))
                _connect = true;
            
            if (Input.GetKeyDown(KeyCode.R))
            {
                for (int i = 0; i < _slimeInstances.Length; i++)
                {
                    if (!_slimeInstances[i].Active) continue;
                    _controlledInstance = i;
                    trans.position = _slimeInstances[i].Center * PBF_Utils.Scale;
                    break;
                }
            }
            
            // if (Input.GetKey(KeyCode.W))
            //     velocity += new float3(0, 0, 1);
            // if (Input.GetKey(KeyCode.S))
            //     velocity += new float3(0, 0, -1);
            // if (Input.GetKey(KeyCode.A))
            //     velocity += new float3(-1, 0, 0);
            // if (Input.GetKey(KeyCode.D))
            //     velocity += new float3(1, 0, 0);
            // if (Input.GetKeyDown(KeyCode.Space))
            //     _velocityY = new float3(0, 3, 0);
            // else
            //     _velocityY = pos.y > 1e-5f ? _velocityY + new float3(0, -5f, 0) * Time.deltaTime : float3.zero;
            _velocity = trans.GetComponent<Rigidbody>().velocity;
            // pos += _velocity * Time.deltaTime;
            // pos.y = Mathf.Max(0, pos.y);
            // trans.position = pos;
        }

        private void OnDrawGizmos()
        {
            if (!Application.isPlaying)
                return;
            
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(_bounds.center, _bounds.size);
            if (gridDebug)
            {
                Gizmos.color = Color.blue;
                for (var i = 0; i < blockNum; i++)
                {
                    var block = _blockBuffer[i];
                    Vector3 blockMinPos = new Vector3(block.x, block.y, block.z) * PBF_Utils.CellSize * 0.4f +
                                          _bounds.min;
                    Vector3 size = new Vector3(PBF_Utils.CellSize, PBF_Utils.CellSize, PBF_Utils.CellSize) * 0.4f;
                    Gizmos.DrawWireCube(blockMinPos + size * 0.5f, size);
                }
            }

            if (componentDebug)
            {
                Gizmos.color = Color.green;
                for (var i = 0; i < _componentsBuffer.Length; i++)
                {
                    var c = _componentsBuffer[i];
                    var size = (c.BoundsMax - c.BoundsMin) * PBF_Utils.Scale * PBF_Utils.CellSize;
                    var center = c.Center * PBF_Utils.Scale * PBF_Utils.CellSize;
                    Gizmos.DrawWireCube(_bounds.min + (Vector3)center, size);
                }
                
                for (var i = 0; i < _slimeInstances.Length; i++)
                {
                    var slime = _slimeInstances[i];
                    if (!slime.Active) continue;
                    Gizmos.DrawWireSphere(slime.Center * PBF_Utils.Scale, slime.Radius * PBF_Utils.Scale);
                    UnityEditor.Handles.Label(slime.Center * PBF_Utils.Scale, $"id:{i}");
                    if (_connect)
                        Gizmos.DrawLine(slime.Center * PBF_Utils.Scale + new float3(0, 0.1f, 0), trans.position + new Vector3(0, 0.1f, 0));
                }

                Gizmos.color = Color.cyan;
                for (var i = 0; i < _colliderBuffer.Length; i++)
                {
                    var c = _colliderBuffer[i];
                    Gizmos.DrawWireCube(c.Center * PBF_Utils.Scale, c.Extent * PBF_Utils.Scale * 2);
                }
            }
        }
    }
}
