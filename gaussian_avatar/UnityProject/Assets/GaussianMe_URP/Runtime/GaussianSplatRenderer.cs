// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace GaussianSplatting.Runtime
{
    class GaussianSplatRenderSystem
    {
        // ReSharper disable MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal static readonly ProfilerMarker s_ProfDraw = new(ProfilerCategory.Render, "GaussianSplat.Draw", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCompose = new(ProfilerCategory.Render, "GaussianSplat.Compose", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCalcView = new(ProfilerCategory.Render, "GaussianSplat.CalcView", MarkerFlags.SampleGPU);
        // ReSharper restore MemberCanBePrivate.Global

        public static GaussianSplatRenderSystem instance => ms_Instance ??= new GaussianSplatRenderSystem();
        static GaussianSplatRenderSystem ms_Instance;

        readonly Dictionary<GaussianSplatRenderer, MaterialPropertyBlock> m_Splats = new();
        readonly HashSet<Camera> m_CameraCommandBuffersDone = new();
        readonly List<(GaussianSplatRenderer, MaterialPropertyBlock)> m_ActiveSplats = new();

        CommandBuffer m_CommandBuffer;

        public void RegisterSplat(GaussianSplatRenderer r)
        {
            if (m_Splats.Count == 0)
            {
                if (GraphicsSettings.currentRenderPipeline == null)
                    Camera.onPreCull += OnPreCullCamera;
            }

            m_Splats.Add(r, new MaterialPropertyBlock());
        }

        public void UnregisterSplat(GaussianSplatRenderer r)
        {
            if (!m_Splats.ContainsKey(r))
                return;
            m_Splats.Remove(r);
            if (m_Splats.Count == 0)
            {
                if (m_CameraCommandBuffersDone != null)
                {
                    if (m_CommandBuffer != null)
                    {
                        foreach (var cam in m_CameraCommandBuffersDone)
                        {
                            if (cam)
                                cam.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                        }
                    }
                    m_CameraCommandBuffersDone.Clear();
                }

                m_ActiveSplats.Clear();
                m_CommandBuffer?.Dispose();
                m_CommandBuffer = null;
                Camera.onPreCull -= OnPreCullCamera;
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public bool GatherSplatsForCamera(Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return false;
            // gather all active & valid splat objects
            m_ActiveSplats.Clear();
            foreach (var (gs, mpb) in m_Splats)
            {
                if (gs == null || !gs.isActiveAndEnabled || !gs.CanActive)
                    continue;
                m_ActiveSplats.Add(new (gs, mpb));
            }
            if (m_ActiveSplats.Count == 0)
                return false;

            // sort them by depth from camera
            var camTr = cam.transform;
            m_ActiveSplats.Sort((a, b) =>
            {
                var trA = a.Item1.transform;
                var trB = b.Item1.transform;
                var posA = camTr.InverseTransformPoint(trA.position);
                var posB = camTr.InverseTransformPoint(trB.position);
                return posA.z.CompareTo(posB.z);
            });

            return true;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public Material SortAndRenderSplats(Camera cam, CommandBuffer cmb)
        {
            Material matComposite = null;
            foreach (var kvp in m_ActiveSplats)
            {
                var gs = kvp.Item1;
                matComposite = gs.m_MatComposite;
                var mpb = kvp.Item2;

                // sort
                var matrix = gs.rootBone.localToWorldMatrix;
                if (gs.m_FrameCounter % gs.m_SortNthFrame == 0)
                    gs.SortPoints(cmb, cam, matrix);
                ++gs.m_FrameCounter;

                // cache view
                kvp.Item2.Clear();
                var displayMat = gs.m_MatSplats;
                if (displayMat == null)
                    continue;

                if (gs.useOrt)
                {
                    gs.SetOrtDataOnMaterial(mpb, cam);
                    mpb.SetBuffer(GaussianSplatRenderer.Props.SplatViewData, gs.m_GpuView);
                    mpb.SetBuffer(GaussianSplatRenderer.Props.OrderBuffer, gs.m_GpuSortKeys);
                }
                else
                {
                    gs.SetAssetDataOnMaterial(mpb);
                    mpb.SetBuffer(GaussianSplatRenderer.Props.SplatChunks, gs.m_GpuChunks);
                    mpb.SetBuffer(GaussianSplatRenderer.Props.SplatViewData, gs.m_GpuView);
                    mpb.SetBuffer(GaussianSplatRenderer.Props.OrderBuffer, gs.m_GpuSortKeys);

                    cmb.BeginSample(s_ProfCalcView);
                    gs.CalcViewData(cmb, cam, matrix);
                    cmb.EndSample(s_ProfCalcView);
                }

                // draw
                cmb.BeginSample(s_ProfDraw);
                cmb.DrawProcedural(gs.m_GpuIndexBuffer, matrix, displayMat, 0, MeshTopology.Triangles, 6, gs.splatCount, mpb);
                cmb.EndSample(s_ProfDraw);
            }
            return matComposite;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        // ReSharper disable once UnusedMethodReturnValue.Global - used by HDRP/URP features that are not always compiled
        public CommandBuffer InitialClearCmdBuffer(Camera cam)
        {
            m_CommandBuffer ??= new CommandBuffer {name = "RenderGaussianSplats"};
            if (GraphicsSettings.currentRenderPipeline == null && cam != null && !m_CameraCommandBuffersDone.Contains(cam))
            {
                cam.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                m_CameraCommandBuffersDone.Add(cam);
            }

            // get render target for all splats
            m_CommandBuffer.Clear();
            return m_CommandBuffer;
        }

        void OnPreCullCamera(Camera cam)
        {
            if (!GatherSplatsForCamera(cam))
                return;

            InitialClearCmdBuffer(cam);

            m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT, -1, -1, 0, FilterMode.Point, GraphicsFormat.R16G16B16A16_SFloat);
            m_CommandBuffer.SetRenderTarget(GaussianSplatRenderer.Props.GaussianSplatRT, BuiltinRenderTextureType.CurrentActive);
            m_CommandBuffer.ClearRenderTarget(RTClearFlags.Color, new Color(0, 0, 0, 0), 0, 0);

            // add sorting, view calc and drawing commands for each splat object
            Material matComposite = SortAndRenderSplats(cam, m_CommandBuffer);

            // compose
            m_CommandBuffer.BeginSample(s_ProfCompose);
            m_CommandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
            m_CommandBuffer.DrawProcedural(Matrix4x4.identity, matComposite, 0, MeshTopology.Triangles, 3, 1);
            m_CommandBuffer.EndSample(s_ProfCompose);
            m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT);
        }
    }

    [ExecuteInEditMode]
    public class GaussianSplatRenderer : MonoBehaviour
    {
        public Transform rootBone;
        public GaussianSplatAsset m_Asset;

        [Range(1,30)] [Tooltip("Sort splats only every N frames")]
        public int m_SortNthFrame = 1;

        public Shader m_ShaderSplats;
        public Shader m_ShaderSplats2;
        public Shader m_ShaderComposite;
        [Tooltip("Gaussian splatting compute shader")]
        public ComputeShader m_CSSplatUtilities;
        public ComputeShader m_CSDeviceRadixSort;

        int m_SplatCount; // initially same as asset splat count, but editing can change this
        GraphicsBuffer m_GpuSortDistances;
        internal GraphicsBuffer m_GpuSortKeys;
        GraphicsBuffer m_GpuPosData;
        GraphicsBuffer m_GpuOtherData;
        Texture m_GpuColorData;
        internal GraphicsBuffer m_GpuChunks;
        internal bool m_GpuChunksValid;
        internal GraphicsBuffer m_GpuView;
        internal GraphicsBuffer m_GpuIndexBuffer;

        GpuSorting m_Sorter;
        GpuSorting.Args m_SorterArgs;

        internal Material m_MatSplats;
        internal Material m_MatComposite;

        public bool cpuSorting;

        internal int m_FrameCounter;
        GaussianSplatAsset m_PrevAsset;
        Hash128 m_PrevHash;

        static readonly ProfilerMarker s_ProfSort = new(ProfilerCategory.Render, "GaussianSplat.Sort", MarkerFlags.SampleGPU);

        internal static class Props
        {
            public static readonly int SplatPos = Shader.PropertyToID("_SplatPos");
            public static readonly int SplatOther = Shader.PropertyToID("_SplatOther");
            public static readonly int SplatColor = Shader.PropertyToID("_SplatColor");
            public static readonly int SplatFormat = Shader.PropertyToID("_SplatFormat");
            public static readonly int SplatChunks = Shader.PropertyToID("_SplatChunks");
            public static readonly int SplatChunkCount = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int SplatViewData = Shader.PropertyToID("_SplatViewData");
            public static readonly int SplatViewData2 = Shader.PropertyToID("_SplatViewData2");
            public static readonly int OrderBuffer = Shader.PropertyToID("_OrderBuffer");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int GaussianSplatRT = Shader.PropertyToID("_GaussianSplatRT");
            public static readonly int SplatSortKeys = Shader.PropertyToID("_SplatSortKeys");
            public static readonly int SplatSortDistances = Shader.PropertyToID("_SplatSortDistances");
            public static readonly int MatrixVP = Shader.PropertyToID("_MatrixVP");
            public static readonly int MatrixMV = Shader.PropertyToID("_MatrixMV");
            public static readonly int MatrixP = Shader.PropertyToID("_MatrixP");
            public static readonly int MatrixObjectToWorld = Shader.PropertyToID("_MatrixObjectToWorld");
            public static readonly int MatrixWorldToObject = Shader.PropertyToID("_MatrixWorldToObject");
            public static readonly int VecScreenParams = Shader.PropertyToID("_VecScreenParams");
            public static readonly int VecWorldSpaceCameraPos = Shader.PropertyToID("_VecWorldSpaceCameraPos");
        }

        public GaussianSplatAsset asset => m_Asset;
        public int splatCount => m_SplatCount;

        enum KernelIndices
        {
            SetIndices,
            CalcDistances,
            CalcDistances1,
            CalcViewData,
        }

        public bool HasValidAsset =>
            m_Asset != null &&
            m_Asset.splatCount > 0 &&
            m_Asset.formatVersion == GaussianSplatAsset.kCurrentVersion &&
            m_Asset.posData != null &&
            m_Asset.otherData != null &&
            m_Asset.shData != null &&
            m_Asset.colorData != null;
        public bool HasValidRenderSetup => m_GpuPosData != null && m_GpuOtherData != null && m_GpuChunks != null;

        public bool CanActive =>
            (HasValidAsset && HasValidRenderSetup) ||
            (useOrt && m_GpuView != null && splatNativeArray.IsCreated);

        const int kGpuViewDataSize = 40;

        void CreateResourcesForAsset()
        {
            if (!HasValidAsset)
                return;

            m_SplatCount = asset.splatCount;
            m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.posData.dataSize / 4), 4) { name = "GaussianPosData" };
            m_GpuPosData.SetData(asset.posData.GetData<uint>());
            m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.otherData.dataSize / 4), 4) { name = "GaussianOtherData" };
            m_GpuOtherData.SetData(asset.otherData.GetData<uint>());
            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
            var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
            var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
            tex.SetPixelData(asset.colorData.GetData<byte>(), 0);
            tex.Apply(false, true);
            ClearTexture();
            m_GpuColorData = tex;
            if (asset.chunkData != null && asset.chunkData.dataSize != 0)
            {
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    (int) (asset.chunkData.dataSize / UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()),
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunks.SetData(asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());
                m_GpuChunksValid = true;
            }
            else
            {
                // just a dummy chunk buffer
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1,
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunksValid = false;
            }

            m_GpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_Asset.splatCount, kGpuViewDataSize);
            m_GpuIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 6, 2);
            // cube indices, most often we use only the first quad
            m_GpuIndexBuffer.SetData(new ushort[]
            {
                0, 1, 2, 1, 3, 2,
                // 4, 6, 5, 5, 6, 7,
                // 0, 2, 4, 4, 2, 6,
                // 1, 5, 3, 5, 7, 3,
                // 0, 4, 1, 4, 5, 1,
                // 2, 3, 6, 3, 7, 6
            });

            InitSortBuffers(splatCount);
        }

        public const int fSize = 10;
        public const int kGpuViewDataSize2 = fSize * 4;

        void CreateResourcesFromSize(int count)
        {
            useOrt = true;

            m_SplatCount = count;
            splatNativeArray = new NativeArray<float>(fSize * count, Allocator.Persistent);

            m_GpuView = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                count,
                kGpuViewDataSize2
            );
            m_GpuIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 6, 2);
            // cube indices, most often we use only the first quad
            m_GpuIndexBuffer.SetData(new ushort[]
            {
                0, 1, 2, 1, 3, 2,
                // 4, 6, 5, 5, 6, 7,
                // 0, 2, 4, 4, 2, 6,
                // 1, 5, 3, 5, 7, 3,
                // 0, 4, 1, 4, 5, 1,
                // 2, 3, 6, 3, 7, 6
            });

            InitSortBuffers(count);
            sizeList = new int[count];
            depthIndex = new uint[count];
        }

        public void SetColor(byte[] bytes)
        {
            var size = bytes.Length / 4;
            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(size);

            var texBytes = new byte[texWidth * texHeight * 4];
            bytes.CopyTo(texBytes, 0);

            var tex = new Texture2D
            (
                texWidth,
                texHeight,
                GraphicsFormat.R8G8B8A8_UNorm,
                TextureCreationFlags.DontInitializePixels |
                TextureCreationFlags.IgnoreMipmapLimit |
                TextureCreationFlags.DontUploadUponCreate)
            {
                name = "GaussianColorData"
            };
            tex.SetPixelData(texBytes, 0);
            tex.Apply(false, true);
            ClearTexture();
            m_GpuColorData = tex;
        }

        public bool useOrt = false;
        NativeArray<float> splatNativeArray;

        public void SetSplatData(
            int count,
            ReadOnlySpan<float> splatData)
        {
            if (m_SplatCount != count || !useOrt)
            {
                OnDisable();
                m_SplatCount = count;
                useOrt = true;
                if (!enabled)
                {
                    enabled = true;
                }
                else
                {
                    OnEnable();
                }
            }
            if (!splatNativeArray.IsCreated)
            {
                splatNativeArray = new NativeArray<float>(fSize * m_SplatCount, Allocator.Persistent);
            }
            splatData.CopyTo(splatNativeArray);
            m_GpuView.SetData(splatNativeArray);
        }

        void InitSortBuffers(int count)
        {
            m_GpuSortDistances?.Dispose();
            m_GpuSortKeys?.Dispose();
            m_SorterArgs.resources.Dispose();

            m_GpuSortDistances = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "GaussianSplatSortDistances" };
            m_GpuSortKeys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "GaussianSplatSortIndices" };

            // init keys buffer to splat indices
            m_CSSplatUtilities.SetBuffer((int)KernelIndices.SetIndices, Props.SplatSortKeys, m_GpuSortKeys);
            m_CSSplatUtilities.SetInt(Props.SplatCount, m_GpuSortDistances.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.SetIndices, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch((int)KernelIndices.SetIndices, (m_GpuSortDistances.count + (int)gsX - 1) / (int)gsX, 1, 1);

            m_SorterArgs.inputKeys = m_GpuSortDistances;
            m_SorterArgs.inputValues = m_GpuSortKeys;
            m_SorterArgs.count = (uint)count;
            if (m_Sorter.Valid)
            {
                m_SorterArgs.resources = GpuSorting.SupportResources.Load((uint)count);
            }

            CSDebugger.Instance.Add<uint>(m_GpuSortDistances, nameof(m_GpuSortDistances));
            CSDebugger.Instance.Add<uint>(m_GpuSortKeys, nameof(m_GpuSortKeys));
            CSDebugger.Instance.Add<uint>(m_SorterArgs.resources.altBuffer, nameof(m_SorterArgs.resources.altBuffer));
            CSDebugger.Instance.Add<uint>(m_SorterArgs.resources.altPayloadBuffer, nameof(m_SorterArgs.resources.altPayloadBuffer));
            CSDebugger.Instance.Add<uint>(m_SorterArgs.resources.passHistBuffer, nameof(m_SorterArgs.resources.passHistBuffer));
            CSDebugger.Instance.Add<uint>(m_SorterArgs.resources.globalHistBuffer, nameof(m_SorterArgs.resources.globalHistBuffer));
        }

        public void OnEnable()
        {
            m_FrameCounter = 0;
            if (!SystemInfo.supportsComputeShaders)
                return;

            Shader shader = m_ShaderSplats;

            if (useOrt)
            {
                shader = m_ShaderSplats2;
            }

            if (shader == null || m_ShaderComposite == null )
                return;

            m_MatSplats = new Material(shader) {name = "GaussianSplats"};
            m_MatComposite = new Material(m_ShaderComposite) {name = "GaussianClearDstAlpha"};

            m_Sorter = new GpuSorting(m_CSSplatUtilities);
            GaussianSplatRenderSystem.instance.RegisterSplat(this);

            if (useOrt)
            {
                CreateResourcesFromSize(m_SplatCount);
            }
            else
            {
                CreateResourcesForAsset();
            }

            if (rootBone == null) rootBone = transform;
        }

        void SetAssetDataOnCS(CommandBuffer cmb, KernelIndices kernel)
        {
            ComputeShader cs = m_CSSplatUtilities;
            int kernelIndex = (int) kernel;
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatPos, m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatChunks, m_GpuChunks);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatOther, m_GpuOtherData);
            cmb.SetComputeTextureParam(cs, kernelIndex, Props.SplatColor, m_GpuColorData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatViewData, m_GpuView);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.OrderBuffer, m_GpuSortKeys);

            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            cmb.SetComputeIntParam(cs, Props.SplatFormat, (int)format);
            cmb.SetComputeIntParam(cs, Props.SplatCount, m_SplatCount);
            cmb.SetComputeIntParam(cs, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
        }

        internal void SetOrtDataOnMaterial(MaterialPropertyBlock mat, Camera cam)
        {
            var tr = rootBone;

            var matView = cam.worldToCameraMatrix;
            var matProj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
            var matO2W = tr.localToWorldMatrix;
            // var matW2O = tr.worldToLocalMatrix;
            var screenW = cam.pixelWidth;
            var screenH = cam.pixelHeight;
            var screenPar = new Vector4(screenW, screenH, 0, 0);
            // var camPos = cam.transform.position;

            if (m_GpuColorData != null)
            {
                mat.SetTexture(Props.SplatColor, m_GpuColorData);
            }

            // m_MatSplats
            mat.SetMatrix(Props.MatrixObjectToWorld, matO2W);
            mat.SetMatrix(Props.MatrixVP, matProj * matView);
            mat.SetMatrix(Props.MatrixMV, matView * matO2W);
            mat.SetMatrix(Props.MatrixP, matProj);
            mat.SetVector(Props.VecScreenParams, screenPar);
        }

        internal void SetAssetDataOnMaterial(MaterialPropertyBlock mat)
        {
            mat.SetBuffer(Props.SplatPos, m_GpuPosData);
            mat.SetBuffer(Props.SplatOther, m_GpuOtherData);
            mat.SetTexture(Props.SplatColor, m_GpuColorData);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            mat.SetInteger(Props.SplatFormat, (int)format);
            mat.SetInteger(Props.SplatCount, m_SplatCount);
            mat.SetInteger(Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
        }

        static void DisposeBuffer(ref GraphicsBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }

        static void DisposeBuffer(ref ComputeBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }

        void DisposeResourcesForAsset()
        {
            DisposeBuffer(ref m_GpuPosData);
            DisposeBuffer(ref m_GpuOtherData);
            DisposeBuffer(ref m_GpuChunks);

            DisposeBuffer(ref m_GpuView);
            DisposeBuffer(ref m_GpuIndexBuffer);
            DisposeBuffer(ref m_GpuSortDistances);
            DisposeBuffer(ref m_GpuSortKeys);

            if (splatNativeArray.IsCreated)
            {
                splatNativeArray.Dispose();
            }

            m_SorterArgs.resources.Dispose();

            m_SplatCount = 0;
            m_GpuChunksValid = false;
        }

        void ClearTexture()
        {
            if (m_GpuColorData != null)
            {
                DestroyImmediate(m_GpuColorData);
            }
            m_GpuColorData = null;
        }

        void OnDestroy()
        {
            ClearTexture();
        }

        public void OnDisable()
        {
            useOrt = false;
            DisposeResourcesForAsset();
            GaussianSplatRenderSystem.instance.UnregisterSplat(this);

            DestroyImmediate(m_MatSplats);
            DestroyImmediate(m_MatComposite);
        }

        internal void CalcViewData(CommandBuffer cmb, Camera cam, Matrix4x4 matrix)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            var tr = rootBone;

            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matProj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            Vector4 screenPar = new Vector4(screenW, screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            // calculate view dependent data for each splat
            SetAssetDataOnCS(cmb, KernelIndices.CalcViewData);

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixVP, matProj * matView);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixP, matProj);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);

            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcViewData, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcViewData, (m_GpuView.count + (int)gsX - 1)/(int)gsX, 1, 1);
        }

        private Matrix4x4 lastProj;
        int[] sizeList;
        int[] counts0 = new int[256 * 256];
        int[] starts0 = new int[256 * 256];
        uint[] depthIndex = null;

        void CPUSort(Matrix4x4 viewProj)
        {
            float dot =
                lastProj[2] * viewProj[2] +
                lastProj[6] * viewProj[6] +
                lastProj[10] * viewProj[10];
            if (Math.Abs(dot - 1) < 0.01f)
            {
                return;
            }

            int maxDepth = int.MinValue;
            int minDepth = int.MaxValue;
            for (var i = 0; i < splatCount; ++i)
            {
                var x = splatNativeArray[i * fSize + 0];
                var y = splatNativeArray[i * fSize + 1];
                var z = splatNativeArray[i * fSize + 2];
                int depth = (int)((
                    viewProj[2] * x +
                    viewProj[6] * y +
                    viewProj[10] * z) * 4096) | 0;
                sizeList[i] = depth;
                if (depth > maxDepth) maxDepth = depth;
                if (depth < minDepth) minDepth = depth;
            }

            for (int i = 0; i < 256 * 256; ++i)
            {
                starts0[i] = 0;
                counts0[i] = 0;
            }

            // This is a 16 bit single-pass counting sort
            float depthInv = (256f * 256f - 1f) / (maxDepth - minDepth);
            for (int i = 0; i < splatCount; ++i)
            {
                sizeList[i] = (int)((sizeList[i] - minDepth) * depthInv) | 0;
                counts0[sizeList[i]]++;
            }
            for (int i = 1; i < 256 * 256; ++i)
                starts0[i] = starts0[i - 1] + counts0[i - 1];
            for (int i = 0; i < splatCount; ++i)
            {
                depthIndex[starts0[sizeList[i]]++] = (uint)i;
            }

            m_GpuSortKeys.SetData(depthIndex);
            lastProj = viewProj;
        }

        internal void SortPoints(CommandBuffer cmd, Camera cam, Matrix4x4 matrix)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            Matrix4x4 worldToCamMatrix = cam.worldToCameraMatrix;
            worldToCamMatrix.m20 *= -1;
            worldToCamMatrix.m21 *= -1;
            worldToCamMatrix.m22 *= -1;

            if (useOrt && cpuSorting)
            {
                // var projMatrix = cam.projectionMatrix;
                // var projMatrix = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
                CPUSort(worldToCamMatrix * matrix);
                return;
            }

            // calculate distance to the camera for each splat
            cmd.BeginSample(s_ProfSort);

            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatFormat, (int)m_Asset.posFormat);
            cmd.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, worldToCamMatrix * matrix);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatCount, m_SplatCount);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
            if (useOrt)
            {
                cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances1, Props.SplatSortDistances, m_GpuSortDistances);
                cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances1, Props.SplatSortKeys, m_GpuSortKeys);
                cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances1, Props.SplatViewData2, m_GpuView);
                m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcDistances1, out uint gsX, out _, out _);
                cmd.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcDistances1, (m_GpuSortDistances.count + (int)gsX - 1)/(int)gsX, 1, 1);
            }
            else
            {
                cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatSortDistances, m_GpuSortDistances);
                cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatSortKeys, m_GpuSortKeys);
                cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatChunks, m_GpuChunks);
                cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatPos, m_GpuPosData);
                m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcDistances, out uint gsX, out _, out _);
                cmd.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, (m_GpuSortDistances.count + (int)gsX - 1)/(int)gsX, 1, 1);
            }

            if (CSDebugger.Instance.DebugThisFrame)
            {
                CSDebugger.Instance.PrintNow("CalcDistances");
            }

            // sort the splats
            m_Sorter.Dispatch(cmd, m_SorterArgs, CSDebugger.Instance.DebugThisFrame);
            cmd.EndSample(s_ProfSort);
        }

        public void Update()
        {
            if (useOrt)
            {
                return;
            }
            var curHash = m_Asset ? m_Asset.dataHash : new Hash128();
            if (m_PrevAsset != m_Asset || m_PrevHash != curHash)
            {
                m_PrevAsset = m_Asset;
                m_PrevHash = curHash;
                DisposeResourcesForAsset();
                CreateResourcesForAsset();
            }
        }

        public void ActivateCamera(int index)
        {
            Camera mainCam = Camera.main;
            if (!mainCam)
                return;
            if (!m_Asset || m_Asset.cameras == null)
                return;

            var selfTr = rootBone;
            var camTr = mainCam.transform;
            var prevParent = camTr.parent;
            var cam = m_Asset.cameras[index];
            camTr.parent = selfTr;
            camTr.localPosition = cam.pos;
            camTr.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
            camTr.parent = prevParent;
            camTr.localScale = Vector3.one;
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(camTr);
#endif
        }
    }
}