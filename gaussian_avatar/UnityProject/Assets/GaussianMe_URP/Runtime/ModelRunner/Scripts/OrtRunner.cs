using System;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Unity;
using GaussianSplatting.Runtime;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;

namespace GaussianMe
{
    public enum OnnxRuntimeType
    {
        Default,
        CoreML,
        NNAPI,
        XNNPack,
        Count
    }

    [System.Serializable]
    public class OrtRunner
    {
        public OnnxRuntimeType onnxRuntimeType;
        private InferenceSession inferenceSession;
        private List<string> inputName;
        private List<string> outputName;
        private List<OrtValue> inputTensor;
        private List<OrtValue> outputTensor;

        bool isReady = false;
        public bool IsReady => isReady;
        bool hasResult = false;
        public bool HasResult => hasResult;

        public int vertexSize = 13453;

        public OrtRunner(int vertexSize = 13453)
        {
            this.vertexSize = vertexSize;
        }

        public static bool IsPlatform(params RuntimePlatform[] platforms)
        {
            var currentPlatform = Application.platform;
            return platforms.Any(platform => platform == currentPlatform);
        }

        public static bool IsSupported(OnnxRuntimeType type)
        {
            switch (type)
            {
                case OnnxRuntimeType.CoreML:
                    return IsPlatform(RuntimePlatform.IPhonePlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer);
                case OnnxRuntimeType.NNAPI:
                    return IsPlatform(RuntimePlatform.Android);
                case OnnxRuntimeType.XNNPack:
                    return IsPlatform(RuntimePlatform.Android, RuntimePlatform.IPhonePlayer);
                case OnnxRuntimeType.Default:
                    return true;
            }
            return false;
        }

        public void SetupOnnxRuntimeType(SessionOptions options, OnnxRuntimeType type, int processorCount)
        {
            switch (type)
            {
                case OnnxRuntimeType.CoreML:
                    options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                    break;
                case OnnxRuntimeType.NNAPI:
                    options.AppendExecutionProvider_Nnapi();
                    break;
                case OnnxRuntimeType.XNNPack:
                    options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
                    int threads = Math.Clamp(processorCount, 1, 4);
                    options.AppendExecutionProvider("XNNPACK", new Dictionary<string, string>
                {
                    { "intra_op_num_threads", threads.ToString() },
                });
                    options.IntraOpNumThreads = 1;
                    break;
            }
        }

        public void CreateONNXSession(byte[] bytes, int bsSize, CancellationToken cancellationToken)
        {
            var processorCount = SystemInfo.processorCount;
            Task.Factory.StartNew(() =>
            {
                if (!IsSupported(onnxRuntimeType))
                {
                    Debug.LogError($"onnxRuntimeType is not supported on {Application.platform} use default instead");
                    onnxRuntimeType = OnnxRuntimeType.Default;
                }

                Debug.Log($"CreateONNXSession with OnnxRuntimeType: {onnxRuntimeType}");

                try
                {
                    using var options = new SessionOptions();

                    SetupOnnxRuntimeType(options, onnxRuntimeType, processorCount);

                    inferenceSession = new InferenceSession(bytes, options);
                    inputName = new List<string>() { "input" };

                    inputTensor = new List<OrtValue>()
                    {
                        OrtValue.CreateTensorValueFromMemory(new float[bsSize], new long[] { 1, bsSize })
                    };
                    outputName = new List<string>() { "output" };
                    int outputSize = GaussianSplatRenderer.fSize;
                    outputTensor = new List<OrtValue>()
                    {
                        OrtValue.CreateTensorValueFromMemory(
                            new float[outputSize * vertexSize],
                            new long[] {vertexSize, outputSize}
                        ),
                    };

                    isReady = true;
                }
                catch (Exception ex)
                {
                    Debug.LogError(ex);
                }
            }, cancellationToken);
        }

        public long lastTime;
        System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();
        public ReadOnlySpan<float> RunSession<T>(T[] bytes) where T : unmanaged
        {
            timer.Restart();
            var inputSpan = inputTensor[0].GetTensorMutableDataAsSpan<T>();
            bytes.CopyTo(inputSpan);

            inferenceSession.Run(null, inputName, inputTensor, outputName, outputTensor);
            timer.Stop();
            lastTime = timer.ElapsedMilliseconds;
            hasResult = true;

            return outputTensor[0].GetTensorDataAsSpan<float>();
        }
        public ReadOnlySpan<float> LastOutput => outputTensor[0].GetTensorDataAsSpan<float>();

        GUIStyle labelStyle;
        GUIStyle LabelStyle
        {
            get
            {
                if (labelStyle == null)
                {
                    labelStyle = new GUIStyle(GUI.skin.label)
                    {
                        fontSize = 36
                    };
                    labelStyle.normal.textColor = Color.green;
                }
                return labelStyle;
            }
        }
        public void OnGUI()
        {
            GUILayout.BeginVertical();
            GUILayout.Space(300f);
            float fps = 1000f / (float)lastTime;
            GUILayout.Label($"FPS: {fps:0.00} (process time: {lastTime:0.00}ms)", LabelStyle);
            GUILayout.EndVertical();
        }
    }
}
