using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using GaussianSplatting.Runtime;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;

namespace GaussianMe
{
    public class BlendshapeFollower : MonoBehaviour
    {
        public SkinnedMeshRenderer target;

        public bool ict;
        int bsSize;
        float[] blendshapes;
        float[] modelInput;
        public Dictionary<int, int> indexMap = new Dictionary<int, int>();

        // Copyed from https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf
        public readonly static string[] mediapipeBSNames = new string[] {
            "browDownLe", // 1 - 
            "browDownRight", // 2 - 
            "browInnerUp", // 3 - 
            "browOuterUpLe", // 4 - 
            "browOuterUpRight", // 5 - 
            "cheekPu", // 6 -  // (blendshape predicted by the FaceMesh model)
            "cheekSquintLe", // 7 - 
            "cheekSquintRight", // 8 - 
            "eyeBlinkLe", // 9 - 
            "eyeBlinkRight", // 10 - 
            "eyeLookDownLe", // 11 - 
            "eyeLookDownRight", // 12 - 
            "eyeLookInLe", // 13 - 
            "eyeLookInRight", // 14 - 
            "eyeLookOutLe", // 15 - 
            "eyeLookOutRight", // 16 - 
            "eyeLookUpLe", // 17 - 
            "eyeLookUpRight", // 18 - 
            "eyeSquintLe", // 19 - 
            "eyeSquintRight", // 20 - 
            "eyeWideLe", // 21 - 
            "eyeWideRight", // 22 - 
            "jawForward", // 23 - 
            "jawLe", // 24 - 
            "jawOpen", // 25 - 
            "jawRight", // 26 - 
            "mouthClose", // 27 - 
            "mouthDimpleLe", // 28 - 
            "mouthDimpleRight", // 29 - 
            "mouthFrownLe", // 30 - 
            "mouthFrownRight", // 31 - 
            "mouthFunnel", // 32 - 
            "mouthLe", // 33 - 
            "mouthLowerDownLe", // 34 - 
            "mouthLowerDownRight", // 35 - 
            "mouthPressLe", // 36 - 
            "mouthPressRight", // 37 - 
            "mouthPucker", // 38 - 
            "mouthRight", // 39 - 
            "mouthRollLower", // 40 - 
            "mouthRollUpper", // 41 - 
            "mouthShrugLower", // 42 - 
            "mouthShrugUpper", // 43 - 
            "mouthSmileLe", // 44 - 
            "mouthSmileRight", // 45 - 
            "mouthStretchLe", // 46 - 
            "mouthStretchRight", // 47 - 
            "mouthUpperUpLe", // 48 - 
            "mouthUpperUpRight", // 49 - 
            "noseSneerLe", // 50 - 
            "noseSneerRight", // 51 - 
            // "tongueOut", // 52 -  // (blendshape predicted by the FaceMesh model)
        };

        public readonly static string[] ictBSNames = new string[] {
            "browDownl", "browDownr", "browInnerUpl", "browInnerUpr", "browOuterUpL",
            "browOuterUpR", "cheekPuff_L", "cheekPuff_R", "cheekSquintL", "cheekSquintR",
            "eyeBlinkL", "eyeBlinkR", "eyeLookDownL", "eyeLookDownR", "eyeLookInL",
            "eyeLookInR", "eyeLookOutL", "eyeLookOutR", "eyeLookUpL", "eyeLookUpR",
            "eyeSquintL", "eyeSquintR", "eyeWideL", "eyeWideR", "jawForward",
            "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleL",
            "mouthDimpleR", "mouthFrownL", "mouthFrownR", "mouthFunnel", "mouthLeft",
            "mouthLowerDownL", "mouthLowerDownR", "mouthPressL", "mouthPressR", "mouthPucker",
            "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
            "mouthSmileL", "mouthSmileR", "mouthStretchL", "mouthStretchR", "mouthUpperUpL",
            "mouthUpperUpR", "noseSneerL", "noseSneerR"
        };
        public TextAsset colorAsset;
        public GaussianSplatRenderer targetRenderer;
        public OrtRunner ortRunner;
        [SerializeField] private OrtAsset model;
        private object locker = new object();

        static bool Contains(string source, string target)
        {
            return source.IndexOf(target, StringComparison.OrdinalIgnoreCase) >= 0;
        }
        static int IndexOf(string item, string[] arr)
        {
            for (int i = arr.Length - 1; i >= 0; --i)
            {
                if (Contains(item, arr[i]))
                {
                    return i;
                }
            }
            return -1;
        }

        public List<string> founded = new List<string>();
        public List<string> notFounded = new List<string>();
        IEnumerator Start()
        {
            ortRunner = new OrtRunner(ict ? 14480 : 13453);
            bsSize = ict ? 53 : 51;
            blendshapes = new float[bsSize];
            modelInput = new float[bsSize];
            ortRunner.CreateONNXSession(model.bytes, bsSize, destroyCancellationToken);

            if (target != null && target.sharedMesh is Mesh mesh)
            {
                indexMap.Clear();
                founded.Clear();
                notFounded.Clear();
                var arr = ict ? ictBSNames : mediapipeBSNames;
                for (int i = 0; i < mesh.blendShapeCount; ++i)
                {
                    var index = IndexOf(mesh.GetBlendShapeName(i), arr);
                    if (index != -1)
                    {
                        indexMap[i] = index;
                        founded.Add(mesh.GetBlendShapeName(i));
                    }
                    else
                    {
                        notFounded.Add(mesh.GetBlendShapeName(i));
                    }
                }
            }

            if (colorAsset != null)
            {
                targetRenderer.SetColor(colorAsset.bytes);
            }

            while (!ortRunner.IsReady)
            {
                yield return null;
            }

            StartThread();
        }

        CancellationTokenSource cancellationTokenSource = null;
        void StartThread()
        {
            cancellationTokenSource = new();
            var token = cancellationTokenSource.Token;

            Task.Factory.StartNew(() =>
            {
                while (true)
                {
                    try
                    {
                        if (ortRunner != null && ortRunner.IsReady)
                        {
                            ortRunner.RunSession(modelInput);
                        }
                        Task.Yield();
                    }
                    catch (System.Threading.ThreadAbortException)
                    {
                        break;
                    }
                    catch (System.Exception e)
                    {
                        Debug.LogError(e);
                    }
                    token.ThrowIfCancellationRequested();
                }
            }, token);
        }

        void OnApplicationPause(bool pause)
        {
            if (pause)
            {
                cancellationTokenSource?.Cancel();
            }
            else
            {
                if (cancellationTokenSource != null)
                {
                    StartThread();
                }
            }
        }

        void OnDisable()
        {
            OnApplicationPause(true);
        }

        void OnEnable()
        {
            OnApplicationPause(false);
        }

        public bool eyefix;
        void CopyICTLR(float[] values)
        {
            values[3] = values[2];
            values[7] = values[6];
            if (eyefix)
            {
                values[23] = values[24] = 1f;
            }
        }

        void UpdateBlendshapeInput()
        {
            foreach (var (smrIndex, mediapipeIndex) in indexMap)
            {
                blendshapes[mediapipeIndex] = target.GetBlendShapeWeight(smrIndex) / 100f;
            }
            if (ict)
            {
                CopyICTLR(blendshapes);
            }
            lock (locker)
            {
                for (int i = 0; i < 51; ++i)
                {
                    modelInput[i] = blendshapes[i];
                }
            }
        }

        void SetRenderingResult()
        {
            targetRenderer.SetSplatData(
                ortRunner.vertexSize,
                ortRunner.LastOutput
            );
        }

        void LateUpdate()
        {
            if (target != null && ortRunner.IsReady)
            {
                UpdateBlendshapeInput();
                SetRenderingResult();
            }
        }

        void OnGUI()
        {
            ortRunner.OnGUI();
        }
    }
}
