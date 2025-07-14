using Unity.Collections;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using System.Collections.Generic;
using GaussianSplatting.Runtime;
using System.Collections;

namespace GaussianMe
{
    public class ModelTester : MonoBehaviour
    {
        public GaussianSplatRenderer target;
        public TextAsset colorAsset;
        public TextAsset testInputFile;
        public bool ict;

        [SerializeField]
        private OrtAsset model;

        float[] LoadTestData(string text)
        {
            var list = new List<float>();
            var lines = text.Split('\n');
            foreach (var line in lines)
            {
                var eachStr = line.Split(' ');
                foreach (var str in eachStr)
                {
                    if (float.TryParse(str, out var val))
                    {
                        list.Add(val);
                    }
                }
            }
            return list.ToArray();
        }

        IEnumerator Start()
        {
            var runner = new OrtRunner(ict ? 14480 : 13453);
            runner.CreateONNXSession(model.bytes, ict ? 53 : 51, destroyCancellationToken);

            var inputArr = LoadTestData(testInputFile.text);

            while (!runner.IsReady)
            {
                yield return null;
            }

            runner.RunSession(inputArr);

            target.SetSplatData(
                runner.vertexSize,
                runner.RunSession(inputArr)
            );
            if (colorAsset != null)
            {
                target.SetColor(colorAsset.bytes);
            }
            CSDebugger.Instance.PrintAfterFrame = 30;
        }
    }
}
