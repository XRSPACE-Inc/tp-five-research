using UnityEngine;
using Microsoft.ML.OnnxRuntime.Unity;
using GaussianSplatting.Runtime;

namespace GaussianMe
{
    public class Runner : MonoBehaviour
    {
        [SerializeField]
        OrtAsset model;
        public GaussianSplatRenderer target;
        public TextAsset colorAsset;
        UDPListener listener;
        public string ip = "127.0.0.1";
        public int port = 12499;

        public OrtRunner ortRunner = new OrtRunner();

        void OnEnable()
        {
            // prints maximum graphics buffer size, in megabytes
            var maxSizeMb = SystemInfo.maxGraphicsBufferSize / 1024 / 1024;
            Debug.Log($"Maximum graphics buffer size is {maxSizeMb} MB");

            ortRunner.CreateONNXSession(model.bytes, 51, destroyCancellationToken);

            if (colorAsset != null)
            {
                target.SetColor(colorAsset.bytes);
            }
        }

        public void OnDestroy()
        {
            listener?.Dispose();
            listener = null;
        }

        void SetSplatData()
        {
            target.SetSplatData(
                ortRunner.vertexSize,
                ortRunner.LastOutput
            );
        }

        void Update()
        {
            if (!hasData)
            {
                return;
            }

            SetSplatData();
        }

        void FirstFrame(byte[] bytes)
        {
            RunSession(bytes);

            listener.onDataIn = RunSession;
            Debug.Log($"FirstFrame Done!");
        }

        bool hasData = false;
        void RunSession(byte[] bytes)
        {
            ortRunner.RunSession(bytes);
            hasData = true;
        }

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
        void OnGUI()
        {
            GUILayout.BeginVertical();
            GUILayout.Space(300f);
            float fps = 1000f / (float)ortRunner.lastTime;
            GUILayout.Label($"FPS: {fps:0.00} (process time: {ortRunner.lastTime:0.00}ms)", LabelStyle);
            GUILayout.EndVertical();
        }
    }
}
