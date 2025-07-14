using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CSDebugger : MonoBehaviour
{
    static CSDebugger instance = null;
    public static CSDebugger Instance
    {
        get
        {
            if (instance == null)
            {
                var go = new GameObject("CSDebugger");
                instance = go.AddComponent<CSDebugger>();
            }
            return instance;
        }
    }

    public class PrintData<T> : PrintDataBase
    {
        public override void PrintBuffer()
        {
            var count = buffer != null ? buffer.count : csbuffer.count;
            var data = new T[count];
            if (buffer != null)
            {
                buffer.GetData(data);
            }
            else if (csbuffer != null)
            {
                csbuffer.GetData(data);
            }

            var str = $"{name}:\n";
            for (int i = 0; i < printSize; ++i)
            {
                str += $"{data[i]} ";
                if ((i + 1) % 10 == 0) str += "\n";
            }
            str += "\n...\n";
            for (int i = count - printSize, j = 0; i < count; ++i, ++j)
            {
                str += $"{data[i]} ";
                if ((j + 1) % 10 == 0) str += "\n";
            }
            Debug.Log(str);
        }
    }

    public abstract class PrintDataBase
    {
        public ComputeBuffer csbuffer;
        public GraphicsBuffer buffer;
        public string name;
        public int printSize = 100;
        public abstract void PrintBuffer();
    }
    public List<PrintDataBase> buffers = new List<PrintDataBase>();
    public int PrintAfterFrame;

    public void Add<T>(GraphicsBuffer buffer, string name)
    {
        buffers.Add(new PrintData<T>()
        { 
            buffer = buffer,
            name = name,
        });
    }
    public void Add<T>(ComputeBuffer buffer, string name)
    {
        buffers.Add(new PrintData<T>()
        { 
            csbuffer = buffer,
            name = name,
        });
    }

    public bool DebugThisFrame;
    public void PrintNow(string label = null)
    {

        Debug.LogWarning($">>>>>>>>>>>>>> CSDebugger Print [{label}] Start");

        if (buffers.Count == 0)
        {
            Debug.LogWarning($"No buffer to print");
        }

        foreach (var buffer in buffers)
        {
            buffer.PrintBuffer();
        }

        Debug.LogWarning($"<<<<<<<<<<<<<< CSDebugger Print [{label}] Start");
    }

    void Update()
    {
        if (PrintAfterFrame > 0)
        {
            if (--PrintAfterFrame == 0)
            {
                DebugThisFrame = true;
            }
            else
            {
                DebugThisFrame = false;
            }
        }
        else
        {
            DebugThisFrame = false;
        }
    }
}
