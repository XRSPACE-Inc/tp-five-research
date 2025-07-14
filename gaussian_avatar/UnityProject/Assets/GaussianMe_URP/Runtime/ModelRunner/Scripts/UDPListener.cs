using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class UDPListener
{
    UdpClient udpSocket;
    CancellationTokenSource cancellationTokenSource;

    public UDPListener(string ip, int port)
    {
        cancellationTokenSource = new CancellationTokenSource();
        CancellationToken token = cancellationTokenSource.Token;
        Task.Factory.StartNew(() => {
            udpSocket = new UdpClient(port);
            var anyIp = new IPEndPoint(IPAddress.Parse(ip), port);
            Debug.Log($"UDP Listener initialized: {anyIp}");
            while (true)
            {
                try
                {
                    var data = udpSocket.Receive(ref anyIp);
                    ProcessData(data);
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

    ~UDPListener()
    {
        cancellationTokenSource?.Cancel();
        cancellationTokenSource = null;
    }

    public void Dispose()
    {
        cancellationTokenSource?.Cancel();
        cancellationTokenSource = null;
    }

    public System.Action<byte[]> onDataIn;
    public void ProcessData(byte[] bytes)
    {
        onDataIn?.Invoke(bytes);
    }
}
