using System;
using System.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace DPDFNetUnity
{
    /// <summary>
    /// 流式 STFT 预处理，对应 Python STFTStreamingPreprocess。
    /// 每送入 hop_size 个单声道样本，产出 [freq_bins*2] 交错（real,imag）频谱。
    /// </summary>
    public class StreamingStft
    {
        private readonly int winLen;
        private readonly int hopSize;
        private readonly float[] window;
        private readonly float[] buffer;
        private readonly Complex[] fftBuf;

        public StreamingStft(int winLen, int hopSize, float[] window)
        {
            this.winLen = winLen;
            this.hopSize = hopSize;
            this.window = window;
            this.buffer = new float[winLen];
            this.fftBuf = new Complex[winLen];
        }

        public void Reset() => Array.Clear(buffer, 0, buffer.Length);

        public void Process(float[] hop, float[] outSpecRI)
        {
            if (hop.Length != hopSize)
                throw new ArgumentException($"expected {hopSize} samples, got {hop.Length}");

            // buffer = concat(buffer[hop:], hop)
            Array.Copy(buffer, hopSize, buffer, 0, winLen - hopSize);
            Array.Copy(hop, 0, buffer, winLen - hopSize, hopSize);

            for (int i = 0; i < winLen; i++)
                fftBuf[i] = new Complex(buffer[i] * window[i], 0.0);

            // numpy.fft.rfft 不缩放；Math.NET Forward(Default) 同样不缩放。
            Fourier.Forward(fftBuf, FourierOptions.Default);

            int freq = winLen / 2 + 1;
            for (int f = 0; f < freq; f++)
            {
                outSpecRI[f * 2] = (float)fftBuf[f].Real;
                outSpecRI[f * 2 + 1] = (float)fftBuf[f].Imaginary;
            }
        }
    }
}
