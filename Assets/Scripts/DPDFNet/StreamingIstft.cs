using System;
using System.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace DPDFNetUnity
{
    /// <summary>
    /// 流式 ISTFT 后处理（重叠相加），对应 Python ISTFTStreamingPostprocess。
    /// 输入 [freq_bins*2] 交错（real,imag），输出 hop_size 个时域样本。
    /// </summary>
    public class StreamingIstft
    {
        private readonly int winLen;
        private readonly int hopSize;
        private readonly float[] window;
        private readonly Complex[] full;
        private readonly float[] newBuf;
        private float[] olaBuffer;

        public StreamingIstft(int winLen, int hopSize, float[] window)
        {
            this.winLen = winLen;
            this.hopSize = hopSize;
            this.window = window;
            this.full = new Complex[winLen];
            this.newBuf = new float[winLen];
            this.olaBuffer = new float[winLen];
        }

        public void Reset() => Array.Clear(olaBuffer, 0, olaBuffer.Length);

        public void Process(float[] specRI, float[] outHop)
        {
            int freq = winLen / 2 + 1;

            // 由 [freq] 复频谱重建 Hermitian 对称的全频谱。
            for (int f = 0; f < freq; f++)
                full[f] = new Complex(specRI[f * 2], specRI[f * 2 + 1]);
            // DC 与 Nyquist 必须为实数，保证输出纯实数（与 numpy.irfft 一致）。
            full[0] = new Complex(full[0].Real, 0.0);
            full[freq - 1] = new Complex(full[freq - 1].Real, 0.0);
            for (int k = 1; k < freq - 1; k++)
                full[winLen - k] = new Complex(full[k].Real, -full[k].Imaginary);

            // numpy.fft.irfft 含 1/N；Math.NET Inverse(Default) 同样含 1/N。
            Fourier.Inverse(full, FourierOptions.Default);

            for (int i = 0; i < winLen; i++)
            {
                float sample = (float)full[i].Real * window[i];
                if (i < winLen - hopSize)
                    newBuf[i] = olaBuffer[hopSize + i] + sample;
                else
                    newBuf[i] = sample; // 后半段为 0，无需相加
            }

            for (int i = 0; i < hopSize; i++)
                outHop[i] = newBuf[i];
            // 将完整 OLA 缓冲复制回 olaBuffer（newBuf 为复用 scratch，不可直接交换）
            Array.Copy(newBuf, 0, olaBuffer, 0, winLen);
        }
    }
}
