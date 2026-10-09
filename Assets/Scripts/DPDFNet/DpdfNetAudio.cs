using System;
using System.Numerics;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

namespace DPDFNetUnity
{
    /// <summary>
    /// Vorbis（Tremolo）分析/合成窗，满足 Princen-Bradley 条件：
    /// 同一窗用于分析与合成时完美重建（w[n]^2 + w[n+hop]^2 == 1，50% 重叠）。
    /// 与 Python vorbis_window / make_stft_config 逐元素一致。
    /// </summary>
    public static class VorbisWindow
    {
        public static float[] Compute(int windowLength)
        {
            float[] w = new float[windowLength];
            float half = windowLength * 0.5f;
            for (int i = 0; i < windowLength; i++)
            {
                float s = Mathf.Sin(0.5f * Mathf.PI * (i + 0.5f) / half);
                w[i] = Mathf.Sin(0.5f * Mathf.PI * s * s);
            }
            return w;
        }
    }

    /// <summary>
    /// STFT/ISTFT 数学核，等价 numpy.fft.rfft / numpy.fft.irfft：
    /// Math.NET Fourier.Forward(Default) 不缩放 = np.fft.rfft；
    /// Fourier.Inverse(Default) 含 1/N = np.fft.irfft（DC/Nyquist 虚部按 numpy 语义置 0）。
    /// 提供「scratch 复用」无分配版与便捷分配版两组重载，供流式处理器与离线示例共用。
    /// </summary>
    public static class StftMath
    {
        /// <summary>
        /// x[0..n) -> outRI[(n/2+1)*2] 交错（real, imag），前向不缩放。
        /// scratch.Length 必须 == n（Math.NET 对整数组做变换），用作工作区避免逐帧分配。
        /// </summary>
        public static void Rfft(float[] x, int n, Complex[] scratch, float[] outRI)
        {
            for (int i = 0; i < n; i++)
                scratch[i] = new Complex(x[i], 0.0);

            Fourier.Forward(scratch, FourierOptions.Default);

            int freq = n / 2 + 1;
            for (int f = 0; f < freq; f++)
            {
                outRI[f * 2] = (float)scratch[f].Real;
                outRI[f * 2 + 1] = (float)scratch[f].Imaginary;
            }
        }

        /// <summary>
        /// specRI[(n/2+1)*2] 交错 -> out[0..n)，逆变换含 1/N。
        /// 由 [freq] 半谱重建 Hermitian 对称全谱；DC 与 Nyquist 虚部置 0（与 numpy.irfft 一致，
        /// 保证输出纯实数）。scratch.Length 必须 == n。
        /// </summary>
        public static void Irfft(float[] specRI, int n, Complex[] scratch, float[] outTime)
        {
            int freq = n / 2 + 1;
            for (int f = 0; f < freq; f++)
                scratch[f] = new Complex(specRI[f * 2], specRI[f * 2 + 1]);

            scratch[0] = new Complex(scratch[0].Real, 0.0);
            scratch[freq - 1] = new Complex(scratch[freq - 1].Real, 0.0);
            for (int k = 1; k < freq - 1; k++)
                scratch[n - k] = new Complex(scratch[k].Real, -scratch[k].Imaginary);

            Fourier.Inverse(scratch, FourierOptions.Default);

            for (int i = 0; i < n; i++)
                outTime[i] = (float)scratch[i].Real;
        }

        /// <summary>便捷版：x -> [F*2] 交错 real/imag（内部临时分配，适合离线逐帧调用）。</summary>
        public static float[] Rfft(float[] x)
        {
            int n = x.Length;
            var outRI = new float[(n / 2 + 1) * 2];
            Rfft(x, n, new Complex[n], outRI);
            return outRI;
        }

        /// <summary>便捷版：[F*2] 交错 -> n 点实信号（内部临时分配，适合离线逐帧调用）。</summary>
        public static float[] Irfft(float[] specRI, int n)
        {
            var outTime = new float[n];
            Irfft(specRI, n, new Complex[n], outTime);
            return outTime;
        }
    }
}
