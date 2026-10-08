using System;

namespace DPDFNetUnity
{
    /// <summary>
    /// Lanczos 窗 sinc 重采样器，任意比率，幅度保持。
    /// 用于将麦克风原生采样率与模型 16 kHz 互转（多数设备支持 16k，可跳过）。
    /// </summary>
    public class AudioResampler
    {
        private readonly int inRate;
        private readonly int outRate;
        private readonly double scale; // out / in
        private const int A = 8;       // Lanczos 半宽

        public AudioResampler(int inRate, int outRate)
        {
            this.inRate = inRate;
            this.outRate = outRate;
            this.scale = (double)outRate / inRate;
        }

        public float[] Resample(float[] input)
        {
            int outLen = (int)Math.Ceiling(input.Length * scale);
            float[] output = new float[outLen];
            for (int oi = 0; oi < outLen; oi++)
            {
                double inPos = oi / scale;
                int center = (int)Math.Floor(inPos);
                double sum = 0.0, wsum = 0.0;
                for (int t = -A; t <= A; t++)
                {
                    int idx = center + t;
                    if (idx < 0 || idx >= input.Length) continue;
                    double x = inPos - idx;
                    double w = Lanczos(x, A);
                    sum += input[idx] * w;
                    wsum += w;
                }
                output[oi] = wsum > 0 ? (float)(sum / wsum) : 0f;
            }
            return output;
        }

        private static double Lanczos(double x, int a)
        {
            if (x == 0) return 1.0;
            if (x <= -a || x >= a) return 0.0;
            double px = Math.PI * x;
            return a * Math.Sin(px) * Math.Sin(px / a) / (px * px);
        }
    }
}
