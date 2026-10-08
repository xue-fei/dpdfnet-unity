using UnityEngine;

namespace DPDFNetUnity
{
    /// <summary>
    /// Vorbis（Tremolo）分析/合成窗，满足 Princen-Bradley 条件：
    /// 同一窗用于分析与合成时完美重建。
    /// 与 Python vorbis_window 逐元素一致。
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
}
