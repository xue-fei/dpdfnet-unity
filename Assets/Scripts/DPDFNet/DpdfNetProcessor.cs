using System;

namespace DPDFNetUnity
{
    /// <summary>
    /// DPDFNet 流式降噪核心：每帧 串/并联 STFT -> ONNX(state 进/出) -> ISTFT。
    /// 与 Python real_time_demo.enhance_frame 完全对应。
    /// 复用内部缓冲区，适合在音频线程逐帧调用。
    /// </summary>
    public class DpdfNetProcessor : IDisposable
    {
        private readonly DpdfNetModelConfig cfg;
        private readonly IOnnxSession session;
        private readonly StreamingStft stft;
        private readonly StreamingIstft istft;
        private float[] state;
        private readonly float[] specBuf;
        private readonly float[] outHop;
        private readonly object lockObj = new object();

        public int HopLength => cfg.hop_length;
        public int SampleRate => cfg.sample_rate;

        public DpdfNetProcessor(DpdfNetModelConfig cfg, IOnnxSession session)
        {
            this.cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            this.session = session ?? throw new ArgumentNullException(nameof(session));
            float[] window = VorbisWindow.Compute(cfg.window_length);
            stft = new StreamingStft(cfg.n_fft, cfg.hop_length, window);
            istft = new StreamingIstft(cfg.n_fft, cfg.hop_length, window);
            state = cfg.BuildInitialState();
            specBuf = new float[cfg.freq_bins * 2];
            outHop = new float[cfg.hop_length];
        }

        /// <summary>处理一个 hop 的噪声样本，返回增强后的 hop（同一数组实例，调用方需立即使用）。</summary>
        public float[] ProcessFrame(float[] noisyHop)
        {
            lock (lockObj)
            {
                stft.Process(noisyHop, specBuf);
                session.Run(specBuf, state, out var specOut, out var stateOut);
                state = stateOut;
                istft.Process(specOut, outHop);
                return outHop;
            }
        }

        /// <summary>重置流式状态（静音间隙/切换模型时调用）。</summary>
        public void Reset()
        {
            lock (lockObj)
            {
                state = cfg.BuildInitialState();
                stft.Reset();
                istft.Reset();
            }
        }

        public void Dispose() => session?.Dispose();
    }
}
