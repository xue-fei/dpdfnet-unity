using System;

namespace DPDFNetUnity
{
    /// <summary>
    /// DPDFNet 模型配置与初始 state 重建（零依赖版）。
    ///
    /// 由于 onnxsim 剥离了 ONNX 自定义 metadata，原本需要一份旁路 JSON 提供
    /// sample_rate / hop / 初始 state。现改为：运行时从 ONNX session 的
    /// FreqBins / StateSize 推导 DSP 参数，并用与 Python onnx_model/layers.py
    /// 完全一致的公式重建初始 state（仅 ErbNorm / SpecNorm 两个切片非零，其余全 0），
    /// 从而彻底去掉 JSON 文件依赖。
    /// </summary>
    public class DpdfNetModelConfig
    {
        // --- DSP 参数（16k 模型族固定，仅 n_fft/window_length 随 freq_bins 变化） ---
        public int sample_rate = 16000;
        public int n_fft = 320;
        public int hop_length = 160;
        public int freq_bins = 161;
        public string window_type = "vorbis";
        public int window_length = 320;
        public int state_size;

        // 归一化层固定参数（与 Python onnx_model/dpdfnet.py 构造 DPDFNet 时一致）
        public const int NbErb = 32;            // nb_erb（固定）
        public const int FreqDf = 4800;          // freq_df
        public const int SamplerateHalf = 8000;  // samplerate // 2 = 16000//2

        private DpdfNetModelConfig() { }

        /// <summary>
        /// 从 ONNX session 推导配置。
        /// freq_bins / state_size 直接来自模型输入形状；
        /// n_fft / window_length 由 freq_bins 推出（= 2*(F-1)）；
        /// hop / sample_rate 为 16k 模型族固定值。
        /// </summary>
        public static DpdfNetModelConfig FromSession(IOnnxSession session)
        {
            int fb = session.FreqBins;
            return new DpdfNetModelConfig
            {
                freq_bins = fb,
                state_size = session.StateSize,
                n_fft = 2 * (fb - 1),
                window_length = 2 * (fb - 1),
                hop_length = 160,
                sample_rate = 16000,
                window_type = "vorbis",
            };
        }

        /// <summary>
        /// 重建初始 state 向量：zeros(state_size)，前 NbErb 填 ErbNorm mu0，
        /// 紧接着填 SpecNorm s0，其余全 0。与 Python DPDFNet.initial_state() 的
        /// chunk 布局（erb_norm → spec_norm → enc/erb_dec/df_dec/mask/df_op 全 0）
        /// 完全一致。
        ///
        /// ErbNorm:  init_vals=[-60, -90], num_feat=NbErb
        ///           mu[i] = -60 + i * (-30 / (NbErb-1))
        /// SpecNorm: init_vals=[0.001, 0.0001], num_feat=NbDf
        ///           s[i]  = 0.001 + i * (-0.0009 / (NbDf-1))
        ///           NbDf  = int((freq_df / (samplerate//2)) * freq_bins)
        /// </summary>
        public float[] BuildInitialState()
        {
            float[] s = new float[state_size];

            // ErbNorm mu0
            double erbStep = (-90.0 - (-60.0)) / (NbErb - 1);
            for (int i = 0; i < NbErb; i++)
                s[i] = (float)(-60.0 + i * erbStep);

            // SpecNorm s0
            int nbDf = (int)((double)FreqDf / SamplerateHalf * freq_bins);
            double specStep = (0.0001 - 0.001) / (nbDf - 1);
            for (int i = 0; i < nbDf; i++)
                s[NbErb + i] = (float)(0.001 + i * specStep);

            return s;
        }
    }

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
