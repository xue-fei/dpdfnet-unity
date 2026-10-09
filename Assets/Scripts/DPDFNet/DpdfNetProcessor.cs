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
    /// DPDFNet 流式降噪核心，严格对齐 Python 官方流式 API
    /// `DPDFNet/package/src/dpdfnet/stream.py` 的 <c>StreamEnhancer</c>：
    ///
    /// - <see cref="Process"/>：送入任意长度的 16 kHz 单声道样本块。内部维护
    ///   win_len 输入帧缓冲 + OLA 输出缓冲，凑满一整窗（win_len=320）才推理一帧，
    ///   每帧提交 hop（160）个增强样本。返回值为本次已提交的增强样本（hop 的整数倍，
    ///   可能为空——首窗未满 win_len 前无输出，即约一个窗 20ms 的固定延迟）。
    /// - <see cref="Flush"/>：流结束时排空尾部——把不足一窗的余量补零凑整窗再处理
    ///   一帧，返回最后至多 hop 个增强样本（不重置状态）。
    /// - <see cref="Reset"/>：重置 RNN state 与内部缓冲；独立音频段之间调用，
    ///   防止 state 跨流泄漏。
    ///
    /// 与 Python 逐帧对应：windowed = in_buf[:win_len] * window → rfft → ONNX(state 进/出)
    /// → irfft * window → OLA（Vorbis 窗满足 w[n]² + w[n+hop]² = 1，50% 重叠 COLA，
    /// 每帧后前 hop 个样本已完全重建）。输入须为模型率（16 kHz）单声道；
    /// 该 C# 移植不做 stream.py 中的重采样（工程内采集/播放/模型率统一 16k）。
    /// 注意：此因果路径与离线 enhance（center=True）非逐位一致，见 stream.py docstring。
    /// </summary>
    public class DpdfNetProcessor : IDisposable
    {
        private readonly DpdfNetModelConfig cfg;
        private readonly IOnnxSession session;
        private readonly float[] window;
        private readonly int winLen;
        private readonly int hop;

        private float[] state;

        // 输入帧缓冲（StreamEnhancer._in_buf）：只增不减容量，逐帧左移消耗。
        private float[] inBuf = Array.Empty<float>();
        private int inLen;

        // OLA 输出缓冲（StreamEnhancer._out_buf），长度 win_len。
        private readonly float[] outBuf;

        // 逐帧 scratch（复用，避免每帧分配）。
        private readonly float[] specBuf;   // [freq_bins*2] 交错 real/imag
        private readonly float[] windowed;  // [win_len] 加窗帧
        private readonly float[] timeFrame; // [win_len] irfft 时域帧
        private readonly System.Numerics.Complex[] fftScratch;

        private readonly object lockObj = new object();

        public int HopLength => hop;
        public int SampleRate => cfg.sample_rate;
        public int WindowLength => winLen;

        public DpdfNetProcessor(DpdfNetModelConfig cfg, IOnnxSession session)
        {
            this.cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            this.session = session ?? throw new ArgumentNullException(nameof(session));
            winLen = cfg.window_length;
            hop = cfg.hop_length;
            window = VorbisWindow.Compute(winLen);
            outBuf = new float[winLen];
            specBuf = new float[cfg.freq_bins * 2];
            windowed = new float[winLen];
            timeFrame = new float[winLen];
            fftScratch = new System.Numerics.Complex[winLen];
            state = cfg.BuildInitialState();
        }

        /// <summary>
        /// 处理任意长度的音频块，返回本次已提交的增强样本（hop 的整数倍，可能为空）。
        /// 对应 StreamEnhancer.process()：内部按 win_len 凑窗逐帧推理，
        /// 返回数组为新分配实例，调用方可长期持有。
        /// </summary>
        public float[] Process(float[] chunk)
        {
            if (chunk == null) throw new ArgumentNullException(nameof(chunk));
            lock (lockObj)
            {
                return ProcessCore(chunk);
            }
        }

        /// <summary>
        /// 流结束时排空尾部：余量补零凑整窗再处理一帧，返回最后至多 hop 个增强样本。
        /// 对应 StreamEnhancer.flush()；不重置状态，换新流前请显式 <see cref="Reset"/>。
        /// </summary>
        public float[] Flush()
        {
            lock (lockObj)
            {
                if (inLen == 0) return Array.Empty<float>();

                // 补零到整窗（StreamEnhancer: pad = zeros(win_len - remainder)）
                float[] pad = new float[winLen - inLen];
                float[] outSamps = ProcessCore(pad);

                // 只保留真实输入对应的输出：一个 hop 的真实音频对应一个 hop 的输出。
                int realOut = Math.Min(hop, outSamps.Length);
                float[] trimmed = new float[realOut];
                Array.Copy(outSamps, 0, trimmed, 0, realOut);
                return trimmed;
            }
        }

        /// <summary>重置流式状态（RNN state + 输入/输出缓冲）。静音间隙或切换音频段时调用。</summary>
        public void Reset()
        {
            lock (lockObj)
            {
                state = cfg.BuildInitialState();
                inLen = 0;
                Array.Clear(outBuf, 0, outBuf.Length);
            }
        }

        public void Dispose() => session?.Dispose();

        // ---- 内部实现（须在 lockObj 内调用）----

        private float[] ProcessCore(float[] chunk)
        {
            if (chunk.Length == 0) return Array.Empty<float>();

            // in_buf = concat(in_buf, chunk)
            int need = inLen + chunk.Length;
            if (inBuf.Length < need)
                Array.Resize(ref inBuf, Math.Max(need, Math.Max(winLen * 2, inBuf.Length * 2)));
            Array.Copy(chunk, 0, inBuf, inLen, chunk.Length);
            inLen = need;

            // 每凑满 win_len 处理一帧，提交 hop 个样本。
            int frames = inLen >= winLen ? (inLen - winLen) / hop + 1 : 0;
            float[] outSamps = new float[frames * hop];
            int outPos = 0;

            while (inLen >= winLen)
            {
                // --- 分析 STFT（causal / center=False）：windowed = in_buf[:win_len] * window ---
                for (int i = 0; i < winLen; i++)
                    windowed[i] = inBuf[i] * window[i];
                StftMath.Rfft(windowed, winLen, fftScratch, specBuf);

                // --- ONNX 推理（state 逐帧进/出回传） ---
                session.Run(specBuf, state, out var specOut, out var stateOut);
                state = stateOut;

                // --- 每帧 ISTFT + 重叠相加：out_buf += irfft(spec_e) * window ---
                StftMath.Irfft(specOut, winLen, fftScratch, timeFrame);
                for (int i = 0; i < winLen; i++)
                    outBuf[i] += timeFrame[i] * window[i];

                // Vorbis 窗满足 50% 重叠 COLA（w[n]² + w[n+hop]² == 1），
                // 故每帧后前 hop 个样本已完全重建，可以提交。
                Array.Copy(outBuf, 0, outSamps, outPos, hop);
                outPos += hop;

                // out_buf 左移 hop、尾部清零；in_buf 消耗 hop。
                Array.Copy(outBuf, hop, outBuf, 0, winLen - hop);
                Array.Clear(outBuf, winLen - hop, hop);
                Array.Copy(inBuf, hop, inBuf, 0, inLen - hop);
                inLen -= hop;
            }

            return outSamps;
        }
    }
}
