using System;
using System.IO;
using System.Numerics;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

namespace DPDFNetUnity
{
    /// <summary>
    /// 16 kHz 音频文件离线降噪示例（不依赖任何额外包，仅需 onnxruntime 包）。
    /// 严格对齐 DPDFNet 源码 onnx_model/infer_dpdfnet_onnx.py 的 enhance_file_onnx：
    ///   center=True 反射填充 STFT -> 流式 ONNX(state) -> center=True ISTFT(win^2 归一化)
    ///   -> postprocess_spec(裁 win_len*2 / 补 win_len*2) -> fit_length。
    /// 与「实时」路径（real_time_demo.py 的因果流式 STFT）不同：文件离线用 librosa 的
    /// center=True 算法，逐帧对齐与边界处理都不一样，故结果以源码为准。
    /// 适用于桌面 / Editor 下以本地文件路径处理；移动端请将模型与音频放在可读写路径
    /// （如 Application.persistentDataPath），因 StreamingAssets 在 Android 内打包为 APK 资源。
    /// </summary>
    public class DpdfNetFileExample : MonoBehaviour
    {
        private void Start()
        {
            ProcessWav(Application.streamingAssetsPath + "/dpdfnet/dpdfnet2.onnx",
                Application.dataPath + "/mix.wav",
                Application.dataPath + "/dpdfnet.wav");
        }

        /// <summary>
        /// 处理一个 16 kHz WAV 文件（单/多声道均可，自动转单声道）。
        /// 输出为同采样率的 16-bit PCM 单声道 WAV。
        /// </summary>
        /// <param name="modelPath">ONNX 模型文件路径（如 StreamingAssets/dpdfnet/dpdfnet2.onnx）</param>
        /// <param name="inputWav">输入 WAV 路径（建议 16 kHz）</param>
        /// <param name="outputWav">输出 WAV 路径</param>
        public void ProcessWav(string modelPath, string inputWav, string outputWav)
        {
            // 1) 读取 WAV -> 单声道 float 数组（归一化到 [-1,1]）
            if (!WavIo.TryRead(inputWav, out float[] samples, out int sampleRate, out _))
                throw new InvalidOperationException($"[DPDFNet] 无法读取 WAV: {inputWav}");

            if (sampleRate != 16000)
                Debug.LogWarning($"[DPDFNet] 输入采样率为 {sampleRate} Hz，DPDFNet 模型为 16 kHz 训练，建议先重采样到 16000 再处理（此处按原率处理，结果可能异常）。");

            // 2) 加载模型（using 释放，替代原先经 DpdfNetProcessor.Dispose 释放）
            using var session = new OnnxRuntimeSession(modelPath);
            var cfg = DpdfNetModelConfig.FromSession(session);

            // 3) 离线增强（严格对齐 infer_dpdfnet_onnx.enhance_file_onnx）
            float[] enhanced = EnhanceOffline(samples, session, cfg);

            // 4) 写回 WAV（16 kHz 单声道 16-bit PCM）
            WavIo.Write(outputWav, enhanced, 16000, 1);
            Debug.Log($"[DPDFNet] 已处理 {samples.Length} 样本 -> {outputWav}");
        }

        /// <summary>
        /// 离线增强，逐行对应 infer_dpdfnet_onnx.enhance_file_onnx（attn_limit 缺省关闭）：
        ///   末端补 win_len -> center=True 反射填充 STFT -> 流式 ONNX(state) ->
        ///   center=True ISTFT(win^2 归一化 OLA, 两端裁 n_fft/2) ->
        ///   postprocess_spec(裁 win_len*2, 补 win_len*2) -> fit_length。
        /// </summary>
        static float[] EnhanceOffline(float[] x, IOnnxSession session, DpdfNetModelConfig cfg)
        {
            int nFft = cfg.n_fft;            // 320
            int hop = cfg.hop_length;        // 160
            int winLen = cfg.window_length;  // 320
            float[] win = VorbisWindow.Compute(winLen);
            int half = nFft / 2;

            // 1) 末端补 win_len 个 0：np.pad(waveform, (0, win_len), mode='constant')
            int L = x.Length;
            float[] xpad = new float[L + winLen];
            Array.Copy(x, 0, xpad, 0, L);

            // 2) center=True 反射填充 n_fft//2（librosa pad_mode='reflect'）
            float[] xp = ReflectPad(xpad, half);

            // 3) center=True 分帧 STFT（librosa.stft，逐帧 rfft）
            int numFrames = 1 + (xp.Length - nFft) / hop;
            var specRI = new float[numFrames][];
            for (int t = 0; t < numFrames; t++)
            {
                var frame = new float[nFft];
                int off = t * hop;
                for (int i = 0; i < nFft; i++) frame[i] = xp[off + i] * win[i];
                specRI[t] = Rfft(frame);
            }

            // 4) 流式 ONNX（state 逐帧回传，对应 run_onnx_streaming）
            float[] state = cfg.BuildInitialState();
            var specE = new float[numFrames][];
            for (int t = 0; t < numFrames; t++)
            {
                session.Run(specRI[t], state, out var se, out var st);
                specE[t] = se;
                state = st;
            }

            // 5) center=True ISTFT（librosa.istft）：win^2 归一化 OLA，再两端裁 n_fft//2
            int expected = nFft + hop * (numFrames - 1);
            float[] y = new float[expected];
            float[] winSum = new float[expected];
            for (int t = 0; t < numFrames; t++)
            {
                float[] td = Irfft(specE[t], nFft);
                int off = t * hop;
                for (int i = 0; i < nFft; i++)
                {
                    y[off + i] += td[i] * win[i];
                    winSum[off + i] += win[i] * win[i];
                }
            }
            for (int i = 0; i < expected; i++)
                y[i] = winSum[i] > 1e-12f ? y[i] / winSum[i] : 0f;

            int waveELen = expected - 2 * half;
            float[] waveE = new float[waveELen];
            Array.Copy(y, half, waveE, 0, waveELen);

            // 6) postprocess_spec：裁掉前 win_len*2，末尾补 win_len*2 个 0（长度不变）
            int drop = winLen * 2;
            float[] post = new float[waveELen];
            if (waveELen > drop)
                Array.Copy(waveE, drop, post, 0, waveELen - drop);
            // 末尾 drop 个 0 由零初始化

            // 7) fit_length：对齐到原始长度
            float[] outArr = new float[L];
            int copyLen = Math.Min(L, post.Length);
            Array.Copy(post, 0, outArr, 0, copyLen);
            return outArr;
        }

        /// <summary>np.pad(x, pad, mode='reflect')：不含边缘重复的对称反射填充。</summary>
        static float[] ReflectPad(float[] x, int pad)
        {
            if (pad <= 0) return (float[])x.Clone();
            int L = x.Length;
            float[] r = new float[L + 2 * pad];
            for (int i = 0; i < L; i++) r[pad + i] = x[i];
            for (int i = 0; i < pad; i++) r[i] = x[pad - i];             // x[pad]..x[1]
            for (int i = 0; i < pad; i++) r[pad + L + i] = x[L - 2 - i]; // x[L-2]..x[L-1-pad]
            return r;
        }

        /// <summary>numpy.fft.rfft：前向不缩放，返回 [F*2] 交错 real/imag。</summary>
        static float[] Rfft(float[] x)
        {
            int n = x.Length;
            var buf = new Complex[n];
            for (int i = 0; i < n; i++) buf[i] = new Complex(x[i], 0.0);
            Fourier.Forward(buf, FourierOptions.Default);
            int F = n / 2 + 1;
            float[] ri = new float[F * 2];
            for (int f = 0; f < F; f++)
            {
                ri[f * 2] = (float)buf[f].Real;
                ri[f * 2 + 1] = (float)buf[f].Imaginary;
            }
            return ri;
        }

        /// <summary>numpy.fft.irfft：逆变换含 1/N，由 [F*2] 交错重建 n 点实信号。</summary>
        static float[] Irfft(float[] specRI, int n)
        {
            int F = n / 2 + 1;
            var full = new Complex[n];
            for (int f = 0; f < F; f++) full[f] = new Complex(specRI[f * 2], specRI[f * 2 + 1]);
            full[0] = new Complex(full[0].Real, 0.0);
            full[F - 1] = new Complex(full[F - 1].Real, 0.0);
            for (int k = 1; k < F - 1; k++)
                full[n - k] = new Complex(full[k].Real, -full[k].Imaginary);
            Fourier.Inverse(full, FourierOptions.Default);
            float[] td = new float[n];
            for (int i = 0; i < n; i++) td[i] = (float)full[i].Real;
            return td;
        }
    }

    /// <summary>
    /// 极简 WAV 读写（仅依赖 System.IO / UnityEngine.Debug），覆盖常见 PCM 格式。
    /// 读：自动转为单声道 float[-1,1]；写：16-bit PCM（单/多声道）。
    /// 仅作示例用途；生产环境可替换为 NAudio / Unity AudioClip 等成熟实现。
    /// </summary>
    public class WavIo
    {
        public static bool TryRead(string path, out float[] mono, out int sampleRate, out int channels)
        {
            mono = null; sampleRate = 0; channels = 0;
            try
            {
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
                using var br = new BinaryReader(fs);

                if (new string(br.ReadChars(4)) != "RIFF") return false;
                br.ReadInt32(); // chunk size
                if (new string(br.ReadChars(4)) != "WAVE") return false;

                int fmtSampleRate = 0, fmtChannels = 0, fmtBits = 0, audioFormat = 0;
                byte[] data = null;
                while (fs.Position < fs.Length)
                {
                    string id = new string(br.ReadChars(4));
                    int size = br.ReadInt32();
                    if (id == "fmt ")
                    {
                        audioFormat = br.ReadInt16();
                        fmtChannels = br.ReadInt16();
                        fmtSampleRate = br.ReadInt32();
                        br.ReadInt32(); // byteRate
                        br.ReadInt16(); // blockAlign
                        fmtBits = br.ReadInt16();
                        int skip = size - 16;
                        if (skip > 0) br.ReadBytes(skip);
                    }
                    else if (id == "data")
                    {
                        data = br.ReadBytes(size);
                    }
                    else
                    {
                        br.ReadBytes(size);
                    }
                }
                if (data == null || fmtSampleRate == 0) return false;

                sampleRate = fmtSampleRate;
                channels = fmtChannels;
                mono = Decode(data, audioFormat, fmtChannels, fmtBits);
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[DPDFNet] WAV 读取失败: {e.Message}");
                return false;
            }
        }

        private static float[] Decode(byte[] data, int audioFormat, int channels, int bits)
        {
            int bytesPerSample = bits / 8;
            int totalSamples = data.Length / (bytesPerSample * channels);
            float[] outBuf = new float[totalSamples];
            int offset = 0;
            for (int i = 0; i < totalSamples; i++)
            {
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                {
                    int pos = offset + c * bytesPerSample;
                    float v = 0f;
                    if (audioFormat == 3) // IEEE float
                    {
                        if (bits == 32) v = BitConverter.ToSingle(data, pos);
                        else if (bits == 64) v = (float)BitConverter.ToDouble(data, pos);
                    }
                    else if (bits == 16)
                    {
                        v = BitConverter.ToInt16(data, pos) / 32768f;
                    }
                    else if (bits == 24)
                    {
                        int s = (data[pos + 2] << 16) | (data[pos + 1] << 8) | data[pos];
                        if ((s & 0x800000) != 0) s |= unchecked((int)0xFF000000);
                        v = s / 8388608f;
                    }
                    else if (bits == 32)
                    {
                        v = BitConverter.ToInt32(data, pos) / 2147483648f;
                    }
                    else if (bits == 8)
                    {
                        v = (data[pos] - 128) / 128f;
                    }
                    sum += v;
                }
                outBuf[i] = sum / channels;
                offset += bytesPerSample * channels;
            }
            return outBuf;
        }

        public static void Write(string path, float[] samples, int sampleRate, int channels)
        {
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);

            int dataLen = samples.Length * 2 * channels; // 16-bit
            bw.Write("RIFF".ToCharArray());
            bw.Write(36 + dataLen);
            bw.Write("WAVE".ToCharArray());

            bw.Write("fmt ".ToCharArray());
            bw.Write(16);                       // PCM fmt chunk 大小
            bw.Write((short)1);                 // PCM
            bw.Write((short)channels);
            bw.Write(sampleRate);
            bw.Write(sampleRate * channels * 2); // byteRate
            bw.Write((short)(channels * 2));     // blockAlign
            bw.Write((short)16);                // bitsPerSample

            bw.Write("data".ToCharArray());
            bw.Write(dataLen);

            if (channels == 1)
            {
                foreach (float s in samples)
                    bw.Write(Clamp16(s));
            }
            else
            {
                // 示例简化：samples 视为单声道，写入第 0 声道，其余声道置 0
                for (int i = 0; i < samples.Length; i++)
                {
                    bw.Write(Clamp16(samples[i]));
                    for (int c = 1; c < channels; c++) bw.Write((short)0);
                }
            }
        }

        private static short Clamp16(float s)
        {
            float v = Math.Max(-1f, Math.Min(1f, s));
            return (short)(v * 32767f);
        }
    }
}