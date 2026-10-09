using System;
using System.IO;
using UnityEngine;

namespace DPDFNetUnity
{
    /// <summary>
    /// 16 kHz 音频文件离线降噪示例（不依赖任何额外包，仅需 onnxruntime 包）。
    /// 流程：读取 WAV -> 逐 hop 调 DpdfNetProcessor.ProcessFrame -> 写回增强后的 WAV。
    /// 适用于桌面 / Editor 下以本地文件路径处理；移动端请将模型与音频放在可读写路径
    /// （如 Application.persistentDataPath），因 StreamingAssets 在 Android 内打包为 APK
    /// 资源，InferenceSession 无法直接以路径读取。
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

            // 2) 加载模型并建立处理器（DpdfNetProcessor 负责释放 session）
            var session = new OnnxRuntimeSession(modelPath);
            var cfg = DpdfNetModelConfig.FromSession(session);
            using var proc = new DpdfNetProcessor(cfg, session);

            int hop = proc.HopLength;
            int nfft = cfg.n_fft;

            // 3) 逐 hop 处理。末尾不足一个 hop 的残差用 0 补齐成整帧，
            //    处理后再裁掉（见下方 clean 区间），不丢有效样本。
            int frames = (samples.Length + hop - 1) / hop;
            int padded = frames * hop;
            float[] outBuf = new float[padded];

            for (int f = 0; f < frames; f++)
            {
                float[] hopIn = new float[hop];
                int src = f * hop;
                int n = Math.Min(hop, samples.Length - src);
                if (n > 0) Array.Copy(samples, src, hopIn, 0, n); // 余下补 0（默认零初始化）

                float[] hopOut = proc.ProcessFrame(hopIn);
                Array.Copy(hopOut, 0, outBuf, f * hop, hop);
            }

            // 4) 裁掉流式 ISTFT 的前导斜坡（约一个窗长 n_fft）与末尾补零，
            //    得到与原始输入等长的干净结果。前导斜坡长度可调 n_fft ~ n_fft*2。
            int frontTrim = Math.Min(nfft, outBuf.Length);
            int cleanLen = Math.Max(0, samples.Length - frontTrim);
            float[] clean = new float[cleanLen];
            if (cleanLen > 0)
                Array.Copy(outBuf, frontTrim, clean, 0, cleanLen);

            // 5) 写回 WAV（16 kHz 单声道 16-bit PCM）
            WavIo.Write(outputWav, clean, 16000, 1);
            Debug.Log($"[DPDFNet] 已处理 {samples.Length} 样本 -> {outputWav}");
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