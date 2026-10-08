using System;
using System.Collections.Generic;
using UnityEngine;

namespace DPDFNetUnity
{
    /// <summary>
    /// 实时麦克风降噪 Demo：挂在带 AudioSource 的 GameObject 上。
    /// 麦克风音频经 OnAudioFilterRead 逐帧送入 DPDFNetProcessor，增强后回写。
    ///
    /// 关键：OnAudioFilterRead 的 data 处于 Unity 输出采样率(AudioSettings.outputSampleRate)，
    /// 不一定是模型率(16000)。因此重采样决策基于 outputSampleRate，而非麦克风 clip 的 native 率。
    /// 若直接把 48k 的 data 当 16k 喂给模型，模型会把分布外输入当成噪声抑制→静音。
    ///
    /// 建议戴耳机避免啸叫。若听不到声音，先把 monitorBypass 设为 true：
    ///   - 仍听不到 → 麦克风采集/输出路由问题（检查权限、AudioListener、是否戴耳机）。
    ///   - 能听到 → 采集正常，问题在模型/重采样（此时应已修复）。
    /// </summary>
    [RequireComponent(typeof(AudioSource))]
    public class DpdfNetMicrophoneDemo : MonoBehaviour
    {
        [Header("模型")]
        [Tooltip("StreamingAssets/dpdfnet 下的模型名（不含扩展名）")] public string modelName = "dpdfnet2";
        [Tooltip("请求麦克风采样率，默认 16000 与模型一致")] public int captureSampleRate = 16000;

        [Header("输出")]
        [Range(0f, 1f)] public float playbackMix = 0f;   // 0=纯增强, 1=纯原始
        public bool monitorBypass = false;                // true=直出原始麦克风(诊断用)
        public bool enableAgc = true;

        [Header("调试")]
        public float lastInferenceMs;
        [Tooltip("每 N 秒打印一次诊断(RMS/帧耗时)")] public float diagInterval = 0.5f;

        private DpdfNetProcessor processor;
        private AudioClip micClip;
        private string micDevice;
        private int micRate;
        private int dspRate;
        private bool needsResample;
        private AudioResampler inResampler;
        private AudioResampler outResampler;

        private readonly Queue<float> inputQueue = new Queue<float>();
        private readonly Queue<float> outputQueue = new Queue<float>();
        private float agcGain = 1f;

        // 诊断累计
        private int filterCalls;
        private float diagTimer;
        private float inRmsSum, outRmsSum;
        private int rmsFrames;
        private bool loggedModelError;

        void Start() => Initialize();
        void OnDisable() => Cleanup();
        void OnDestroy() => Cleanup();

        [ContextMenu("Reinitialize")]
        public void Initialize()
        {
            Cleanup();

            string jsonPath = System.IO.Path.Combine(Application.streamingAssetsPath, "dpdfnet", modelName + ".json");
            if (!System.IO.File.Exists(jsonPath))
            {
                Debug.LogError($"[DPDFNet] 未找到旁路元数据: {jsonPath}。请先运行 generate_unity_sidecar_meta.py。");
                return;
            }
            var cfg = DpdfNetModelConfig.Load(System.IO.File.ReadAllText(jsonPath));

            string onnxPath = System.IO.Path.Combine(Application.streamingAssetsPath, "dpdfnet", modelName + ".onnx");
            IOnnxSession session;
            try
            {
                session = new OnnxRuntimeSession(onnxPath);
            }
            catch (Exception e)
            {
                Debug.LogError($"[DPDFNet] 加载 ONNX 失败（是否已安装 Microsoft.ML.OnnxRuntime？）：{e.Message}");
                return;
            }
            processor = new DpdfNetProcessor(cfg, session);

            if (Microphone.devices.Length == 0)
            {
                Debug.LogError("[DPDFNet] 未检测到麦克风设备。");
                return;
            }
            micDevice = Microphone.devices[0];
            micClip = Microphone.Start(micDevice, true, 2, captureSampleRate);
            micRate = micClip.frequency;
            dspRate = AudioSettings.outputSampleRate;

            // 重采样必须基于 data 的真实速率(outputSampleRate)，否则 48k data 当 16k 喂模型会静音。
            needsResample = dspRate != cfg.sample_rate;
            if (needsResample)
            {
                inResampler = new AudioResampler(dspRate, cfg.sample_rate);
                outResampler = new AudioResampler(cfg.sample_rate, dspRate);
                Debug.Log($"[DPDFNet] 重采样 {dspRate} -> {cfg.sample_rate} -> {dspRate}");
            }
            else
            {
                Debug.Log($"[DPDFNet] 无需重采样 (dspRate={dspRate}, model={cfg.sample_rate})");
            }

            var src = GetComponent<AudioSource>();
            src.clip = micClip;
            src.loop = true;
            src.Play();
            Debug.Log($"[DPDFNet] micRate={micRate}, dspRate={dspRate}, captureSampleRate={captureSampleRate}");
        }

        void OnAudioFilterRead(float[] data, int channels)
        {
            filterCalls++;
            if (processor == null) return;
            int frameLen = data.Length / channels;

            // 1. 取通道 0 输入（播放率）
            float[] pcmIn = new float[frameLen];
            for (int i = 0; i < frameLen; i++) pcmIn[i] = data[i * channels];

            // 诊断：输入 RMS
            double inSq = 0;
            for (int i = 0; i < frameLen; i++) inSq += pcmIn[i] * pcmIn[i];
            float inRms = Mathf.Sqrt((float)(inSq / frameLen) + 1e-12f);

            if (monitorBypass)
            {
                // 直出原始麦克风，用于隔离采集问题
                float[] block0 = new float[frameLen];
                for (int i = 0; i < frameLen; i++)
                    for (int c = 0; c < channels; c++)
                        data[i * channels + c] = pcmIn[i];
                AccumDiag(inRms, inRms);
                return;
            }

            // 2. 转模型率（如需）
            float[] modelIn = needsResample ? inResampler.Resample(pcmIn) : pcmIn;

            // 3. 入队并整 hop 处理
            for (int i = 0; i < modelIn.Length; i++) inputQueue.Enqueue(modelIn[i]);
            int hop = processor.HopLength;
            var produced = new List<float>(modelIn.Length);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (inputQueue.Count >= hop)
            {
                float[] hopBuf = new float[hop];
                for (int i = 0; i < hop; i++) hopBuf[i] = inputQueue.Dequeue();
                float[] enh;
                try
                {
                    enh = processor.ProcessFrame(hopBuf);
                }
                catch (Exception e)
                {
                    if (!loggedModelError)
                    {
                        Debug.LogError($"[DPDFNet] 推理异常：{e.Message}");
                        loggedModelError = true;
                    }
                    enh = hopBuf; // 异常时原样透传，避免静音
                }
                for (int i = 0; i < hop; i++) produced.Add(enh[i]);
            }
            if (produced.Count > 0)
                lastInferenceMs = (float)(sw.ElapsedTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency);

            // 4. 转回播放率并入输出队列
            float[] toOutput = produced.ToArray();
            if (needsResample && toOutput.Length > 0)
                toOutput = outResampler.Resample(toOutput);
            for (int i = 0; i < toOutput.Length; i++) outputQueue.Enqueue(toOutput[i]);

            // 5. 组装播放块 + 块级 AGC（对应 Python apply_output_agc）
            float[] block = new float[frameLen];
            for (int i = 0; i < frameLen; i++)
            {
                float enhanced = outputQueue.Count > 0 ? outputQueue.Dequeue() : (i > 0 ? block[i - 1] : 0f);
                block[i] = (1f - playbackMix) * enhanced + playbackMix * pcmIn[i];
            }
            if (enableAgc)
            {
                double sumSq = 0;
                for (int i = 0; i < frameLen; i++) sumSq += block[i] * block[i];
                float rms = Mathf.Sqrt((float)(sumSq / frameLen) + 1e-12f);
                const float targetRms = 0.12f, floor = 1e-3f, minGain = 0.25f, maxGain = 8f;
                float desired = Mathf.Clamp(targetRms / Mathf.Max(rms, floor), minGain, maxGain);
                agcGain = Mathf.Lerp(agcGain, desired, 0.1f);
                for (int i = 0; i < frameLen; i++) block[i] = Mathf.Clamp(block[i] * agcGain, -1f, 1f);
            }
            else
            {
                for (int i = 0; i < frameLen; i++) block[i] = Mathf.Clamp(block[i], -1f, 1f);
            }

            // 6. 写回（所有通道）
            for (int i = 0; i < frameLen; i++)
                for (int c = 0; c < channels; c++)
                    data[i * channels + c] = block[i];

            // 诊断：输出 RMS
            double outSq = 0;
            for (int i = 0; i < frameLen; i++) outSq += block[i] * block[i];
            float outRms = Mathf.Sqrt((float)(outSq / frameLen) + 1e-12f);
            AccumDiag(inRms, outRms);
        }

        private void AccumDiag(float inRms, float outRms)
        {
            inRmsSum += inRms;
            outRmsSum += outRms;
            rmsFrames++;
        }

        void Update()
        {
            diagTimer += Time.deltaTime;
            if (diagTimer >= diagInterval && rmsFrames > 0)
            {
                float avgIn = inRmsSum / rmsFrames;
                float avgOut = outRmsSum / rmsFrames;
                Debug.Log($"[DPDFNet] calls={filterCalls} inRMS={avgIn:F4} outRMS={avgOut:F4} infer={lastInferenceMs:F2}ms needsResample={needsResample} qIn={inputQueue.Count} qOut={outputQueue.Count}");
                inRmsSum = outRmsSum = 0f;
                rmsFrames = 0;
                diagTimer = 0f;
            }
        }

        private void Cleanup()
        {
            if (micDevice != null && micClip != null && Microphone.IsRecording(micDevice))
                Microphone.End(micDevice);
            processor?.Dispose();
            processor = null;
            micDevice = null;
            micClip = null;
            inputQueue.Clear();
            outputQueue.Clear();
        }
    }
}
