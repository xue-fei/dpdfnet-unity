using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DPDFNetUnity
{
    /// <summary>
    /// 实时麦克风降噪 Demo（无 OnAudioFilterRead / 无重采样架构）。
    ///
    /// 参考原工程 DPDFNet/real_time_demo.py 的本质：采集率 == 模型率 == 播放率（均为 16 kHz），
    /// 因此全程不需要重采样，也不存在此前 OnAudioFilterRead(data 处于输出率) 导致的采样率错位与队列漂移。
    ///
    /// 管线：
    ///   麦克风(16000) --Microphone.GetData 逐帧--> DPDFNetProcessor.ProcessFrame(hop=160)
    ///        --> 增强样本入播放环形队列 --> PCMReaderCallback 喂给 playClip --> AudioSource 播放。
    ///
    /// 说明：
    ///   - 用 Microphone.GetData 轮询读取（带环绕安全处理 + 跨帧 leftover 续传，不丢样本）。
    ///   - 播放通过 AudioClip.Create 的 PCMReaderCallback 提供数据，AudioSource 直接播放该 clip。
    ///     PCMReaderCallback 是 clip 的数据供给回调，并非 OnAudioFilterRead。
    ///   - 采集时钟与播放时钟均源自同一音频硬件、同为 16000，速率天然匹配，环形队列仅作平滑缓冲。
    /// 建议戴耳机避免啸叫。
    /// </summary>
    [RequireComponent(typeof(AudioSource))]
    public class DpdfNetMicrophoneDemo : MonoBehaviour
    {
        [Header("模型")]
        [Tooltip("StreamingAssets/dpdfnet 下的模型名（不含扩展名）")] public string modelName = "dpdfnet2";
        [Tooltip("请求麦克风采样率，必须与模型一致(16000)")] public int captureSampleRate = 16000;

        [Header("输出")]
        [Range(0f, 1f)] public float playbackMix = 0f;   // 0=纯增强, 1=纯原始
        public bool monitorBypass = false;                // true=直出原始麦克风(诊断用)
        public bool enableAgc = true;

        [Header("调试")]
        public float lastInferenceMs;
        [Tooltip("每 N 秒打印一次诊断(RMS/帧耗时)")] public float diagInterval = 0.5f;

        private DpdfNetProcessor processor;
        private AudioClip micClip;
        private AudioClip playClip;
        private string micDevice;
        private AudioSource src;
        private int micRate;

        // 播放环形队列（主线程写，音频线程读）。用 lock 保护。
        private readonly Queue<float> playbackQueue = new Queue<float>();
        private readonly object queueLock = new object();
        private const int MaxQueueSamples = 16000 * 4;   // 4s 上限，溢出丢弃最旧

        // 麦克风读取游标与跨帧续传
        private int micPrevPos;
        private readonly Queue<float> pending = new Queue<float>();
        private float[] mixBuf;
        private bool running;

        // 诊断累计
        private float diagTimer;
        private float inRmsSum, outRmsSum;
        private int rmsFrames;
        private bool loggedModelError;
        private float agcGain = 1f;

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

            if (cfg.sample_rate != captureSampleRate)
                Debug.LogWarning($"[DPDFNet] 模型率 {cfg.sample_rate} 与 captureSampleRate {captureSampleRate} 不一致，建议统一为 16000。");

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
            mixBuf = new float[processor.HopLength];

            if (Microphone.devices.Length == 0)
            {
                Debug.LogError("[DPDFNet] 未检测到麦克风设备。");
                return;
            }
            micDevice = Microphone.devices[0];
            // 直接以模型率(16000)录制，原生即 16k，无需重采样。
            micClip = Microphone.Start(micDevice, true, 1, captureSampleRate);
            micRate = micClip.frequency;

            // 播放 clip：16k、单声道、循环、由 PCMReaderCallback 持续供给增强后的样本。
            playClip = AudioClip.Create("DPDFNetOut", captureSampleRate * 2, 1, captureSampleRate, true, OnAudioRead, null);
            src = GetComponent<AudioSource>();
            src.clip = playClip;
            src.loop = true;
            src.playOnAwake = false;
            src.Play();

            micPrevPos = 0;
            pending.Clear();
            lock (queueLock) playbackQueue.Clear();
            running = true;
            StartCoroutine(CaptureLoop());
            Debug.Log($"[DPDFNet] 采集率={micRate}, 模型率={cfg.sample_rate}, 播放率={captureSampleRate}（全程 16k，无重采样）");
        }

        private IEnumerator CaptureLoop()
        {
            int total = micClip.samples;
            int hop = processor.HopLength;
            float[] hopBuf = new float[hop];
            while (running && micClip != null)
            {
                int pos = Microphone.GetPosition(micDevice);
                int avail = (pos - micPrevPos + total) % total;
                if (avail > 0)
                {
                    // 环绕安全读取 avail 个样本到 region（最多 2 次 GetData）。
                    float[] region = new float[avail];
                    int first = Math.Min(avail, total - micPrevPos);
                    if (first == avail)
                    {
                        micClip.GetData(region, micPrevPos);
                    }
                    else
                    {
                        float[] head = new float[first];
                        micClip.GetData(head, micPrevPos);
                        Array.Copy(head, 0, region, 0, first);
                        float[] tail = new float[avail - first];
                        micClip.GetData(tail, 0);
                        Array.Copy(tail, 0, region, first, avail - first);
                    }
                    micPrevPos = (micPrevPos + avail) % total;

                    // 入 pending，按 hop 精确切帧（跨帧 leftover 续传，不丢样本）。
                    for (int i = 0; i < avail; i++) pending.Enqueue(region[i]);
                    while (pending.Count >= hop)
                    {
                        for (int i = 0; i < hop; i++) hopBuf[i] = pending.Dequeue();
                        ProcessHop(hopBuf);
                    }
                }
                yield return null;
            }
        }

        private void ProcessHop(float[] hop)
        {
            // 诊断：输入 RMS
            double inSq = 0;
            for (int i = 0; i < hop.Length; i++) inSq += hop[i] * hop[i];
            float inRms = (float)Math.Sqrt(inSq / hop.Length + 1e-12f);

            float[] enh;
            if (monitorBypass)
            {
                enh = hop;             // 跳过模型，原样透传
                lastInferenceMs = 0f;
            }
            else
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                try
                {
                    enh = processor.ProcessFrame(hop);
                }
                catch (Exception e)
                {
                    if (!loggedModelError)
                    {
                        Debug.LogError($"[DPDFNet] 推理异常：{e.Message}");
                        loggedModelError = true;
                    }
                    enh = hop;         // 异常时原样透传，避免静音
                }
                lastInferenceMs = (float)(sw.ElapsedTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency);
            }

            // 混合 + 块级 AGC（对应 Python apply_output_agc），结果写入 mixBuf。
            float outRms;
            lock (queueLock)
            {
                double mSq = 0;
                for (int i = 0; i < hop.Length; i++)
                {
                    float v = (1f - playbackMix) * enh[i] + playbackMix * hop[i];
                    mixBuf[i] = v;
                    mSq += v * v;
                }
                float rms = (float)Math.Sqrt(mSq / hop.Length + 1e-12f);
                if (enableAgc)
                {
                    const float targetRms = 0.12f, floor = 1e-3f, minGain = 0.25f, maxGain = 8f;
                    float desired = Mathf.Clamp(targetRms / Mathf.Max(rms, floor), minGain, maxGain);
                    agcGain = Mathf.Lerp(agcGain, desired, 0.05f);
                    for (int i = 0; i < hop.Length; i++)
                        mixBuf[i] = Mathf.Clamp(mixBuf[i] * agcGain, -1f, 1f);
                    outRms = (float)Math.Sqrt((mSq * agcGain * agcGain) / hop.Length + 1e-12f);
                }
                else
                {
                    for (int i = 0; i < hop.Length; i++)
                        mixBuf[i] = Mathf.Clamp(mixBuf[i], -1f, 1f);
                    outRms = rms;
                }

                for (int i = 0; i < hop.Length; i++) playbackQueue.Enqueue(mixBuf[i]);
                while (playbackQueue.Count > MaxQueueSamples) playbackQueue.Dequeue(); // 溢出丢弃最旧
            }

            AccumDiag(inRms, outRms);
        }

        // 由 Unity 音频线程调用，把增强样本供给 playClip。
        private void OnAudioRead(float[] data)
        {
            lock (queueLock)
            {
                for (int i = 0; i < data.Length; i++)
                    data[i] = playbackQueue.Count > 0 ? playbackQueue.Dequeue() : 0f;
            }
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
                int q;
                lock (queueLock) q = playbackQueue.Count;
                Debug.Log($"[DPDFNet] inRMS={avgIn:F4} outRMS={avgOut:F4} infer={lastInferenceMs:F2}ms bypass={monitorBypass} qPlay={q}");
                inRmsSum = outRmsSum = 0f;
                rmsFrames = 0;
                diagTimer = 0f;
            }
        }

        private void Cleanup()
        {
            running = false;
            StopAllCoroutines();
            if (micDevice != null && micClip != null && Microphone.IsRecording(micDevice))
                Microphone.End(micDevice);
            if (src != null && src.isPlaying) src.Stop();
            processor?.Dispose();
            processor = null;
            micDevice = null;
            micClip = null;
            playClip = null;
            pending.Clear();
            lock (queueLock) playbackQueue.Clear();
        }
    }
}
