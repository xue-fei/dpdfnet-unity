# DPDFNet 在 Unity 中的 16 kHz 实时音频处理

参考原始工程 `DPDFNet/`（Python / onnxruntime）移植的流式语音降噪管线。
ONNX 模型位于 `Assets/StreamingAssets/dpdfnet/`，Unity 端逐帧执行：

```
麦克风(PCM) -> 流式 STFT(Vorbis窗) -> ONNX(state 进/出) -> 流式 ISTFT -> 输出
```

## 已就绪的资源

- `Assets/StreamingAssets/dpdfnet/*.onnx`：baseline / dpdfnet2 / dpdfnet4 / dpdfnet8（均为 **16 kHz** 模型，参数 `sr=16000, n_fft=320, hop=160, freq_bins=161`）。
- `Assets/Scripts/DPDFNet/`：C# 实现（核心 4 个文件，按层分组，外加 1 个可选示例文件）：

  | 文件 | 职责 |
  |---|---|
  | `DpdfNetAudio.cs` | DSP 层：`VorbisWindow`（Vorbis 窗）、`StftMath`（rfft/irfft 数学核，等价 `np.fft.rfft/irfft`） |
  | `OnnxRuntimeSession.cs` | ONNX 层：`IOnnxSession` 接口 + `OnnxRuntimeSession`（Microsoft.ML.OnnxRuntime 后端，含定制版 Tensor 扁平拷贝兼容） |
  | `DpdfNetProcessor.cs` | 模型配置 `DpdfNetModelConfig`（零依赖推导 + 初始 state 重建）+ 核心编排 `DpdfNetProcessor`（对齐 Python `stream.py` 的 `StreamEnhancer`） |
  | `DpdfNetMicrophoneDemo.cs` | `MonoBehaviour` Demo：麦克风 16k 采集 → 增强 → `AudioSource` 播放（无 `OnAudioFilterRead`、无重采样）。ONNX 推理经 `Loom` 后台线程执行，主线程仅做采集与落地，不阻塞帧率 |
  | `DpdfNetFileExample.cs` | **可选示例**：对 16 kHz WAV 文件做离线降噪（`DpdfNetFileExample.ProcessWav` + 极简 `WavIo` 读写），见下方「16 kHz 音频文件处理示例」 |

> **零依赖**：初始 state 与 DSP 参数均不在任何外部文件中。运行时从 ONNX 输入形状（`FreqBins` / `StateSize`）推导 `n_fft` / `hop` / `state_size`，并用与 Python `onnx_model/layers.py` 一致的公式重建 `ErbNorm` / `SpecNorm` 初值（仅两切片非零，其余为 0）。因此更换/重新导出 ONNX（同 16k 模型族）无需任何额外步骤。

## 依赖（通过 Unity Package Manager 引入，仅需以下两个包）

在 `Packages/manifest.json` 的 `dependencies` 中加入两个 git 包（已打包 ONNX 运行时与 CPU 原生库，无需手动放置 DLL）：

```json
{
  "dependencies": {
    "onnxruntime": "https://github.com/xue-fei/onnxruntime-unity.git",
    "onnxruntime-cpu": "https://github.com/xue-fei/onnxruntime-unity-cpu.git"
  }
}
```

- `onnxruntime`：提供 ONNX 推理能力，API 与 Python `onnxruntime` 一一对应（含托管包装与平台原生库）。
- `onnxruntime-cpu`：CPU 后端原生库。

保存后 Unity 会自动从 git 拉取并编译，无需 NuGet 或手动拷贝 DLL。

## 使用

1. 在带 `AudioSource` 的 GameObject 上挂 `DpdfNetMicrophoneDemo`。
2. Inspector 设置：
   - `Model Name`：`dpdfnet2`（或 baseline/dpdfnet4/dpdfnet8）。
   - `Capture Sample Rate`：默认 `16000`（与模型一致；Unity 会把麦克风重采样到该率写入 micClip）。
   - `Playback Mix`：`0`=纯增强，`1`=纯原始（用于对比）。
   - `Monitor Bypass`：勾选则直出原始麦克风（跳过模型），用于隔离采集/播放问题。
   - `Enable Agc`：输出自动增益。
3. **戴耳机**运行以避免啸叫。
4. 运行日志会输出 `inRMS / outRMS / infer(ms) / qPlay`（播放队列长度）。单帧推理耗时 `infer` 应远小于 10 ms @16k。

## 关键移植点（与 Python 对齐）

| 项 | 值 / 说明 |
|---|---|
| 采样率 / 帧长 / 跳数 | 16000 / 320 / 160（16k 模型族） |
| 窗 | Vorbis，分析=合成（满足 Princen-Bradley） |
| 频谱布局 | `[1,1,161,2]`（batch, time, freq, real/imag） |
| 状态向量 | 单维 `[state_size]`，首帧由 C# 公式重建（与 Python `initial_state()` 一致），逐帧 `state_out→state_in` 回传 |
| 归一化 `wnorm` | 已烘焙进 ONNX 图，Unity 直接喂原始 STFT |
| FFT | Math.NET `Forward`(不缩放) / `Inverse`(含 1/N)，等价 `np.fft.rfft/irfft` |
| 流式编排 | 对齐 Python `stream.py` 的 `StreamEnhancer`：win_len 输入帧缓冲 + OLA 输出缓冲 + `Flush` 排空尾部 |

## 实时架构说明（无 OnAudioFilterRead / 无自实现重采样）

参考原工程 `real_time_demo.py` 的本质——**采集率 == 模型率 == 播放率（均为 16 kHz）**，因此全程不做重采样，从根上避免此前 `OnAudioFilterRead(data 处于 Unity 输出率)` 导致的采样率错位与队列漂移。

- **采集**：`Microphone.Start(device, true, 1, 16000)` 直接以模型率录制；用 `Microphone.GetPosition` + `GetData` 轮询读取，带环绕安全处理，跨帧 leftover 续传（不丢样本）。
- **增强**：每读到 `hop=160` 个 16k 样本即调用 `DpdfNetProcessor.Process`（StreamEnhancer 语义：内部 win_len=320 帧缓冲，首窗未满 ~20ms 无输出，之后每次返回 1 个 hop 的增强样本）。输出样本 m 与输入样本 m 一一对应（延迟镜像），Demo 中经 `rawDelay` 队列与原始样本配对供 `playbackMix` 混合。
- **播放**：增强样本写入一个 `Queue<float>` 环形队列；`AudioClip.Create(..., PCMReaderCallback)` 由 Unity 音频线程回调取数据，对应 `AudioSource` 播放该 clip。`PCMReaderCallback` 是 clip 的数据供给回调，并非 `OnAudioFilterRead`。
- 采集与播放时钟同源（同一音频硬件、同为 16k），速率天然匹配，队列仅作平滑缓冲（上限 4s，溢出丢最旧）。

## 性能与扩展

- `OnnxRuntimeSession` 已设单线程 + `ORT_ENABLE_ALL` 图优化，与 Python 一致。
- `IOnnxSession` 抽象便于替换为 **Unity Sentis**（`com.unity.sentis` 包，导入 ONNX 后在 GPU/CPU 上推理，免去原生 DLL），适合 iOS/Android 发布。
- 实时路径的 ONNX 推理经 `Loom.RunAsync` 在后台线程执行（`inferLock` 串行保证 state 帧序），主线程协程只做采集与提交；背压超 `MaxInflight` 丢帧降质而非无限延迟。
- **实时路径**（`DpdfNetProcessor` / `DpdfNetMicrophoneDemo`）：对齐 Python 官方流式 API `stream.py` 的 `StreamEnhancer`——`Process(chunk)` 接受任意长度音频块（内部 win_len 帧缓冲 + COLA OLA），`Flush()` 流结束时补零排空尾部，`Reset()` 重置 RNN state。因果 STFT（center=False），输出与输入逐样本对齐、整体延迟一个窗（~20ms@16k）；与 `real_time_demo.py` 的逐 hop 滑窗方案等价但提交时机不同（win_len 缓冲后按 hop 提交，边界伪影更少）。
- **离线文件路径**（`DpdfNetFileExample`）：与源码 `onnx_model/infer_dpdfnet_onnx.py` 的 `enhance_file_onnx` 逐行对齐——`center=True` 反射填充 STFT → 流式 ONNX(state) → `center=True` ISTFT（win² 归一化 OLA，两端裁 `n_fft/2`）→ `postprocess_spec`（裁 `win_len*2`/补 `win_len*2`）→ `fit_length`。**两条路径帧对齐与边界处理不同，结果不可互换**；文件处理以 `infer_dpdfnet_onnx` 为准。流式 API（StreamEnhancer）与离线 enhance（center=True）亦非逐位一致（见 `stream.py` docstring），属算法固有差异。

## 16 kHz 音频文件处理示例

`DpdfNetFileExample.cs` 提供一行式离线降噪：读取 16 kHz WAV → 按源码 `infer_dpdfnet_onnx.enhance_file_onnx`（`center=True` 反射填充 STFT + 流式 ONNX + `center=True` ISTFT + `postprocess_spec`）处理 → 写回增强后的 WAV。仅依赖 onnxruntime 包，自带极简 `WavIo`（读 PCM 16/24/32-bit、float；写 16-bit PCM），无需 NAudio / librosa 等额外依赖。

### 用法

```csharp
// 桌面 / Editor：直接用本地文件路径
string model = @"C:\project\Assets\StreamingAssets\dpdfnet\dpdfnet2.onnx";
string input = @"C:\audio\noisy_16k.wav";
string output = @"C:\audio\enhanced_16k.wav";
DPDFNetUnity.DpdfNetFileExample.ProcessWav(model, input, output);

// Unity 运行时（如按钮回调），用 StreamingAssets 路径（桌面端可直接读）：
string modelPath = System.IO.Path.Combine(Application.streamingAssetsPath, "dpdfnet", "dpdfnet2.onnx");
DPDFNetUnity.DpdfNetFileExample.ProcessWav(modelPath, inputWav, outputWav);
```

### 关键逻辑（逐行对齐 Python `infer_dpdfnet_onnx.enhance_file_onnx`）

1. **读取**：`WavIo.TryRead` 把任意声道/位深转成单声道 `float[]`（归一化到 [-1,1]）；非 16 kHz 文件会告警（模型是 16k 训练的，建议先重采样）。
2. **建 session**：`new OnnxRuntimeSession(modelPath)` → `DpdfNetModelConfig.FromSession`（`using` 释放 session）。
3. **末端补 `win_len` + `center=True` 反射填充 `n_fft/2`**：对应 `np.pad(waveform,(0,win_len))` 与 librosa `pad_mode='reflect'`。
4. **center STFT + 流式 ONNX + center ISTFT**：逐帧 `rfft(win*frame)`，state 逐帧回传（`BuildInitialState` 起始）；`irfft*win` 后 win² 归一化 OLA，两端裁 `n_fft/2`（librosa `istft(center=True)`）。
5. **`postprocess_spec` + `fit_length`**：裁掉前 `win_len*2`、末尾补 `win_len*2` 个 0，再裁/补到原始长度 L。`win_len*2` 的裁剪抵消模型内部 ~4 hop 的延迟（CyclicBuffer），故输出与输入对齐。

### 平台注意

- **Android / iOS**：`StreamingAssets` 在打包后位于 APK/IPA 内，无法以文件系统路径直接读取。需先把模型/音频拷贝到 `Application.persistentDataPath`（可读写），再传入 `ProcessWav`。
- 示例 `WavIo` 仅覆盖常见格式且多声道写入简化（仅填第 0 声道）。生产环境可替换为 NAudio 或 Unity `AudioClip` 导入，但核心处理流程（离线 STFT → ONNX → ISTFT）不变。
