# DPDFNet 在 Unity 中的 16 kHz 实时音频处理

参考原始工程 `DPDFNet/`（Python / onnxruntime）移植的流式语音降噪管线。
ONNX 模型位于 `Assets/StreamingAssets/dpdfnet/`，Unity 端逐帧执行：

```
麦克风(PCM) -> 流式 STFT(Vorbis窗) -> ONNX(state 进/出) -> 流式 ISTFT -> 输出
```

## 已就绪的资源

- `Assets/StreamingAssets/dpdfnet/*.onnx`：baseline / dpdfnet2 / dpdfnet4 / dpdfnet8（均为 **16 kHz** 模型，参数 `sr=16000, n_fft=320, hop=160, freq_bins=161`）。
- `Assets/StreamingAssets/dpdfnet/*.json`：旁路元数据（模型自定义 metadata 被 onnxsim 剥离，故用 `DPDFNet/generate_unity_sidecar_meta.py` 基于参考工程**确定性**重建初始 state；仅 `ErbNorm`/`SpecNorm` 两切片非零，其余为 0）。
- `Assets/Scripts/DPDFNet/`：C# 实现。

> 若日后更换/重新导出 ONNX，请重新运行 `DPDFNet/generate_unity_sidecar_meta.py` 以更新对应 JSON。

## 依赖（需要手动加入 Unity 工程）

### 1. Microsoft.ML.OnnxRuntime（CPU）
提供 ONNX 推理能力，API 与 Python `onnxruntime` 一一对应。
- 取得 NuGet 包 `Microsoft.ML.OnnxRuntime`（CPU 版）。
- 将托管程序集 `Microsoft.ML.OnnxRuntime.dll` 放入 `Assets/Plugins/OnnxRuntime/`。
- 将**原生库**放入对应平台子目录并设置 Import Settings（Plugin 类型：
  - Windows：`onnxruntime.dll`（x64/x86）→ `Assets/Plugins/OnnxRuntime/x86_64/`
  - Linux：`libonnxruntime.so`；Android：`libonnxruntime.so`（需 ndk）；macOS：`libonnxruntime.dylib`；iOS：静态库 + bitcode。
- 也可通过 Unity 的 NuGet 方案（如 `NuGetForUnity`）安装，本质相同。

### 2. MathNet.Numerics
提供任意长度（320 点）FFT，与 `numpy.fft` 的实/虚 FFT 约定一致。
- 取得 `MathNet.Numerics`（netstandard2.0 构建）。
- 将 `MathNet.Numerics.dll` 放入 `Assets/Plugins/`。
- 代码已使用 `System.Numerics.Complex`，Unity 内置支持。

## 使用

1. 在带 `AudioSource` 的 GameObject 上挂 `DpdfNetMicrophoneDemo`。
2. Inspector 设置：
   - `Model Name`：`dpdfnet2`（或 baseline/dpdfnet4/dpdfnet8）。
   - `Capture Sample Rate`：默认 `16000`（与模型一致；设备不支持时自动重采样）。
   - `Playback Mix`：`0`=纯增强，`1`=纯原始（用于对比）。
   - `Enable Agc`：输出自动增益。
3. **戴耳机**运行以避免啸叫。
4. 运行日志会输出重采样信息、`lastInferenceMs`（单帧推理耗时，应远小于 10 ms @16k）。

## 关键移植点（与 Python 对齐）

| 项 | 值 / 说明 |
|---|---|
| 采样率 / 帧长 / 跳数 | 16000 / 320 / 160（16k 模型族） |
| 窗 | Vorbis，分析=合成（满足 Princen-Bradley） |
| 频谱布局 | `[1,1,161,2]`（batch, time, freq, real/imag） |
| 状态向量 | 单维 `[state_size]`，首帧由 JSON 初值重建，逐帧 `state_out→state_in` 回传 |
| 归一化 `wnorm` | 已烘焙进 ONNX 图，Unity 直接喂原始 STFT |
| FFT | Math.NET `Forward`(不缩放) / `Inverse`(含 1/N)，等价 `np.fft.rfft/irfft` |
| 流式 ISTFT | OLA，与 `real_time_demo.py` 完全一致 |

## 性能与扩展

- `OnnxRuntimeSession` 已设单线程 + `ORT_ENABLE_ALL` 图优化，与 Python 一致。
- `IOnnxSession` 抽象便于替换为 **Unity Sentis**（`com.unity.sentis` 包，导入 ONNX 后在 GPU/CPU 上推理，免去原生 DLL），适合 iOS/Android 发布。
- 实时路径在 `OnAudioFilterRead` 内逐 hop 推理；若单帧耗时偏高，可改为双缓冲工作线程（生产/消费环形队列）。
- 离线处理文件：直接循环调用 `DpdfNetProcessor.ProcessFrame`，最后裁剪前 `n_fft*2` 个样本（对应 Python `postprocess_spec` 的尾零对齐）。
