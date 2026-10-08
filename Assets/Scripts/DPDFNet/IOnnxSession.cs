namespace DPDFNetUnity
{
    /// <summary>
    /// ONNX 推理抽象。当前由 OnnxRuntimeSession 实现（Microsoft.ML.OnnxRuntime）。
    /// 便于后续替换为 Unity Sentis 等后端。
    /// </summary>
    public interface IOnnxSession : System.IDisposable
    {
        int FreqBins { get; }
        int StateSize { get; }

        /// <summary>
        /// 单帧推理。specRI: 长度 FreqBins*2（交错 real/imag）；
        /// stateIn: 长度 StateSize；输出 specRIOut / stateOut 同样规格。
        /// </summary>
        void Run(float[] specRI, float[] stateIn, out float[] specRIOut, out float[] stateOut);
    }
}
