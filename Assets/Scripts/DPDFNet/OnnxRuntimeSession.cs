using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DPDFNetUnity
{
    /// <summary>
    /// 基于 Microsoft.ML.OnnxRuntime 的 ONNX 流式推理后端。
    /// API 与 Python onnxruntime 一一对应（单线程、CPU、全图优化）。
    /// </summary>
    public class OnnxRuntimeSession : IOnnxSession, IDisposable
    {
        private readonly InferenceSession session;
        private readonly string specName;
        private readonly string stateInName;
        public int FreqBins { get; }
        public int StateSize { get; }

        public OnnxRuntimeSession(string modelPath)
        {
            var options = new SessionOptions();
            options.IntraOpNumThreads = 1;
            options.InterOpNumThreads = 1;
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            session = new InferenceSession(modelPath, options);

            var inputs = session.InputMetadata;
            var keys = new List<string>(inputs.Keys);
            specName = inputs.ContainsKey("spec") ? "spec" : keys[0];
            stateInName = inputs.ContainsKey("state_in") ? "state_in" : keys[1];

            var specDims = inputs[specName].Dimensions;
            FreqBins = specDims.Length >= 3 ? specDims[2] : 161;
            var stateDims = inputs[stateInName].Dimensions;
            StateSize = stateDims.Length >= 1 ? stateDims[0] : 0;

            if (FreqBins <= 0) FreqBins = 161;
            if (StateSize <= 0)
                throw new InvalidOperationException("[DPDFNet] 无法从 ONNX 输入形状推断 state_size。");
        }

        public void Run(float[] specRI, float[] stateIn, out float[] specRIOut, out float[] stateOut)
        {
            var specTensor = new DenseTensor<float>(specRI, new[] { 1, 1, FreqBins, 2 });
            var stateTensor = new DenseTensor<float>(stateIn, new[] { StateSize });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(specName, specTensor),
                NamedOnnxValue.CreateFromTensor(stateInName, stateTensor),
            };
            using var results = session.Run(inputs);
            specRIOut = ToArray(results[0].AsTensor<float>());
            stateOut = ToArray(results[1].AsTensor<float>());
        }

        public void Dispose()
        {
            session?.Dispose();
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// 把 ONNX Runtime 的 Tensor&lt;T&gt; 拷贝为扁平数组。
        /// 该定制版（onnxruntime@f8cfbf149d）的 Tensor/ DenseTensor 均无 ToArray()，
        /// 且 params int[] 索引器（this[ReadOnlySpan&lt;int&gt;]）对多维张量（如 spec [1,1,161,2]）
        /// 单下标访问会越界。正确做法：运行时实例是 DenseTensor&lt;T&gt;，其公开 Buffer(Memory&lt;T&gt;)
        /// 暴露底层扁平缓冲，整块 Span 拷贝即可（row-major，与构造张量布局一致）。
        /// 对非 DenseTensor 兜底用基类公开的 GetValue(int) 扁平访问器遍历。
        /// </summary>
        private static float[] ToArray(Tensor<float> tensor)
        {
            if (tensor is DenseTensor<float> dt)
            {
                var span = dt.Buffer.Span;
                var arr = new float[span.Length];
                span.CopyTo(arr.AsSpan());
                return arr;
            }
            var arr2 = new float[tensor.Length];
            for (int i = 0; i < arr2.Length; i++)
                arr2[i] = tensor.GetValue(i);
            return arr2;
        }
    }
}
