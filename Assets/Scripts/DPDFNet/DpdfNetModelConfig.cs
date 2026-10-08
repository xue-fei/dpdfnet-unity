using System;
using UnityEngine;

namespace DPDFNetUnity
{
    /// <summary>
    /// 解析自 StreamingAssets/dpdfnet/&lt;model&gt;.json 的模型配置。
    /// 由于 onnxsim 剥离了 ONNX 自定义 metadata，初始 state 由该旁路 JSON 提供
    /// （仅 ErbNorm / SpecNorm 两个切片非零，其余全 0）。
    /// </summary>
    [Serializable]
    public class DpdfNetModelConfig
    {
        public string model_name;
        public int sample_rate = 16000;
        public int n_fft = 320;
        public int hop_length = 160;
        public int freq_bins = 161;
        public string window_type = "vorbis";
        public int window_length = 320;
        public int state_size;
        public int erb_norm_state_size;
        public int spec_norm_state_size;
        public float[] erb_norm_init;
        public float[] spec_norm_init;

        public static DpdfNetModelConfig Load(string json)
        {
            return JsonUtility.FromJson<DpdfNetModelConfig>(json);
        }

        /// <summary>
        /// 重建初始 state 向量：zeros(state_size)，再填入两个归一化初值切片。
        /// 与 Python real_time_demo.load_initial_state_from_metadata 完全对应。
        /// </summary>
        public float[] BuildInitialState()
        {
            float[] s = new float[state_size];
            if (erb_norm_init != null)
                Array.Copy(erb_norm_init, 0, s, 0, Math.Min(erb_norm_state_size, erb_norm_init.Length));
            if (spec_norm_init != null)
                Array.Copy(spec_norm_init, 0, s, erb_norm_state_size, Math.Min(spec_norm_state_size, spec_norm_init.Length));
            return s;
        }
    }
}
