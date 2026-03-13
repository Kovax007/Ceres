#region Using directives

using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.MCGS.Test;

public static class ONNXTests
{
public static unsafe void RunONNXTest()
{
    string modelPath = @"d:\nets\744706_spsa032025.pb_fp16.onnx";// weights_run1_811971.pb.gz_fp16.onnx";
    string inputName = "/input/planes"; // Float16, [batch,112,8,8]

    // Three optimization profiles: 1..4, 5..11, 12..64
    string trtMin = $"{inputName}:1x112x8x8;{inputName}:5x112x8x8;{inputName}:12x112x8x8";
    string trtOpt = $"{inputName}:4x112x8x8;{inputName}:9x112x8x8;{inputName}:32x112x8x8";
    string trtMax = $"{inputName}:4x112x8x8;{inputName}:11x112x8x8;{inputName}:64x112x8x8";

    // Append TensorRT EP with FP16 + explicit multi-profile shapes.
    var trtOptions = new Dictionary<string, string>
    {
      ["trt_fp16_enable"] = "1",
      ["device_id"] = "0",

      ["trt_profile_min_shapes"] = trtMin,
      ["trt_profile_opt_shapes"] = trtOpt,
      ["trt_profile_max_shapes"] = trtMax,

      // Optional niceties:
      // ["trt_engine_cache_enable"] = "1",
      // ["trt_engine_cache_path"] = "./trt_cache",
      // ["trt_timing_cache_enable"] = "1",
    };

    //so.AppendExecutionProvider_Tensorrt(trtProviderOptions);
    OrtTensorRTProviderOptions trtProviderOptions = new OrtTensorRTProviderOptions();
    trtProviderOptions.UpdateOptions(trtOptions);

    SessionOptions so = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtProviderOptions);
    InferenceSession session = new InferenceSession(modelPath, so);

    OrtMemoryInfo memoryInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA_PINNED,
                                                 OrtAllocatorType.DeviceAllocator, 0, OrtMemType.CpuOutput);
    OrtAllocator ortAllocator = new OrtAllocator(session, memoryInfo);

    OrtValue ovv = OrtValue.CreateAllocatedTensorValue(ortAllocator, TensorElementType.UInt8, [64, 64, 137]);
    OrtValue ovv1 = OrtValue.CreateTensorValueFromMemory<byte>(memoryInfo, new byte[3], [3, 3, 3]);

    OrtMemoryAllocation allocated = ortAllocator.Allocate(10000);

    OrtPinnedMemoryManager<byte> omm = new (allocated);
    omm.Span[999] = 33;

    byte* ptr = (byte*)allocated.DangerousGetHandle();
    Span<byte> span = new Span<byte>(ptr,10000);
    span[9999] = 7;

    Memory<byte> memOverAllocation = default;

    void RunBatch(int batchSize)
    {
      Tensor<Float16> input = RandomHalfTensor(batchSize, 112, 8, 8);
      List<NamedOnnxValue> container = new List<NamedOnnxValue>([NamedOnnxValue.CreateFromTensor(inputName, input)]);


      IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(container);
      Console.WriteLine($"batch={batchSize}");
      foreach (var r in results)
      {
        // All outputs are Float16: /output/policy [batch,1858], /output/wdl [batch,3], /output/mlh [batch,1]
        var t = r.AsTensor<Float16>();
        Console.WriteLine(r.Name + " " + r + " ");
        //	string shape = "[" + string.Join(",", t.Dimensions.Select(d => d.ToString())) + "]";
        //	Console.WriteLine($"  {r.Name,-16} -> shape {shape}, elemType Float16");
      }
    }

    RunBatch(4);  // picks profile 1..4
    RunBatch(9);  // picks profile 5..11
  }

  static Tensor<Float16> RandomHalfTensor(int n, int c, int h, int w)
  {
    var t = new DenseTensor<Float16>(new[] { n, c, h, w });
    var rand = new Random();
    // Fill with [0,1) cast to Half
    var span = t.Buffer.Span;
    for (int i = 0; i < span.Length; i++)
    {
      float v = (float)rand.NextDouble();
      span[i] = (Float16)v;
    }
    return t;
  }
}
