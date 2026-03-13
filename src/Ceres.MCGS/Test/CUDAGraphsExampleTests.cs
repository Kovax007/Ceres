#region Using directives

using System;
using System.Collections.Generic;
using Ceres.Base.Benchmarking;
using ManagedCuda;
using ManagedCuda.BasicTypes;
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

public static class CUDAGraphsExampleTests
{
  public static unsafe void BuildAndRunGraph()
  {
    // ---- Change these to match your model ----
    const string ModelPath = @"e:\cout\nets\c1-256-10-i8.onnx";
    const string INPUT_NAME = "squares_byte";
    const string OUTPUT_NAME = "value";
    long[] InputShape = { 2, 64, 137 }; // NCHW
    long[] OutputShape = { 2, 3 };
    // -----------------------------------------

    // 0) ManagedCUDA primary context (shares the CUDA primary context with ORT)
    using var ctx = new PrimaryContext(0); // device 0. Dispose at the very end. :contentReference[oaicite:0]{index=0}

    // 1) Build SessionOptions + enable CUDA Graphs
    using var cudaOpts = new OrtCUDAProviderOptions();
    cudaOpts.UpdateOptions(new Dictionary<string, string>
    {
      ["enable_cuda_graph"] = "1"  // capture on first Run; replay later
    });
    using var so = SessionOptions.MakeSessionOptionWithCudaProvider(cudaOpts); // device 0 by default. :contentReference[oaicite:1]{index=1}
    so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

    using var session = new InferenceSession(ModelPath, so);

    // 2) Make ONNX 'device' memory info ("Cuda", DeviceAllocator, Default)
    using var miCuda = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, deviceId: 0, OrtMemType.Default);                     // device/default mem type. :contentReference[oaicite:2]{index=2}

    // 3) Allocate DEVICE buffers with ManagedCUDA (stable addresses for CUDA Graphs)
    int inElems = checked((int)NumElements(InputShape));
    int outElems = checked((int)NumElements(OutputShape));

    using var dIn = new CudaDeviceVariable<float>(inElems);
    using var dOut = new CudaDeviceVariable<float>(outElems);

    // 4) Initialize host input and copy -> device (ManagedCUDA)
    byte[] hIn = new byte[inElems];
    var rng = new Random(123);
    for (int i = 0; i < hIn.Length; ++i) hIn[i] = (byte)(100 * rng.NextDouble());
    dIn.CopyToDevice(hIn);  // H->D

    // 5) Wrap existing DEVICE pointers as OrtValue tensors (no copies; ORT doesn't own memory)
    using var inOrt = OrtValue.CreateTensorValueWithData(miCuda, TensorElementType.UInt8, InputShape,
                                                         CUdeviceptrToIntPtr(dIn.DevicePointer), sizeof(byte) * (long)inElems);

    using var outOrt = OrtValue.CreateTensorValueWithData(miCuda, TensorElementType.Float16, OutputShape,
                                                          CUdeviceptrToIntPtr(dOut.DevicePointer), sizeof(Float16) * (long)outElems);

    // 6) IOBinding (device-to-device)
    using var io = session.CreateIoBinding();
    io.BindInput(INPUT_NAME, inOrt);
    io.BindOutput(OUTPUT_NAME, outOrt);

    // Optional: tag graph id (0 by default). First run captures; later runs replay. :contentReference[oaicite:3]{index=3}
    using var ro = new RunOptions();
    ro.AddRunConfigEntry("gpu_graph_id", "0");

    // 7) Run #1 = capture; subsequent runs = replay
    session.RunWithBinding(ro, io);
    session.RunWithBinding(ro, io);

    // 8) Copy device output back to host to inspect
    float[] hOut = new float[outElems];
    dOut.CopyToHost(hOut);  // D->H
    Console.WriteLine("Output (first 10 floats):");
    for (int i = 0; i < Math.Min(10, hOut.Length); ++i) Console.WriteLine($"  y[{i}] = {hOut[i]}");


    for (int i = 0; i < hIn.Length; ++i) hIn[i] = (byte)(100 * rng.NextDouble());
    dIn.CopyToDevice(hIn);  // H->D
    session.RunWithBinding(ro, io);
    dOut.CopyToHost(hOut);  // D->H
    Console.WriteLine("Output (first 10 floats):");
    for (int i = 0; i < Math.Min(10, hOut.Length); ++i) Console.WriteLine($"  y[{i}] = {hOut[i]}");
  }


  static long NumElements(long[] shape)
  {
    long n = 1;
    foreach (var d in shape) { if (d <= 0) throw new ArgumentException("No dynamic dims in this minimal sample."); n *= d; }
    return n;
  }

  // ---- Interop shim: convert ManagedCUDA CUdeviceptr -> IntPtr for ORT ----
  static IntPtr CUdeviceptrToIntPtr(CUdeviceptr dptr)
  {
    // CUdeviceptr has an implicit cast to ulong; convert to IntPtr for 64-bit process. :contentReference[oaicite:4]{index=4}
    ulong addr = dptr;                 // implicit operator ulong
    return (IntPtr)(long)addr;         // assume 64-bit .NET; adjust if targeting 32-bit
  }

  /// <summary>
  /// Bug: https://github.com/microsoft/onnxruntime/issues/22583
  /// Confirmed to fail, outputs are usually wrong (zero) but perhaps once worked on the first vector input
  /// </summary>
  public static void ORTTestGraph()
  {
    using OrtTensorRTProviderOptions opt = new();
    opt.UpdateOptions(new Dictionary<string, string>()
    {
        { "device_id", "0" },
        { "trt_cuda_graph_enable", "1" }
    });

    using SessionOptions sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider(opt);
    //using SessionOptions sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider();
    using InferenceSession inferenceSession = new InferenceSession(@"c:\temp\dummy_model.onnx", sessionOptions);
    using RunOptions runOptions = new RunOptions();

    using var inInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA_PINNED, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.CpuInput);
    using var outInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA_PINNED, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.CpuOutput);

    using OrtAllocator cudaAllocatorIn = new OrtAllocator(inferenceSession, inInfo);
    using OrtAllocator cudaAllocatorOut = new OrtAllocator(inferenceSession, outInfo);

    using OrtIoBinding ioBinding = inferenceSession.CreateIoBinding();
    using OrtValue inputTensor = OrtValue.CreateAllocatedTensorValue(cudaAllocatorIn, TensorElementType.Float, [2, 3]);
    using OrtValue outputTensor = OrtValue.CreateAllocatedTensorValue(cudaAllocatorOut, TensorElementType.Float, [2, 3]);

    ioBinding.BindOutput("output_image", outputTensor);

    //    var infoIn = inputTensor.GetTensorMemoryInfo();
    //    var infoOut = outputTensor.GetTensorMemoryInfo();

    while (true)
    {
      using (new TimingBlock("xx"))
      {
        for (int i = 0; i < 1000; i++)
        {
          Span<float> inputSpan0 = inputTensor.GetTensorMutableDataAsSpan<float>();
          float[] inputData0 = [i * 1, i * 2, i * 1, i * 2, i * 1, i * 2];
          inputData0.AsSpan().CopyTo(inputSpan0);
          ioBinding.BindInput("x", inputTensor);

          ioBinding.SynchronizeBoundInputs();
          inferenceSession.RunWithBinding(runOptions, ioBinding);
          ioBinding.SynchronizeBoundOutputs();

          PrintTensor<float>(outputTensor, "result");
        }
      }
    }

    static void PrintTensor<T>(OrtValue t, string header) where T : unmanaged
      => Console.WriteLine(header + ": " + string.Join(" ", t.GetTensorDataAsSpan<T>().ToArray()));
  }

}
