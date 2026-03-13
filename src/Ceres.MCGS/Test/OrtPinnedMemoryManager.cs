#region Using directives

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Ceres;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.Positions;
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

sealed unsafe class OrtPinnedMemoryManager<T> : MemoryManager<T> where T : unmanaged
{
  private readonly OrtMemoryAllocation _alloc;
  private readonly byte* _ptr;
  private readonly int _length;
  private bool _addedRef;

  public OrtPinnedMemoryManager(OrtMemoryAllocation alloc)
  {
    _alloc = alloc ?? throw new ArgumentNullException(nameof(alloc));
    _alloc.DangerousAddRef(ref _addedRef); // keep handle alive while this Memory exists
    _ptr = (byte*)_alloc.DangerousGetHandle(); // address of the allocation
    _length = checked((int)alloc.Size / Marshal.SizeOf<T>());
  }

  public Span<T> Span => GetSpan();
  public override Span<T> GetSpan() => new Span<T>(_ptr, _length);

  public override MemoryHandle Pin(int elementIndex = 0) { throw new NotImplementedException(); }// => new MemoryHandle(_ptr + elementIndex, pinnable: null); }

  public override void Unpin() { }

  protected override void Dispose(bool disposing)
  {
    if (_addedRef)
    {
      _alloc.DangerousRelease();
      _addedRef = false;
    }
  }


  public static void ORTTestGraphSimpleBroken()
  {

    //    EncodedPositionWithHistory eph = default;
    //    eph.SetFromPosition(Position.StartPosition);
    //    TPGSquareRecord[] tsq = new TPGSquareRecord[64];
    //    TPGRecordConverter.ConvertToTPGRecordSquares(in eph, true, tsq, default, false, 0.3f, 0.3f);

    const int NUM_POS = 2;

    NNEvaluatorOptionsCeres optionsCeres = new();
    EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true;

    byte[] bytesAll1 = new byte[NUM_POS * 64 * 137];
    byte[] bytesAll2 = new byte[NUM_POS * 64 * 137];

    EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(NUM_POS, NNEvaluator.InputTypes.All);
    builder.Add(PositionWithHistory.StartPosition);
    builder.Add(PositionWithHistory.FromPosition(Position.FromFEN("8/8/5k2/1R3pp1/1p5p/2b2P1P/2P1K3/8 b - - 2 43")));
    for (int i = 0; i < NUM_POS - 2; i++)
    {
      builder.Add(PositionWithHistory.StartPosition);
    }

    EncodedPositionBatchFlat batch = builder.GetBatch();
    TPGRecordConverter.ConvertPositionsToRawSquareBytes(builder.GetBatch(), true, batch.Moves, false,
                                                        optionsCeres.QNegativeBlunders, optionsCeres.QPositiveBlunders,
                                                        out _, bytesAll1, null);
    builder.ResetBatch();
    builder.Add(PositionWithHistory.FromPosition(Position.FromFEN("8/8/5k2/1R3pp1/1p5p/2b2P1P/2P1K3/8 b - - 2 43")));
    builder.Add(PositionWithHistory.StartPosition);
    for (int i = 0; i < NUM_POS - 2; i++)
    {
      builder.Add(PositionWithHistory.StartPosition);
    }
    TPGRecordConverter.ConvertPositionsToRawSquareBytes(builder.GetBatch(), true, batch.Moves, false,
                                                        optionsCeres.QNegativeBlunders, optionsCeres.QPositiveBlunders,
                                                        out _, bytesAll2, null);



    ONNXTestEngine engine1 = new();
    engine1.MakeEngine(NUM_POS); // zzz

    RunOptions roGraph1, roGraph2;
    roGraph1 = new();
    roGraph1.AddRunConfigEntry("gpu_graph_id", "0");
    roGraph2 = new();
    roGraph2.AddRunConfigEntry("gpu_graph_id", "1");

    IDisposableReadOnlyCollection<OrtValue> resultsONN;
#if NOT
    ONNXExecutor.EXTERNAL_RUNNER = (int numPos, Array arr) =>
    {
      Span<byte> inSpan = engine1.inputOrt.GetTensorMutableDataAsSpan<byte>();
      ((byte[])arr).AsSpan<byte>().Slice(0, numPos * 64 * 137).CopyTo(inSpan);
      engine1.ioBinding.BindInput("squares_byte", engine1.inputOrt);
       engine1.session.RunWithBinding(roGraph1, engine1.ioBinding);
      return engine1.ioBinding.GetOutputValues();
    };
#endif
    for (int ix = 0; ix < 30; ix++)
    {
      Span<byte> inSpan = engine1.inputOrt.GetTensorMutableDataAsSpan<byte>();
      // The key to make this work is to copy data first and then rebind the input
      (ix % 2 == 0 ? bytesAll1 : bytesAll2).AsSpan<byte>().Slice(0, batch.NumPos * 64 * 137).CopyTo(inSpan);
      engine1.ioBinding.BindInput("squares_byte", engine1.inputOrt);

      TimingStats ts = new();
      using (new TimingBlock(ts, TimingBlock.LoggingType.None))
      {
        //          resultsONN = session.RunWithBoundResults(ro, ioBinding);
        engine1.session.RunWithBinding(ix < 15 ? roGraph1 : roGraph2, engine1.ioBinding);
      }
      resultsONN = engine1.ioBinding.GetOutputValues();
      Span<Float16> outSpan = resultsONN[0].GetTensorMutableDataAsSpan<Float16>();
      Console.WriteLine(ix + " " + (float)outSpan[0] + " " + (float)outSpan[3] + "   "
                        + (1000 * ts.ElapsedTimeSecs) + " " + (1000 * ts.CPUTimeSecs));
    }
  }


  public class ONNXTestEngine
  {
    public OrtTensorRTProviderOptions opt;
    public SessionOptions sessionOptions;
    public InferenceSession session;
    public OrtMemoryInfo miPinned;
    public OrtAllocator pinnedAllocator;
    public OrtValue inputOrt, outputOrt;
    public OrtIoBinding ioBinding;

    public void MakeEngine(int NUM_POS)
    {
      opt = new();
      opt.UpdateOptions(new Dictionary<string, string>()
    {
        { "device_id", "0" },
        { "trt_cuda_graph_enable", "1" },

        { "trt_profile_min_shapes", $"squares_byte:{NUM_POS}x64x137" },
        { "trt_profile_opt_shapes", $"squares_byte:{NUM_POS}x64x137" },
        { "trt_profile_max_shapes", $"squares_byte:{NUM_POS}x64x137" },

//       { "trt_builder_optimization_level", "3" },

//      { "trt_engine_cache_enable" , "1" },
//      { "trt_engine_cache_path", @"d:\temp\trtcache" },
//      { "trt_timing_cache_enable", "1"},
//      { "trt_timing_cache_path", @"d:\temp\trtcache" },

        {  "trt_fp16_enable", "1" }
    });


      //      string cachePrefix = FileUtils.FileNameSanitized(shortID);
      //      cachePrefix += $"_bs{minBatch}-{maxBatch}" + (useCudaGraphs ? "-graph" : "");
      //        ["trt_engine_cache_prefix"] = cachePrefix;
      //        ["trt_force_timing_cache"] = "1";

      long[] SHAPE_IN = [NUM_POS, 64, 137];
      long[] SHAPE_OUT = [NUM_POS, 1858];
      const string NET = "C1-256-10-i8.onnx";
      sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider(opt);
      using (new TimingBlock("load model"))
      {
        session = new InferenceSession($@"e:\cout\nets\{NET}", sessionOptions);
        //using InferenceSession session = new InferenceSession(@"e:\cout\nets\mix384_80_6late.onnx", sessionOptions);
      }
      miPinned = new(OrtMemoryInfo.allocatorCUDA_PINNED, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.CpuOutput);
      pinnedAllocator = new OrtAllocator(session, miPinned);
      inputOrt = OrtValue.CreateAllocatedTensorValue(pinnedAllocator, TensorElementType.UInt8, SHAPE_IN);
      outputOrt = OrtValue.CreateAllocatedTensorValue(pinnedAllocator, TensorElementType.Float16, SHAPE_OUT);
      ioBinding = session.CreateIoBinding();
      ioBinding.BindOutput("policy", outputOrt);
    }
  }

  public static void ORTTestGraphSimpleWorksNonCERES()
  {
    const bool CERES = false;
    var TYPE = CERES ? TensorElementType.UInt8 : TensorElementType.Float;
    using OrtTensorRTProviderOptions opt = new();
    opt.UpdateOptions(new Dictionary<string, string>()
    {
        { "device_id", "0" },
        { "trt_cuda_graph_enable", "1" }
      });

    long[] SHAPE = CERES ? [2, 64, 137] : [2, 3];

    using SessionOptions sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider(opt);
    using InferenceSession session = new InferenceSession(CERES ? @"e:\cout\nets\c1-640-34-i8.onnx"
                                                               : @"c:\temp\dummy_model.onnx", sessionOptions);
    using RunOptions runOptions = new RunOptions();

    // 2) Create CUDA-PINNED memory info and allocator
    //    Host-pinned (page-locked) memory: name="CudaPinned", type=DeviceAllocator, memType=CpuOutput
    using OrtMemoryInfo miPinned = new(OrtMemoryInfo.allocatorCUDA_PINNED,
                                        OrtAllocatorType.DeviceAllocator,
                                        deviceId: 0, OrtMemType.CpuOutput);

    using OrtAllocator pinnedAllocator = new(session, miPinned);
    using OrtValue inputOrt = OrtValue.CreateAllocatedTensorValue(pinnedAllocator, TensorElementType.Float, SHAPE);
    using OrtValue outputOrt = OrtValue.CreateAllocatedTensorValue(pinnedAllocator, TensorElementType.Float, SHAPE);

    using OrtIoBinding ioBinding = session.CreateIoBinding();
    ioBinding.BindOutput("output_image", outputOrt);

    // Fill input with random floats (directly into pinned memory)
    var rand = new Random(123);
    Span<float> inSpan = inputOrt.GetTensorMutableDataAsSpan<float>();

    for (int ix = 0; ix < 5; ix++)
    {
      for (int i = 0; i < inSpan.Length; ++i) { inSpan[i] = (float)rand.NextDouble(); }

      // 6) Run with binding
      ioBinding.BindInput(CERES ? "squares_byte" : "x", inputOrt);
      RunOptions ro = new();
      //ro.AddRunConfigEntry("gpu_graph_id", "0");
      session.RunWithBinding(new RunOptions(), ioBinding);

      // 7) Retrieve output from the pinned buffer (CPU-accessible)
      var outSpan = outputOrt.GetTensorMutableDataAsSpan<float>();
      Console.WriteLine("Output (first 10 floats):");
      int limit = Math.Min(10, outSpan.Length);
      for (int i = 0; i < limit; ++i)
      {
        Console.WriteLine($"  y[{i}] = {outSpan[i]} was: {inSpan[i]}");
      }
    }


    System.Environment.Exit(3);
  }

}
