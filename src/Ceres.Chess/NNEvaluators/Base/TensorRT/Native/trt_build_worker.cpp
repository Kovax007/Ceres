/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

/*
  Standalone TensorRT engine builder worker.
  Spawned as a separate OS process per GPU for truly parallel engine building.
  Each process gets its own TensorRT runtime, avoiding internal serialization
  that occurs when multiple threads share a single TRT builder in one process.

  Builds engine(s) for specified batch sizes, saves to cache, then exits.
  The parent process loads all cached engines afterwards.

  Usage: trt_build_worker <onnxPath> <deviceId> <cacheDir> <batch1,batch2,...>
                          <builderOptLevel> <tilingOptLevel> <spinWait> <cudaGraphs>
                          <fp16> <bf16> <fp8> <best>
                          <fp32PostAttNorm> <fp32PostAttNormStrict> <fp32SmolgenNorm>
                          <refittable>
                          [<timingCacheOutputPath>]
*/

#include "TensorRTWrapper.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

static std::vector<int> ParseBatchSizes(const char* str)
{
  std::vector<int> sizes;
  std::string s(str);
  size_t pos = 0;
  while (pos < s.size())
  {
    size_t comma = s.find(',', pos);
    if (comma == std::string::npos) comma = s.size();
    sizes.push_back(std::stoi(s.substr(pos, comma - pos)));
    pos = comma + 1;
  }
  return sizes;
}

int main(int argc, char* argv[])
{
  if (argc < 17)
  {
    fprintf(stderr, "Usage: trt_build_worker <onnxPath> <deviceId> <cacheDir> <batch1,batch2,...>\n"
                    "       <builderOptLevel> <tilingOptLevel> <spinWait> <cudaGraphs>\n"
                    "       <fp16> <bf16> <fp8> <best>\n"
                    "       <fp32PostAttNorm> <fp32PostAttNormStrict> <fp32SmolgenNorm>\n"
                    "       <refittable>\n"
                    "       [<timingCacheOutputPath>]\n");
    return 1;
  }

  const char* onnxPath = argv[1];
  int deviceId = atoi(argv[2]);
  const char* cacheDir = argv[3];

  std::vector<int> batchSizes = ParseBatchSizes(argv[4]);
  int numProfiles = (int)batchSizes.size();

  TRT_BuildOptions options;
  TRT_InitBuildOptions(&options);
  options.builderOptimizationLevel    = atoi(argv[5]);
  options.tilingOptimizationLevel     = atoi(argv[6]);
  options.useSpinWait                 = atoi(argv[7]);
  options.useCudaGraphs               = atoi(argv[8]);
  options.useFP16                     = atoi(argv[9]);
  options.useBF16                     = atoi(argv[10]);
  options.useFP8                      = atoi(argv[11]);
  options.useBest                     = atoi(argv[12]);
  options.fp32PostAttentionNorm       = atoi(argv[13]);
  options.fp32PostAttentionNormStrict = atoi(argv[14]);
  options.fp32SmolgenNorm             = atoi(argv[15]);
  options.refittable                  = atoi(argv[16]);

  // Optional timing cache output path (arg 17)
  const char* timingCacheOutputPath = (argc >= 18) ? argv[17] : nullptr;

  fprintf(stderr, "[Worker GPU %d] Building profiles [%s]\n", deviceId, argv[4]);
  if (timingCacheOutputPath)
  {
    fprintf(stderr, "[Worker GPU %d] Timing cache output: %s\n", deviceId, timingCacheOutputPath);
  }

  int32_t initResult = TRT_Init();
  if (initResult != 0)
  {
    fprintf(stderr, "[Worker GPU %d] TRT_Init failed: %s\n", deviceId, TRT_GetLastError());
    return 2;
  }

  TRT_EngineHandle handles[64];
  int32_t wasCached = 0;
  int32_t result = TRT_LoadONNXMultiProfileCachedWithTimingCache(
    onnxPath, batchSizes.data(), numProfiles, &options, deviceId,
    cacheDir, 0,
    nullptr, timingCacheOutputPath,
    &wasCached, handles);

  if (result != 0)
  {
    fprintf(stderr, "[Worker GPU %d] Build FAILED (code %d): %s\n",
            deviceId, result, TRT_GetLastError());
    TRT_Shutdown();
    return 3;
  }

  // Dispose handles — we only need the cache file on disk
  for (int i = 0; i < numProfiles; i++)
  {
    TRT_FreeEngine(handles[i]);
  }

  if (wasCached)
  {
    fprintf(stderr, "[Worker GPU %d] Loaded from cache (already built)\n", deviceId);
  }
  else
  {
    fprintf(stderr, "[Worker GPU %d] Build complete, engine cached.\n", deviceId);
  }

  TRT_Shutdown();
  return 0;
}
