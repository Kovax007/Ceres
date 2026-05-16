#region License notice
/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres.  If not, see <http://www.gnu.org/licenses/>.
*/
#endregion

using System;
using Ceres.MCGS.Environment;
using Ceres.MCGS.Worker;

namespace Ceres.MCGS;

/// <summary>
/// Entry point for the Ceres.MCGS executable.
///
/// Dispatches CLI invocations to one of:
///   --worker [--gpu N] [--port P] [--host H] [--worker-config PATH]
///        Run a persistent TCP worker for distributed SPSA/NES tuning.
///   anything else (or no args)
///        Fall through to the upstream MCGSLaunch.Launch() — regular UCI / dev mode.
///
/// SPSA tournament-runner mode (--config) is reached through the top-level
/// Ceres binary's Program.cs, not from here.
/// </summary>
public static class WorkerEntry
{
  public static void Main(string[] args)
  {
    // --worker mode: persistent TCP worker for distributed SPSA/NES tuning
    if (args != null && Array.IndexOf(args, "--worker") >= 0)
    {
      WorkerLocalConfig localConfig = null;
      int? gpuOverride = null;
      int? portOverride = null;
      string hostOverride = null;

      for (int i = 0; i < args.Length; i++)
      {
        if (args[i] == "--worker-config" && i + 1 < args.Length)
        {
          localConfig = WorkerLocalConfig.Load(args[i + 1]);
        }
        if (args[i] == "--gpu" && i + 1 < args.Length)
        {
          gpuOverride = int.Parse(args[i + 1]);
        }
        if (args[i] == "--port" && i + 1 < args.Length)
        {
          portOverride = int.Parse(args[i + 1]);
        }
        if (args[i] == "--host" && i + 1 < args.Length)
        {
          hostOverride = args[i + 1];
        }
      }

      int gpuId = gpuOverride ?? localConfig?.GpuId ?? 0;
      int port = portOverride ?? localConfig?.Port ?? 5100;
      string host = hostOverride ?? localConfig?.BindHost ?? "0.0.0.0";

      WorkerServer.LaunchWorkerAsync(gpuId, port, localConfig, host).GetAwaiter().GetResult();
      return;
    }

    // Default: regular UCI / dev mode via upstream's launcher
    MCGSLaunch.Launch(args);
  }
}
