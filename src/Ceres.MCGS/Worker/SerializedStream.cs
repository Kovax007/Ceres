// SerializedStream — thread-safe write wrapper around a NetworkStream.
//
// NetworkStream is documented as not safe for concurrent writes; this class
// serializes all SendResponseAsync calls through a single SemaphoreSlim so
// that game-thread progress writes don't interleave with the final result
// send. Without this, the per-message length-prefix and body bytes from
// concurrent writers shred each other on the wire — the reader sees a
// corrupt length, blocks forever on read, and the chunk silently never
// completes. Caught by ultrareview 2026-05-21.

using System;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace Ceres.MCGS.Worker
{
  /// <summary>
  /// Write-side-serializing wrapper around a NetworkStream. Read-side access
  /// is left direct via <see cref="UnderlyingStream"/> because header reads
  /// in HandleClientAsync are single-threaded by construction.
  /// </summary>
  public sealed class SerializedStream : IDisposable
  {
    private readonly NetworkStream _stream;
    private readonly SemaphoreSlim _writeLock = new SemaphoreSlim(1, 1);

    public SerializedStream(NetworkStream stream)
    {
      _stream = stream;
    }

    /// <summary>
    /// Underlying NetworkStream — use for header reads only. Concurrent
    /// writes must go through SendResponseAsync to avoid interleaving.
    /// </summary>
    public NetworkStream UnderlyingStream => _stream;

    /// <summary>
    /// Serialized SendResponseAsync. Multiple game threads may call this
    /// concurrently; the semaphore guarantees one write at a time.
    /// </summary>
    public async Task SendResponseAsync<T>(T response, CancellationToken ct)
    {
      await _writeLock.WaitAsync(ct);
      try
      {
        await WorkerProtocol.SendResponseAsync(_stream, response, ct);
      }
      finally
      {
        _writeLock.Release();
      }
    }

    public void Dispose()
    {
      _writeLock.Dispose();
    }
  }
}
