# MPI Implementation Compatibility

This is ferrompi's reference for selecting an MPI implementation, building
against it, and running under it. Each fact below is stated once; what an
individual method does lives in that method's own rustdoc on docs.rs, not
here.

## Platforms and MPI libraries

Linux is the supported platform and the only one CI tests (x86_64,
`ubuntu-24.04` runners). macOS builds but is not CI-tested. `LongDoubleInt`
and `LongInt` are available only on Linux, and only on x86_64, aarch64 and
little-endian powerpc64. Windows is not supported.

ferrompi needs an MPI 3.1 library. It enables its MPI 4.0 operations —
persistent collectives, `Mpi::create_from_group`, and the `_c` large-count
calls — only when the library's `mpi.h` reports `MPI_VERSION >= 4`. Below
that threshold, persistent collectives and `create_from_group` return
`Error::NotSupported`, and a count above `i32::MAX` returns `Error::Mpi` with
class `Count`. The variable-count collectives (`gatherv`, `scatterv`,
`allgatherv`, `alltoallv` and their persistent forms) take their counts as
`i32` arrays; a count above `i32::MAX` returns that same `Count` error on
every MPI, including one with `MPI_VERSION >= 4`.

The table below is the outcome of ferrompi's three CI builds: MPICH 4.2.1
(`ubuntu-24.04`, the Noble package hotfixed to that version, np 2/3/4,
default and `rma`), Open MPI 4.1.6 (`ubuntu-24.04`, np 4, default and
`rma`), and Open MPI 5.0.7 (a pinned `debian:trixie` container, np 4,
`rma`). Every other MPI — MPICH 3.x, Intel MPI, Cray MPICH, or any other
Open MPI version — is untested; expect it to behave like the MPICH or Open
MPI row that reports the same `MPI_VERSION`.

| Feature family | MPICH 4.2.1 | Open MPI 4.1.6 | Open MPI 5.0.7 |
| --- | --- | --- | --- |
| Blocking and nonblocking collectives and point-to-point | tested | tested | tested |
| Persistent point-to-point | tested | tested | tested |
| Persistent collectives | tested | `NotSupported` | `NotSupported` |
| `create_from_group` | tested | `NotSupported` | `NotSupported` |
| Counts above `i32::MAX` | tested | `Err` (class `Count`) | `Err` (class `Count`) |
| RMA windows (`rma`) | tested | tested | tested |

MPICH 4.2.1 reports `MPI_VERSION` 4, which is why it takes the "tested" cell
throughout. Open MPI 4.1.6 and 5.0.7 both report MPI 3.1 (`MPI_VERSION` 3),
so ferrompi compiles its MPI 4.0 stubs against them regardless of what the
library itself implements: Open MPI 5.0.7's library exports
`MPI_Allreduce_init`, but ferrompi's gate reads the header's `MPI_VERSION`,
not the library's symbol table, so the call still returns `NotSupported` on
that build.

## Known implementation issues

One item each: the symptom, the affected library, and what to do.

- MPICH 4.2.x: `scatter_init_inplace` (persistent `scatter` with
  `MPI_IN_PLACE` at the root) deadlocks. Fixed in MPICH 4.3. Use
  `scatter_init` with a separate send buffer on 4.2.x instead;
  `examples/test_inplace.rs` detects the affected version string at
  runtime and skips only this one case, not the other in-place persistent
  collectives.
- Ubuntu 24.04's `mpich` 4.2.0 package starts every rank as a singleton —
  each sees a world of size 1 — once more than one process is launched. CI
  installs MPICH 4.2.1 from a pinned Ubuntu snapshot instead
  ([Launchpad bug 2072338](https://bugs.launchpad.net/ubuntu/+source/mpich/+bug/2072338)).
  To avoid it, use an MPICH 4.2.1 or later package;
  `.github/scripts/mpich-hotfix.sh` shows the exact packages CI installs.
- `Win::sync` outside a passive-target epoch is erroneous; MPICH returns an
  error for it. Call it between `Win::lock`/`Win::lock_all` and the guard's
  drop — see the `Win::sync` method's own documentation on docs.rs for the
  full rule. Other libraries may accept it silently, so a program tested
  only there can fail on MPICH.
- After a failed `wait_all`, which of the other requests are still pending
  depends on the library and on timing. On MPICH, the requests that come
  after the failed one in the slice stay pending. ferrompi's own rule — a
  request is marked completed only once MPI actually completes it,
  successfully or not — holds on every library; rely on that rather than an
  implementation's particular pending behaviour. To finish the rest, call
  `Request::wait_all` again on the same slice; completed entries are
  skipped.
- Open MPI 4.1.6 and 5.0.7: a nonblocking receive truncated by a send from a
  rank to itself is not reported as an error; the receive returns success.
  A blocking `recv` of the same message does report `Truncate`; use it for
  self-messages where truncation must be detected.
- Open MPI 4.1.6 and 5.0.7: a persistent request that fails inside `wait`,
  `test` or `wait_all` is freed by MPI. ferrompi marks the owning
  `PersistentRequest` inactive; a later `start` on it returns `Err` with
  class `Request`. Create a new request to run the operation again.
- Open MPI 4.1: `Win::create` over a self/TCP-only transport fails with
  `MPI_ERR_WIN`, even at a single process; restricting the transport to one
  interface does not help. In that configuration, `Win::allocate` works
  when all ranks share one host (CI's setup); prefer it there.
- Open MPI 4.1.6, for `Win::allocate` windows whose ranks all share one
  host (these use Open MPI's shared-memory one-sided component, `osc/sm`):
  once a window has been used for post-start-complete-wait (PSCW) epochs,
  a further PSCW epoch opened with `WinPscwAssert::no_check()` can let
  `wait_exposure` return before the matching put has landed. Open
  `no_check()` PSCW epochs only on a freshly allocated window.
- Open MPI 4.1's TCP transport, on a host where an extra interface sits
  behind a NAT rule (for example Docker's `docker0` bridge and its
  masquerade rule): sends between ranks on the same host can hang. This
  was observed with consecutive persistent sends. Open MPI spreads
  messages across one TCP path per interface, and messages on the
  NAT-rewritten path are never delivered. Same-host ranks use TCP only
  when shared memory is disabled, as CI does with `OMPI_MCA_btl=self,tcp`.
  Restrict TCP to the interfaces that actually connect the ranks. CI's
  ranks share one host, so it sets `OMPI_MCA_btl_tcp_if_include=lo`. A
  multi-host job must name its cluster interface or subnet instead (for
  example `eth0` or `10.0.0.0/16`), never `lo`.
- If the MPI library refuses to install ferrompi's error handler on a
  newly created window (reported for Open MPI 4.x), ferrompi prints a
  warning to stderr, and RMA errors on that window then abort the process
  instead of returning `Err`.
- When `MPI_Finalize` is skipped because a window is still alive — see the
  `Mpi` type's own documentation for the full rule — MPICH's `mpiexec`
  exits 0. Open MPI 4.1.6 and 5.0.7 instead exit 1 and print an "exiting
  improperly" notice. On every library, drop all windows before the `Mpi`
  handle so `MPI_Finalize` runs; under Open MPI, a skipped finalize makes
  the job exit non-zero, which batch schedulers and CI treat as a failure.
- Fault-tolerant MPI (ULFM): ferrompi exposes no failure-acknowledgement
  API. With the library's process-fault-tolerance mode enabled (Open MPI
  5: `mpiexec --with-ft ulfm`), a receive from any source that a process
  failure leaves pending (`MPI_ERR_PROC_FAILED_PENDING`) cannot complete
  through ferrompi. The `wait` or `test` call that reports it prints
  `ferrompi: receive pending after a process failure` to stderr and aborts
  the process instead of returning. Other failures return `Err` as usual.

## Building

`build.rs` detects the MPI installation through two tiers, tried in order.

**Explicit tier.** The first non-empty variable among `MPI_PKG_CONFIG`,
`MPICC` and `CRAY_MPICH_DIR` wins outright: if its probe fails, the build
panics naming that variable (`MPI_PKG_CONFIG=<name>: ...`), with no
fall-through to the next variable or to auto-detection.

**Auto tier**, tried only when none of the three is set: pkg-config for
`mpich`, then `ompi`, then `mpi`; then `mpicc` on `PATH`; then `/usr`,
`/usr/local`, `/opt/mpich` and `/opt/openmpi`, each checked for
`include/mpi.h` plus a matching library under `lib`, `lib64` or the Debian
multiarch `lib` directory.

A compiler wrapper — `mpicc`, or the path named by `MPICC` — is queried with
`-show` first (MPICH, Intel MPI), then `--showme` (Open MPI). Only its
`-I`, `-L`, `-l` and `-D` tokens are used; every other token is ignored.

- **Open MPI**: auto-detected through pkg-config's `ompi` package, or
  select it explicitly with `MPI_PKG_CONFIG=ompi` or `MPICC=<path>/mpicc`
  (an Open MPI wrapper answers `--showme`).
- **Cray**: the `cray-mpich` module sets `CRAY_MPICH_DIR`. As one of the
  three explicit variables, it outranks auto-detection — it does not
  outrank `MPI_PKG_CONFIG` or `MPICC`, which are checked first. `build.rs`
  uses `$CRAY_MPICH_DIR/lib/pkgconfig/mpich.pc` when it exists, otherwise
  `include/mpi.h` together with `libmpich` or `libmpi` under `lib` or
  `lib64`. Never set `MPICC` to Cray's `cc`: it answers neither `-show` nor
  `--showme`, so the build fails.
- **MPI 5 standard ABI**: set `MPICC` to the implementation's ABI wrapper,
  for example `MPICC=<prefix>/bin/mpicc_abi`. CI builds the examples this
  way against the MPI Forum's reference ABI stubs. Such a build compiles
  and links, but nothing runs it: runtime support is planned for 0.8.0. See
  ADR-0006 (`ferrompi::doc::adr_0006_mpi5_abi_direction`) for the
  direction. The draft ABI — `MPI_ABI_VERSION` defined with `MPI_VERSION`
  below 5, as an `-DMPI_ABI` MPICH 4.3 build produces — is rejected at
  compile time.

Cargo reruns `build.rs`, and therefore recompiles the C shim, whenever
`MPI_PKG_CONFIG`, `MPICC`, `CRAY_MPICH_DIR` or `PATH` changes — including a
`module load` that only changes `PATH`.

Two Cargo features gate optional functionality: `rma` enables RMA and
shared-memory window operations; `numa` implies `rma` and adds the SLURM
helpers (the `slurm` module and `SlurmInfo`), needing no extra system
library.

## Running

`build.rs` embeds no rpath in the binary. If the MPI library is not on the
loader's default search path, a binary built against it fails to start with
an error such as `libmpi.so.12: cannot open shared object file`. Make the
library discoverable at run time: set `LD_LIBRARY_PATH` (`DYLD_LIBRARY_PATH`
on macOS), register its directory with `ldconfig`, or load the cluster's MPI
module.

Launch with the library's own `mpiexec` or `srun`, not another
implementation's.

Run-time costs of ferrompi's own checks:

- `Win::create` and `Win::allocate` each perform one allgather of 8 bytes
  per rank, for the RMA bounds check; `SharedWindow::allocate` does not;
- `Win::allocate` and `SharedWindow::allocate` also zero their segment,
  which adds a lock-all epoch, two barriers, and a `memset` that commits
  every page at construction time. For example, a 64 MiB window at 4 ranks
  took about 60 ms per create-and-free, against about 18 ms for the raw MPI
  calls without ferrompi's zeroing, measured on MPICH 4.2.3 on one host;
- every `Win` data-transfer call adds a local bounds check — no
  communication — costing a few nanoseconds.

### Reporting compatibility

Compatibility reports from implementations outside the three CI builds
above are the primary way this reference improves. When filing a report,
include:

```text
- ferrompi version: (output of `grep '^version' Cargo.toml`)
- MPI implementation and version: (output of `mpiexec --version`)
- OS and distribution:
- Compiler: (output of `mpicc --version`, or the wrapper named by MPICC)
- Cargo features tested: (e.g., default, rma, numa)
- Test command: (e.g., `MPI_NP_LIST=4 ./tests/run_mpi_tests.sh rma`)
- Test results: the runner's summary line, plus any `FAIL <example> (np=N)`
  lines
```
