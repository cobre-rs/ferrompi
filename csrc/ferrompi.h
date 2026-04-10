/**
 * ferrompi.h - Thin C wrapper for MPI 4.x features
 * 
 * This header provides a stable FFI interface for Rust to access
 * MPI functionality, particularly MPI 4.0+ features not available
 * in rsmpi.
 * 
 * Design principles:
 * - Minimal logic in C, just call forwarding
 * - Use fixed-width types for FFI safety
 * - Handle table for opaque MPI objects
 * - Error codes returned directly (MPI_SUCCESS = 0)
 */

#ifndef ferrompi_H
#define ferrompi_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Datatype Tags
 * ============================================================
 * These tags map to MPI_Datatype values in the C layer.
 * They must match the Rust DatatypeTag enum discriminants.
 */

#define FERROMPI_F32  0
#define FERROMPI_F64  1
#define FERROMPI_I32  2
#define FERROMPI_I64  3
#define FERROMPI_U8   4
#define FERROMPI_U32  5
#define FERROMPI_U64  6

/* ============================================================
 * Communicator Split Type Constants
 * ============================================================
 * These constants map to MPI split type values in the C layer.
 * They must match the Rust SplitType enum discriminants.
 */

#define FERROMPI_COMM_TYPE_SHARED 0

/* ============================================================
 * Initialization and Finalization
 * ============================================================ */

/**
 * Initialize MPI with thread support
 * @param required Thread level required (0=SINGLE, 1=FUNNELED, 2=SERIALIZED, 3=MULTIPLE)
 * @param provided Output: actual thread level provided
 * @return MPI error code
 */
int ferrompi_init_thread(int required, int* provided);

/**
 * Initialize MPI (single-threaded)
 * @return MPI error code
 */
int ferrompi_init(void);

/**
 * Finalize MPI
 * @return MPI error code
 */
int ferrompi_finalize(void);

/**
 * Check if MPI is initialized
 * @param flag Output: 1 if initialized, 0 otherwise
 * @return MPI error code
 */
int ferrompi_initialized(int* flag);

/**
 * Check if MPI is finalized
 * @param flag Output: 1 if finalized, 0 otherwise
 * @return MPI error code
 */
int ferrompi_finalized(int* flag);

/* ============================================================
 * Communicator Operations
 * ============================================================ */

/**
 * Get handle to MPI_COMM_WORLD
 * @return Communicator handle (always 0 for COMM_WORLD)
 */
int32_t ferrompi_comm_world(void);

/**
 * Get rank in communicator
 * @param comm Communicator handle
 * @param rank Output: rank of calling process
 * @return MPI error code
 */
int ferrompi_comm_rank(int32_t comm, int32_t* rank);

/**
 * Get size of communicator
 * @param comm Communicator handle
 * @param size Output: number of processes
 * @return MPI error code
 */
int ferrompi_comm_size(int32_t comm, int32_t* size);

/**
 * Duplicate a communicator
 * @param comm Source communicator handle
 * @param newcomm Output: new communicator handle
 * @return MPI error code
 */
int ferrompi_comm_dup(int32_t comm, int32_t* newcomm);

/**
 * Free a communicator
 * @param comm Communicator handle to free
 * @return MPI error code
 */
int ferrompi_comm_free(int32_t comm);

/**
 * Split a communicator into sub-communicators based on color and key.
 * Processes with the same color are placed in the same new communicator.
 * The key controls rank ordering within the new communicator.
 * Pass color=-1 to opt out (maps to MPI_UNDEFINED); newcomm will be set to -1.
 * @param comm Source communicator handle
 * @param color Color value (sub-communicator identifier, or -1 for MPI_UNDEFINED)
 * @param key Key value (rank ordering control)
 * @param newcomm Output: new communicator handle (-1 if process opted out)
 * @return MPI error code
 */
int ferrompi_comm_split(int32_t comm, int32_t color, int32_t key, int32_t* newcomm);

/**
 * Split a communicator by type (e.g., shared memory).
 * Processes that share the same resource (e.g., same physical node for
 * FERROMPI_COMM_TYPE_SHARED) are placed in the same new communicator.
 * The key controls rank ordering within the new communicator.
 * If the split produces MPI_COMM_NULL for this process, newcomm is set to -1.
 * @param comm Source communicator handle
 * @param split_type Split type constant (FERROMPI_COMM_TYPE_SHARED)
 * @param key Key value (rank ordering control)
 * @param newcomm Output: new communicator handle (-1 if MPI_COMM_NULL)
 * @return MPI error code
 */
int ferrompi_comm_split_type(int32_t comm, int32_t split_type, int32_t key, int32_t* newcomm);

/* ============================================================
 * Synchronization
 * ============================================================ */

/**
 * Barrier synchronization
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_barrier(int32_t comm);

/* ============================================================
 * Generic Point-to-Point Communication
 * ============================================================ */

/**
 * Blocking send (generic)
 * @param buf Data buffer
 * @param count Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param dest Destination rank
 * @param tag Message tag
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_send(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm
);

/**
 * Blocking receive (generic)
 * @param buf Receive buffer
 * @param count Maximum number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param source Source rank (or -1 for MPI_ANY_SOURCE)
 * @param tag Message tag (or -1 for MPI_ANY_TAG)
 * @param comm Communicator handle
 * @param actual_source Output: actual source rank
 * @param actual_tag Output: actual tag
 * @param actual_count Output: actual count received
 * @return MPI error code
 */
int ferrompi_recv(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
);

/**
 * Nonblocking send (generic)
 * @param buf Data buffer (must remain valid until request completes)
 * @param count Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param dest Destination rank
 * @param tag Message tag
 * @param comm Communicator handle
 * @param request Output: request handle
 * @return MPI error code
 */
int ferrompi_isend(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm,
    int64_t* request
);

/**
 * Nonblocking receive (generic)
 * @param buf Receive buffer (must remain valid until request completes)
 * @param count Maximum number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param source Source rank (or -1 for MPI_ANY_SOURCE)
 * @param tag Message tag (or -1 for MPI_ANY_TAG)
 * @param comm Communicator handle
 * @param request Output: request handle
 * @return MPI error code
 */
int ferrompi_irecv(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm,
    int64_t* request
);

/**
 * Blocking send-receive (generic)
 *
 * Sends data to one process and receives from another (or the same)
 * in a single operation, avoiding deadlocks.
 *
 * @param sendbuf Data buffer to send
 * @param sendcount Number of elements to send
 * @param send_datatype_tag Send datatype tag
 * @param dest Destination rank
 * @param sendtag Send message tag
 * @param recvbuf Receive buffer
 * @param recvcount Maximum number of elements to receive
 * @param recv_datatype_tag Receive datatype tag
 * @param source Source rank (or -1 for MPI_ANY_SOURCE)
 * @param recvtag Receive message tag (or -1 for MPI_ANY_TAG)
 * @param comm Communicator handle
 * @param actual_source Output: actual source rank
 * @param actual_tag Output: actual tag
 * @param actual_count Output: actual count received
 * @return MPI error code
 */
int ferrompi_sendrecv(
    const void* sendbuf,
    int64_t sendcount,
    int32_t send_datatype_tag,
    int32_t dest,
    int32_t sendtag,
    void* recvbuf,
    int64_t recvcount,
    int32_t recv_datatype_tag,
    int32_t source,
    int32_t recvtag,
    int32_t comm,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
);

/* ============================================================
 * Message Probing
 * ============================================================ */

/**
 * Blocking probe for an incoming message (MPI_Probe).
 *
 * Waits until a matching message is available and returns status
 * information (source, tag, element count) without actually receiving
 * the message. Use this to determine the size of an incoming message
 * before allocating a receive buffer.
 *
 * @param source Source rank (or -1 for MPI_ANY_SOURCE)
 * @param tag Message tag (or -1 for MPI_ANY_TAG)
 * @param comm Communicator handle
 * @param actual_source Output: actual source rank
 * @param actual_tag Output: actual tag
 * @param count Output: number of elements (via MPI_Get_count)
 * @param datatype_tag Datatype tag for MPI_Get_count
 * @return MPI error code
 */
int ferrompi_probe(
    int32_t source,
    int32_t tag,
    int32_t comm,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* count,
    int32_t datatype_tag
);

/**
 * Nonblocking probe for an incoming message (MPI_Iprobe).
 *
 * Checks whether a matching message is available without blocking.
 * If a message is available, sets flag=1 and populates the status
 * fields; otherwise sets flag=0.
 *
 * @param source Source rank (or -1 for MPI_ANY_SOURCE)
 * @param tag Message tag (or -1 for MPI_ANY_TAG)
 * @param comm Communicator handle
 * @param flag Output: 1 if a message is available, 0 otherwise
 * @param actual_source Output: actual source rank (valid only when flag=1)
 * @param actual_tag Output: actual tag (valid only when flag=1)
 * @param count Output: number of elements (valid only when flag=1)
 * @param datatype_tag Datatype tag for MPI_Get_count
 * @return MPI error code
 */
int ferrompi_iprobe(
    int32_t source,
    int32_t tag,
    int32_t comm,
    int32_t* flag,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* count,
    int32_t datatype_tag
);

/* ============================================================
 * Generic Collective Operations - Blocking
 * ============================================================ */

/** Broadcast (generic) */
int ferrompi_bcast(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm);

/** Reduce (generic) */
int ferrompi_reduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t root, int32_t comm);

/**
 * In-place reduce to root (generic).
 *
 * At root (is_root != 0): buf is both input and output (uses MPI_IN_PLACE as sendbuf).
 * At non-root (is_root == 0): buf is the send buffer, recvbuf is ignored.
 *
 * @param buf Data buffer (input on all ranks, output only at root)
 * @param count Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param op Reduction operation
 * @param root Root rank
 * @param is_root Non-zero if this process is the root
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_reduce_inplace(void* buf, int64_t count, int32_t datatype_tag, int32_t op, int32_t root, int32_t is_root, int32_t comm);

/** All-reduce (generic) */
int ferrompi_allreduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm);

/** In-place all-reduce (generic) */
int ferrompi_allreduce_inplace(void* buf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm);

/**
 * Inclusive prefix reduction (scan).
 *
 * On rank i, recvbuf contains the reduction of sendbuf values from ranks 0..=i.
 * Uses MPI_Scan_c for counts exceeding INT_MAX on MPI 4.0+.
 *
 * @param sendbuf Send buffer
 * @param recvbuf Receive buffer (same size as sendbuf)
 * @param count Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param op Reduction operation
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_scan(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm);

/**
 * Exclusive prefix reduction (exscan).
 *
 * On rank i, recvbuf contains the reduction of sendbuf values from ranks 0..i-1.
 * The receive buffer on rank 0 is undefined per the MPI standard.
 * Uses MPI_Exscan_c for counts exceeding INT_MAX on MPI 4.0+.
 *
 * @param sendbuf Send buffer
 * @param recvbuf Receive buffer (same size as sendbuf; undefined on rank 0)
 * @param count Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param op Reduction operation
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_exscan(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm);

/** Gather (generic) */
int ferrompi_gather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm);

/** All-gather (generic) */
int ferrompi_allgather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm);

/** Scatter (generic) */
int ferrompi_scatter(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm);

/**
 * All-to-all personalized communication (MPI_Alltoall).
 *
 * Each process sends sendcount elements to every other process and receives
 * recvcount elements from each. Uses MPI_Alltoall_c for counts exceeding
 * INT_MAX on MPI 4.0+.
 *
 * @param sendbuf Send buffer (sendcount * size elements)
 * @param sendcount Number of elements sent to each process
 * @param recvbuf Receive buffer (recvcount * size elements)
 * @param recvcount Number of elements received from each process
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_alltoall(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm);

/**
 * Reduce-scatter with uniform block size (MPI_Reduce_scatter_block).
 *
 * Equivalent to a reduction followed by a scatter where each process
 * receives the same number of elements (recvcount). The send buffer
 * must contain recvcount * size elements.
 *
 * @param sendbuf Send buffer (recvcount * size elements)
 * @param recvbuf Receive buffer (recvcount elements)
 * @param recvcount Number of elements per process after scatter
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param op Reduction operation
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_reduce_scatter_block(const void* sendbuf, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t op, int32_t comm);

/* ============================================================
 * Generic V-Collectives (variable-count)
 * ============================================================
 * V-collectives allow each rank to send/receive a different number of
 * elements. Counts and displacement arrays use int32_t (matching the MPI
 * standard `int` on all supported platforms).
 */

/**
 * Gather variable amounts of data to root (MPI_Gatherv).
 * @param sendbuf Send buffer
 * @param sendcount Number of elements to send from this rank
 * @param recvbuf Receive buffer (significant only at root)
 * @param recvcounts Array of length size: recvcounts[i] = elements from rank i
 * @param displs Array of length size: displacement in recvbuf for rank i
 * @param datatype_tag Datatype tag
 * @param root Root rank
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_gatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t root, int32_t comm
);

/**
 * Scatter variable amounts of data from root (MPI_Scatterv).
 * @param sendbuf Send buffer (significant only at root)
 * @param sendcounts Array of length size: sendcounts[i] = elements to rank i
 * @param displs Array of length size: displacement in sendbuf for rank i
 * @param recvbuf Receive buffer
 * @param recvcount Number of elements to receive at this rank
 * @param datatype_tag Datatype tag
 * @param root Root rank
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_scatterv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* displs,
    void* recvbuf, int64_t recvcount,
    int32_t datatype_tag, int32_t root, int32_t comm
);

/**
 * All-gather variable amounts of data (MPI_Allgatherv).
 * @param sendbuf Send buffer
 * @param sendcount Number of elements to send from this rank
 * @param recvbuf Receive buffer
 * @param recvcounts Array of length size: recvcounts[i] = elements from rank i
 * @param displs Array of length size: displacement in recvbuf for rank i
 * @param datatype_tag Datatype tag
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_allgatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t comm
);

/**
 * All-to-all with variable counts (MPI_Alltoallv).
 * @param sendbuf Send buffer
 * @param sendcounts Array of length size: sendcounts[i] = elements to rank i
 * @param sdispls Array of length size: send displacement for rank i
 * @param recvbuf Receive buffer
 * @param recvcounts Array of length size: recvcounts[i] = elements from rank i
 * @param rdispls Array of length size: receive displacement for rank i
 * @param datatype_tag Datatype tag
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_alltoallv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls,
    void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls,
    int32_t datatype_tag, int32_t comm
);

/* ============================================================
 * Generic Collective Operations - Nonblocking
 * ============================================================ */

/** Nonblocking broadcast (generic) */
int ferrompi_ibcast(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Nonblocking all-reduce (generic) */
int ferrompi_iallreduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Nonblocking reduce (generic) */
int ferrompi_ireduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t root, int32_t comm, int64_t* request);

/** Nonblocking gather (generic) */
int ferrompi_igather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Nonblocking all-gather (generic) */
int ferrompi_iallgather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm, int64_t* request);

/** Nonblocking scatter (generic) */
int ferrompi_iscatter(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Nonblocking barrier */
int ferrompi_ibarrier(int32_t comm, int64_t* request);

/** Nonblocking inclusive scan (generic) */
int ferrompi_iscan(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Nonblocking exclusive scan (generic) */
int ferrompi_iexscan(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Nonblocking all-to-all (generic) */
int ferrompi_ialltoall(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm, int64_t* request);

/** Nonblocking gatherv (generic, variable-count) */
int ferrompi_igatherv(const void* sendbuf, int64_t sendcount, void* recvbuf, const int32_t* recvcounts, const int32_t* displs, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Nonblocking scatterv (generic, variable-count) */
int ferrompi_iscatterv(const void* sendbuf, const int32_t* sendcounts, const int32_t* displs, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Nonblocking all-gatherv (generic, variable-count) */
int ferrompi_iallgatherv(const void* sendbuf, int64_t sendcount, void* recvbuf, const int32_t* recvcounts, const int32_t* displs, int32_t datatype_tag, int32_t comm, int64_t* request);

/** Nonblocking all-to-allv (generic, variable-count) */
int ferrompi_ialltoallv(const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls, void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls, int32_t datatype_tag, int32_t comm, int64_t* request);

/** Nonblocking reduce-scatter with uniform block size (generic) */
int ferrompi_ireduce_scatter_block(const void* sendbuf, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/* ============================================================
 * Generic Persistent Collectives (MPI 4.0+)
 * ============================================================ */

/** Initialize persistent broadcast (generic) */
int ferrompi_bcast_init(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent all-reduce (generic) */
int ferrompi_allreduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent all-reduce in-place (generic) */
int ferrompi_allreduce_init_inplace(void* buf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent gather (generic) */
int ferrompi_gather_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent reduce (generic) */
int ferrompi_reduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent scatter (generic) */
int ferrompi_scatter_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent all-gather (generic) */
int ferrompi_allgather_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm, int64_t* request);

/** Initialize persistent scan (generic) */
int ferrompi_scan_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent exclusive scan (generic) */
int ferrompi_exscan_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent all-to-all (generic) */
int ferrompi_alltoall_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm, int64_t* request);

/** Initialize persistent gatherv (generic, variable-count) */
int ferrompi_gatherv_init(const void* sendbuf, int64_t sendcount, void* recvbuf, const int32_t* recvcounts, const int32_t* displs, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent scatterv (generic, variable-count) */
int ferrompi_scatterv_init(const void* sendbuf, const int32_t* sendcounts, const int32_t* displs, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent all-gatherv (generic, variable-count) */
int ferrompi_allgatherv_init(const void* sendbuf, int64_t sendcount, void* recvbuf, const int32_t* recvcounts, const int32_t* displs, int32_t datatype_tag, int32_t comm, int64_t* request);

/** Initialize persistent all-to-allv (generic, variable-count) */
int ferrompi_alltoallv_init(const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls, void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls, int32_t datatype_tag, int32_t comm, int64_t* request);

/** Initialize persistent reduce-scatter with uniform block size (generic) */
int ferrompi_reduce_scatter_block_init(const void* sendbuf, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/* ============================================================
 * Info Object Operations
 * ============================================================ */

/**
 * Create a new MPI_Info object
 * @param info_handle Output: info handle
 * @return MPI error code
 */
int ferrompi_info_create(int32_t* info_handle);

/**
 * Free an MPI_Info object
 * @param info_handle Info handle to free (no-op if invalid or already freed)
 * @return MPI error code
 */
int ferrompi_info_free(int32_t info_handle);

/**
 * Set a key-value pair on an MPI_Info object
 * @param info_handle Info handle
 * @param key Null-terminated key string
 * @param value Null-terminated value string
 * @return MPI error code
 */
int ferrompi_info_set(int32_t info_handle, const char* key, const char* value);

/**
 * Get a value by key from an MPI_Info object
 * @param info_handle Info handle
 * @param key Null-terminated key string
 * @param value Output buffer for value string
 * @param valuelen Input/output: buffer size on input, actual length on output
 * @param flag Output: 1 if key was found, 0 otherwise
 * @return MPI error code
 */
int ferrompi_info_get(int32_t info_handle, const char* key, char* value, int32_t* valuelen, int32_t* flag);

/* ============================================================
 * Error Information
 * ============================================================ */

/**
 * Get error class and message string for an MPI error code
 * @param code MPI error code
 * @param error_class Output: error class
 * @param message Output: error message (at least MPI_MAX_ERROR_STRING bytes)
 * @param msg_len Output: actual message length
 * @return MPI error code
 */
int ferrompi_error_info(int code, int32_t* error_class, char* message, int32_t* msg_len);

/* ============================================================
 * Request Management
 * ============================================================ */

/**
 * Wait for request completion
 * @param request Request handle
 * @return MPI error code
 */
int ferrompi_wait(int64_t request);

/**
 * Test if request is complete
 * @param request Request handle
 * @param flag Output: 1 if complete, 0 otherwise
 * @return MPI error code
 */
int ferrompi_test(int64_t request, int32_t* flag);

/**
 * Wait for all requests
 * @param count Number of requests
 * @param requests Array of request handles
 * @return MPI error code
 */
int ferrompi_waitall(int64_t count, int64_t* requests);

/**
 * Free a request handle
 * @param request Request handle to free
 * @return MPI error code
 */
int ferrompi_request_free(int64_t request);

/**
 * Start a persistent request
 * @param request Persistent request handle
 * @return MPI error code
 */
int ferrompi_start(int64_t request);

/**
 * Start multiple persistent requests
 * @param count Number of requests
 * @param requests Array of request handles
 * @return MPI error code
 */
int ferrompi_startall(int64_t count, int64_t* requests);

/* ============================================================
 * RMA Window Operations (MPI 3.0+)
 * ============================================================
 *
 * Window handle table supports up to 256 concurrent MPI_Win objects.
 */

/* Lock type constants for MPI_Win_lock / MPI_Win_lock_all */
#define FERROMPI_LOCK_EXCLUSIVE 0
#define FERROMPI_LOCK_SHARED    1

/**
 * Allocate a shared-memory MPI window (MPI_Win_allocate_shared).
 *
 * All processes in the communicator collectively allocate shared memory.
 * The communicator must be created via MPI_Comm_split_type with
 * MPI_COMM_TYPE_SHARED so all ranks share a physical memory region.
 *
 * @param size Number of bytes to allocate on this rank
 * @param disp_unit Displacement unit in bytes (e.g., sizeof(double))
 * @param info Info handle (negative for MPI_INFO_NULL)
 * @param comm Communicator handle (must be shared-memory communicator)
 * @param baseptr Output: pointer to allocated memory
 * @param win Output: window handle
 * @return MPI error code
 */
int ferrompi_win_allocate_shared(int64_t size, int32_t disp_unit, int32_t info,
                                  int32_t comm, void** baseptr, int32_t* win);

/**
 * Query the shared-memory region of another rank (MPI_Win_shared_query).
 *
 * Returns the base pointer, size, and displacement unit of the shared
 * memory segment belonging to the specified rank in the window.
 *
 * @param win Window handle
 * @param rank Rank to query
 * @param size Output: size in bytes of the segment at the queried rank
 * @param disp_unit Output: displacement unit at the queried rank
 * @param baseptr Output: pointer to the queried rank's shared memory
 * @return MPI error code
 */
int ferrompi_win_shared_query(int32_t win, int32_t rank,
                               int64_t* size, int32_t* disp_unit, void** baseptr);

/**
 * Free an MPI window (MPI_Win_free).
 *
 * Releases the window and its associated resources. The window handle
 * slot is freed for reuse. No-op if the handle is invalid.
 *
 * @param win Window handle to free
 * @return MPI error code
 */
int ferrompi_win_free(int32_t win);

/**
 * Perform a fence synchronization on a window (MPI_Win_fence).
 *
 * Synchronizes RMA calls on the window. Used for active target
 * (fence-based) synchronization epochs.
 *
 * @param assert_val Assertion hint (0 for no assertion)
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_fence(int32_t assert_val, int32_t win);

/**
 * Lock a window at a target rank (MPI_Win_lock).
 *
 * Starts a passive target access epoch for RMA operations on the
 * specified target rank. Must be paired with ferrompi_win_unlock.
 *
 * @param lock_type Lock type (FERROMPI_LOCK_EXCLUSIVE or FERROMPI_LOCK_SHARED)
 * @param rank Target rank
 * @param assert_val Assertion hint (0 for no assertion)
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_lock(int32_t lock_type, int32_t rank, int32_t assert_val, int32_t win);

/**
 * Unlock a window at a target rank (MPI_Win_unlock).
 *
 * Completes the passive target access epoch started by ferrompi_win_lock.
 *
 * @param rank Target rank
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_unlock(int32_t rank, int32_t win);

/**
 * Lock a window at all ranks (MPI_Win_lock_all).
 *
 * Starts a shared passive target access epoch for all ranks.
 * Must be paired with ferrompi_win_unlock_all.
 *
 * @param assert_val Assertion hint (0 for no assertion)
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_lock_all(int32_t assert_val, int32_t win);

/**
 * Unlock a window at all ranks (MPI_Win_unlock_all).
 *
 * Completes the shared passive target access epoch started by
 * ferrompi_win_lock_all.
 *
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_unlock_all(int32_t win);

/**
 * Flush pending RMA operations to a target rank (MPI_Win_flush).
 *
 * Ensures that all RMA operations issued to the target rank have
 * completed at both origin and target. The access epoch is not closed.
 *
 * @param rank Target rank
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_flush(int32_t rank, int32_t win);

/**
 * Flush pending RMA operations to all ranks (MPI_Win_flush_all).
 *
 * Ensures that all RMA operations issued to any target have completed
 * at both origin and target. The access epoch is not closed.
 *
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_flush_all(int32_t win);

/* ============================================================
 * Utility Functions
 * ============================================================ */

/**
 * Get MPI library version string (implementation-specific, e.g. "Open MPI v4.1.6")
 * @param buf Output buffer (at least MPI_MAX_LIBRARY_VERSION_STRING bytes)
 * @param len Output: actual length
 * @return MPI error code
 */
int ferrompi_get_library_version(char* buf, int32_t* len);

/**
 * Get MPI standard version string (e.g. "MPI 4.0")
 * @param version Output buffer (at least 256 bytes)
 * @param len Output: actual length
 * @return MPI error code
 */
int ferrompi_get_version(char* version, int32_t* len);

/**
 * Get processor name
 * @param name Output buffer (at least MPI_MAX_PROCESSOR_NAME bytes)
 * @param len Output: actual length
 * @return MPI error code
 */
int ferrompi_get_processor_name(char* name, int32_t* len);

/**
 * Get wall clock time
 * @return Wall clock time in seconds
 */
double ferrompi_wtime(void);

/**
 * Abort MPI execution
 * @param comm Communicator handle
 * @param errorcode Error code to return
 * @return Does not return
 */
int ferrompi_abort(int32_t comm, int32_t errorcode);

/* ============================================================
 * Error Class Constants
 * ============================================================ */

/** Get the MPI_ERR_FILE error class value (implementation-specific). */
int32_t ferrompi_err_file(void);
/** Get the MPI_ERR_INFO error class value (implementation-specific). */
int32_t ferrompi_err_info(void);
/** Get the MPI_ERR_WIN error class value (implementation-specific). */
int32_t ferrompi_err_win(void);

#ifdef __cplusplus
}
#endif

#endif /* ferrompi_H */
