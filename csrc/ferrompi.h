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

/* Paired value+index types for MPI_MAXLOC / MPI_MINLOC.
 * These must match the Rust DatatypeTag enum discriminants 7-12. */
#define FERROMPI_FLOAT_INT        7
#define FERROMPI_DOUBLE_INT       8
#define FERROMPI_LONG_INT         9
#define FERROMPI_2INT            10
#define FERROMPI_SHORT_INT       11
#define FERROMPI_LONG_DOUBLE_INT 12

/* Opaque 1-byte unit for type-erased bitwise reductions (MPI_BYTE).
 * Must match Rust DatatypeTag::Byte = 13. */
#define FERROMPI_BYTE            13

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

/**
 * Create a sub-communicator from a parent communicator and a group (MPI_Comm_create).
 *
 * This is a collective operation over the parent communicator — every rank in
 * comm_handle must call it, even ranks not in group_handle.
 *
 * Ranks that are members of group_handle receive a new communicator handle.
 * Ranks not in group_handle receive newcomm_handle = -1 (MPI returns
 * MPI_COMM_NULL for those ranks).
 *
 * The "parent" suffix disambiguates from the MPI 4.0+ MPI_Comm_create_from_group
 * (ticket-050), which does not require a parent communicator.
 *
 * @param comm_handle   Parent communicator handle
 * @param group_handle  Group handle (must be a subset of comm_handle's group)
 * @param newcomm_handle Output: new communicator handle, or -1 if not in group
 * @return MPI error code
 */
int ferrompi_comm_create_from_group_parent(int32_t comm_handle,
                                           int32_t group_handle,
                                           int32_t* newcomm_handle);

/**
 * Create a communicator from a group without requiring a parent communicator
 * (MPI 4.0+ only, MPI_Comm_create_from_group).
 *
 * This is NOT a collective over an existing communicator — the call is
 * collective only over the processes that share the same group and stringtag.
 * Ranks with different tags or in different groups produce separate
 * communicators.
 *
 * Returns MPI_ERR_OTHER on MPI < 4.0 (caller maps to Error::NotSupported).
 *
 * @param group_handle  Group handle
 * @param stringtag     Null-terminated string tag (disambiguates concurrent calls)
 * @param newcomm_handle Output: new communicator handle
 * @return MPI error code (MPI_ERR_OTHER on MPI < 4.0)
 */
int ferrompi_comm_create_from_group(int32_t group_handle,
                                    const char* stringtag,
                                    int32_t* newcomm_handle);

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

/** In-place gather (generic). Valid only at root; non-root returns MPI_ERR_ARG. */
int ferrompi_gather_inplace(void* recvbuf, int64_t recvcount,
                             int32_t datatype_tag, int32_t root,
                             int32_t is_root, int32_t comm);

/** In-place all-gather (generic). Valid at every rank. */
int ferrompi_allgather_inplace(void* recvbuf, int64_t recvcount,
                                int32_t datatype_tag, int32_t comm);

/**
 * In-place scatter (generic).
 *
 * At root (is_root != 0): sendbuf is the full sendcount*size buffer; MPI_IN_PLACE
 * is passed as recvbuf so root's own slot is retained in place. recvbuf is ignored.
 * At non-root (is_root == 0): regular scatter path; sendbuf is ignored (NULL), recvbuf
 * receives recvcount elements.
 *
 * @param sendbuf Send buffer (significant only at root)
 * @param sendcount Number of elements sent to each process (significant only at root)
 * @param recvbuf Receive buffer (significant only at non-root; ignored at root)
 * @param recvcount Number of elements to receive (significant only at non-root)
 * @param datatype_tag Datatype tag
 * @param root Root rank
 * @param is_root Non-zero if this process is the root
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_scatter_inplace(const void* sendbuf, int64_t sendcount,
                              void* recvbuf, int64_t recvcount,
                              int32_t datatype_tag, int32_t root,
                              int32_t is_root, int32_t comm);

/**
 * In-place all-to-all personalized communication (generic).
 *
 * recvbuf serves as both send and receive buffer (MPI_IN_PLACE as sendbuf).
 * Before the call, rank r must pre-write into slot s the payload destined for rank s.
 * After the call, slot s contains the data received FROM rank s.
 *
 * @param recvbuf Combined send/receive buffer (recvcount * size elements)
 * @param recvcount Number of elements per rank
 * @param datatype_tag Datatype tag
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_alltoall_inplace(void* recvbuf, int64_t recvcount,
                               int32_t datatype_tag, int32_t comm);

/** Nonblocking in-place gather (generic). Valid only at root (is_root != 0); non-root returns MPI_ERR_ARG. */
int ferrompi_igather_inplace(void* recvbuf, int64_t recvcount,
                              int32_t datatype_tag, int32_t root,
                              int32_t is_root, int32_t comm,
                              int64_t* request);

/** Nonblocking in-place all-gather (generic). Valid at every rank. */
int ferrompi_iallgather_inplace(void* recvbuf, int64_t recvcount,
                                 int32_t datatype_tag, int32_t comm,
                                 int64_t* request);

/**
 * Nonblocking in-place scatter (generic).
 *
 * At root (is_root != 0): sendbuf is the full sendcount*size buffer; MPI_IN_PLACE
 * is passed as recvbuf so root's own slot is retained in place.
 * At non-root (is_root == 0): regular scatter path; sendbuf is NULL, recvbuf
 * receives recvcount elements.
 */
int ferrompi_iscatter_inplace(const void* sendbuf, int64_t sendcount,
                               void* recvbuf, int64_t recvcount,
                               int32_t datatype_tag, int32_t root,
                               int32_t is_root, int32_t comm,
                               int64_t* request);

/** Nonblocking in-place all-to-all (generic). Valid at every rank. */
int ferrompi_ialltoall_inplace(void* recvbuf, int64_t recvcount,
                                int32_t datatype_tag, int32_t comm,
                                int64_t* request);

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
 * Persistent Point-to-Point (MPI 1.1+)
 * ============================================================ */

/**
 * Initialize a persistent send (MPI_Send_init).
 *
 * The returned request handle must be started with ferrompi_start and
 * completed with ferrompi_wait. It may be restarted multiple times.
 * The caller must not modify buf between start and wait.
 *
 * @param buf          Send buffer
 * @param count        Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F64, etc.)
 * @param dest         Destination rank
 * @param tag          Message tag
 * @param comm_handle  Communicator handle
 * @param request_handle Output: persistent request handle
 * @return MPI error code
 */
int ferrompi_send_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
);

/**
 * Initialize a persistent receive (MPI_Recv_init).
 *
 * The returned request handle must be started with ferrompi_start and
 * completed with ferrompi_wait. It may be restarted multiple times.
 * Use source=-1 for MPI_ANY_SOURCE and tag=-1 for MPI_ANY_TAG.
 *
 * @param buf          Receive buffer (must remain valid for lifetime of handle)
 * @param count        Maximum number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F64, etc.)
 * @param source       Source rank (or -1 for MPI_ANY_SOURCE)
 * @param tag          Message tag (or -1 for MPI_ANY_TAG)
 * @param comm_handle  Communicator handle
 * @param request_handle Output: persistent request handle
 * @return MPI error code
 */
int ferrompi_recv_init(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
);

/**
 * Initialize a persistent ready-mode send (MPI_Rsend_init).
 *
 * Ready-mode sends require that the matching receive has already been posted
 * on the destination rank before start() is called. This is an optimization
 * hint to the MPI implementation to skip protocol negotiation. Calling
 * start() before the matching receive is posted is undefined behavior in MPI.
 *
 * The returned request handle must be started with ferrompi_start and
 * completed with ferrompi_wait. It may be restarted multiple times.
 * The caller must not modify buf between start and wait.
 *
 * @param buf          Send buffer
 * @param count        Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F64, etc.)
 * @param dest         Destination rank
 * @param tag          Message tag
 * @param comm_handle  Communicator handle
 * @param request_handle Output: persistent request handle
 * @return MPI error code
 */
int ferrompi_rsend_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
);

/**
 * Initialize a persistent synchronous-mode send (MPI_Ssend_init).
 *
 * Synchronous-mode sends complete only after the matching receive has begun.
 * Unlike standard sends, the MPI implementation cannot buffer the message
 * internally, so the sender blocks in wait() until the receiver has called
 * start() on its matching receive. This eliminates the possibility of silent
 * buffering, making it useful for debugging deadlocks and for algorithms that
 * require strict sender/receiver handshake semantics.
 *
 * The returned request handle must be started with ferrompi_start and
 * completed with ferrompi_wait. It may be restarted multiple times.
 * The caller must not modify buf between start and wait.
 *
 * @param buf          Send buffer
 * @param count        Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F64, etc.)
 * @param dest         Destination rank
 * @param tag          Message tag
 * @param comm_handle  Communicator handle
 * @param request_handle Output: persistent request handle
 * @return MPI error code
 */
int ferrompi_ssend_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
);

/**
 * Attach a user buffer for use by buffered sends (MPI_Buffer_attach).
 *
 * Only one buffer may be attached per process at a time. The size parameter
 * is cast to int; buffers larger than INT_MAX bytes will truncate or fail.
 *
 * @param buffer  Pointer to the buffer (must remain valid until detach)
 * @param size    Size of the buffer in bytes
 * @return MPI error code
 */
int ferrompi_buffer_attach(void* buffer, int64_t size);

/**
 * Detach the previously attached buffer (MPI_Buffer_detach).
 *
 * Blocks until all buffered sends using the buffer have completed.
 *
 * @param buffer  Output: pointer to the detached buffer
 * @param size    Output: size of the detached buffer in bytes
 * @return MPI error code
 */
int ferrompi_buffer_detach(void** buffer, int64_t* size);

/**
 * Initialize a persistent buffered-mode send (MPI_Bsend_init).
 *
 * Buffered sends require a user buffer to be attached via
 * ferrompi_buffer_attach before start() is called. The recommended
 * buffer size is MPI_BSEND_OVERHEAD + sum(send sizes).
 *
 * The returned request handle must be started with ferrompi_start and
 * completed with ferrompi_wait. It may be restarted multiple times.
 * The caller must not modify buf between start and wait.
 *
 * @param buf          Send buffer
 * @param count        Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F64, etc.)
 * @param dest         Destination rank
 * @param tag          Message tag
 * @param comm_handle  Communicator handle
 * @param request_handle Output: persistent request handle
 * @return MPI error code
 */
int ferrompi_bsend_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
);

/* ============================================================
 * Generic Persistent Collectives (MPI 4.0+)
 * ============================================================ */

/** Initialize persistent broadcast (generic) */
int ferrompi_bcast_init(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent all-reduce (generic) */
int ferrompi_allreduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent all-reduce in-place (generic) */
int ferrompi_allreduce_init_inplace(void* buf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent in-place gather (generic). Valid only at root (is_root != 0); non-root returns MPI_ERR_ARG. */
int ferrompi_gather_init_inplace(void* recvbuf, int64_t recvcount,
                                  int32_t datatype_tag, int32_t root,
                                  int32_t is_root, int32_t comm,
                                  int64_t* request);

/** Initialize persistent in-place all-gather (generic). Valid at every rank. */
int ferrompi_allgather_init_inplace(void* recvbuf, int64_t recvcount,
                                     int32_t datatype_tag, int32_t comm,
                                     int64_t* request);

/**
 * Initialize persistent in-place scatter (generic).
 *
 * At root (is_root != 0): sendbuf is the full sendcount*size buffer; MPI_IN_PLACE
 * is passed as recvbuf so root's own slot is retained in place.
 * At non-root (is_root == 0): sendbuf is NULL, recvbuf receives recvcount elements.
 */
int ferrompi_scatter_init_inplace(const void* sendbuf, int64_t sendcount,
                                   void* recvbuf, int64_t recvcount,
                                   int32_t datatype_tag, int32_t root,
                                   int32_t is_root, int32_t comm,
                                   int64_t* request);

/** Initialize persistent in-place all-to-all (generic). Valid at every rank. */
int ferrompi_alltoall_init_inplace(void* recvbuf, int64_t recvcount,
                                    int32_t datatype_tag, int32_t comm,
                                    int64_t* request);

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
 * Non-destructive status query (MPI_Request_get_status).
 * Sets *flag = 1 if the request is complete, 0 otherwise.
 * Does NOT free the request handle.
 * @param request Request handle to query
 * @param flag    Output: 1 if complete, 0 otherwise
 * @return MPI error code
 */
int ferrompi_request_get_status(int64_t request, int32_t* flag);

/**
 * Request cancellation of a pending nonblocking operation (MPI_Cancel).
 * Does NOT free the request handle; the caller must still call wait.
 * @param request Request handle to cancel
 * @return MPI error code
 */
int ferrompi_cancel(int64_t request);

/**
 * Wait for any one request to complete (MPI_Waitany).
 * @param count    Number of requests
 * @param requests Array of request handles (updated in place)
 * @param index    Output: index of completed request, or -1 if all null
 * @return MPI error code
 */
int ferrompi_waitany(int64_t count, int64_t* requests, int32_t* index);

/**
 * Wait until at least one request completes (MPI_Waitsome).
 * @param count    Number of requests
 * @param requests Array of request handles (updated in place)
 * @param outcount Output: number of completed requests, or -1 if all null
 * @param indices  Output array (length >= count): indices of completed requests
 * @return MPI error code
 */
int ferrompi_waitsome(int64_t count, int64_t* requests, int64_t* outcount, int32_t* indices);

/**
 * Test if any one request has completed (MPI_Testany).
 * @param count    Number of requests
 * @param requests Array of request handles (updated in place)
 * @param index    Output: index of completed request, or -1 if all null / none done
 * @param flag     Output: 1 if a request completed (or all null), 0 otherwise
 * @return MPI error code
 */
int ferrompi_testany(int64_t count, int64_t* requests, int32_t* index, int32_t* flag);

/**
 * Test how many requests have completed (MPI_Testsome).
 * @param count    Number of requests
 * @param requests Array of request handles (updated in place)
 * @param outcount Output: number completed, 0 if none, -1 if all null
 * @param indices  Output array (length >= count): indices of completed requests
 * @return MPI error code
 */
int ferrompi_testsome(int64_t count, int64_t* requests, int64_t* outcount, int32_t* indices);

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
 * Create a general-purpose RMA window over a caller-supplied buffer (MPI_Win_create).
 *
 * @param base      Pointer to the local window buffer (may be NULL for zero-size windows)
 * @param size      Size in bytes of the local window buffer
 * @param disp_unit Displacement unit in bytes (typically sizeof(T))
 * @param info      Info handle (negative for MPI_INFO_NULL)
 * @param comm      Communicator handle
 * @param win       Output: window handle
 * @return MPI error code
 */
int ferrompi_win_create(void* base, int64_t size, int32_t disp_unit, int32_t info,
                         int32_t comm, int32_t* win);

/**
 * Allocate a general-purpose RMA window with MPI-managed memory (MPI_Win_allocate).
 *
 * @param size      Size in bytes of the local window buffer
 * @param disp_unit Displacement unit in bytes (typically sizeof(T))
 * @param info      Info handle (negative for MPI_INFO_NULL)
 * @param comm      Communicator handle
 * @param baseptr   Output: pointer to the MPI-allocated memory
 * @param win       Output: window handle
 * @return MPI error code
 */
int ferrompi_win_allocate(int64_t size, int32_t disp_unit, int32_t info,
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
 * Query the MPI fence assertion mode constants.
 *
 * Writes the four fence assertion constants into a 4-element array:
 *   out[0] = MPI_MODE_NOSTORE
 *   out[1] = MPI_MODE_NOPUT
 *   out[2] = MPI_MODE_NOPRECEDE
 *   out[3] = MPI_MODE_NOSUCCEED
 *
 * The values are implementation-defined and must be queried at runtime.
 *
 * @param out 4-element int32_t array to receive the constants
 */
void ferrompi_win_fence_mode_values(int32_t* out);

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

/**
 * Locally complete pending RMA operations to a target rank (MPI_Win_flush_local).
 *
 * Ensures that all RMA operations issued to the target rank have completed
 * locally (the origin buffer is safe to reuse), but the remote may not yet
 * have observed the writes. The access epoch is not closed.
 *
 * @param rank Target rank
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_flush_local(int32_t rank, int32_t win);

/**
 * Locally complete pending RMA operations to all ranks (MPI_Win_flush_local_all).
 *
 * Ensures that all RMA operations issued to any target have completed locally.
 * The access epoch is not closed.
 *
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_flush_local_all(int32_t win);

/**
 * Memory barrier between the local public window copy and private copy (MPI_Win_sync).
 *
 * Synchronizes the private and public copies of the window. This is a local
 * operation — it does not require a surrounding epoch.
 *
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_sync(int32_t win);

/**
 * Start an exposure epoch on a window (MPI_Win_post).
 *
 * Called by the target ranks to expose their window to the access group.
 * Must be matched by a corresponding ferrompi_win_wait or ferrompi_win_test.
 *
 * @param group  Group of ranks that will issue RMA operations against this window
 * @param assert_val Assertion hint bitmask (0 for no assertion)
 * @param win    Window handle
 * @return MPI error code
 */
int ferrompi_win_post(int32_t group, int32_t assert_val, int32_t win);

/**
 * Start an access epoch on a window (MPI_Win_start).
 *
 * Called by the origin ranks to begin issuing RMA operations. The group
 * identifies the ranks that have called or will call ferrompi_win_post.
 * Must be matched by a corresponding ferrompi_win_complete.
 *
 * @param group  Group of ranks that have called or will call Win_post
 * @param assert_val Assertion hint bitmask (0 for no assertion)
 * @param win    Window handle
 * @return MPI error code
 */
int ferrompi_win_start(int32_t group, int32_t assert_val, int32_t win);

/**
 * Complete an access epoch on a window (MPI_Win_complete).
 *
 * Closes the access epoch started by ferrompi_win_start. All RMA
 * operations issued since the matching Win_start are completed.
 *
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_complete(int32_t win);

/**
 * Complete an exposure epoch on a window (MPI_Win_wait).
 *
 * Blocks until all RMA operations issued by the access group are complete.
 * Closes the exposure epoch started by ferrompi_win_post.
 *
 * @param win Window handle
 * @return MPI error code
 */
int ferrompi_win_wait(int32_t win);

/**
 * Test whether an exposure epoch has completed (MPI_Win_test).
 *
 * Nonblocking probe: returns immediately and sets *flag to 1 if the
 * exposure epoch started by ferrompi_win_post has completed, 0 otherwise.
 *
 * @param win  Window handle
 * @param flag Output: set to 1 if epoch is complete, 0 otherwise
 * @return MPI error code
 */
int ferrompi_win_test(int32_t win, int32_t* flag);

/**
 * Query the MPI PSCW assertion mode constants.
 *
 * Writes the three PSCW assertion constants into a 3-element array:
 *   out[0] = MPI_MODE_NOCHECK
 *   out[1] = MPI_MODE_NOSTORE
 *   out[2] = MPI_MODE_NOPUT
 *
 * The values are implementation-defined and must be queried at runtime.
 *
 * @param out 3-element int32_t array to receive the constants
 */
void ferrompi_win_pscw_mode_values(int32_t* out);

/**
 * One-sided write: copy data from a local origin buffer into a remote rank's
 * window memory.
 *
 * Wraps MPI_Put. The operation completes when the surrounding epoch closes
 * (fence, complete, or unlock). On MPI < 3 builds the stub returns
 * MPI_ERR_OTHER.
 *
 * @param origin       Local origin buffer (read-only).
 * @param origin_count Number of elements in the origin buffer.
 * @param origin_dt_tag DatatypeTag for the origin element type.
 * @param target_rank  Destination rank.
 * @param target_disp  Displacement in units of the window's disp_unit.
 * @param target_count Number of elements to write at the target.
 * @param target_dt_tag DatatypeTag for the target element type.
 * @param win_handle   Window handle.
 * @return MPI error code (MPI_SUCCESS = 0).
 */
int ferrompi_put(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                 int32_t target_rank, int64_t target_disp, int64_t target_count,
                 int32_t target_dt_tag, int32_t win_handle);

/**
 * Request-returning one-sided write: copy data from a local origin buffer into
 * a remote rank's window memory and return a request handle for local-buffer
 * completion tracking.
 *
 * Wraps MPI_Rput. The returned request completes when the local origin buffer
 * is safe to reuse — NOT when the remote rank has observed the write. The
 * remote completion still requires closing the surrounding epoch (fence,
 * complete, or unlock). On MPI < 3 builds the stub returns MPI_ERR_OTHER.
 *
 * @param origin          Local origin buffer (read-only).
 * @param origin_count    Number of elements in the origin buffer.
 * @param origin_dt_tag   DatatypeTag for the origin element type.
 * @param target_rank     Destination rank.
 * @param target_disp     Displacement in units of the window's disp_unit.
 * @param target_count    Number of elements to write at the target.
 * @param target_dt_tag   DatatypeTag for the target element type.
 * @param win_handle      Window handle.
 * @param request_handle  Output: request table index for the new request.
 * @return MPI error code (MPI_SUCCESS = 0), or MPI_ERR_OTHER if the request
 *         table is exhausted.
 */
int ferrompi_rput(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                  int32_t target_rank, int64_t target_disp, int64_t target_count,
                  int32_t target_dt_tag, int32_t win_handle, int64_t* request_handle);

/**
 * One-sided read: copy data from a remote rank's window memory into a local
 * origin buffer.
 *
 * Wraps MPI_Get. The operation completes when the surrounding epoch closes
 * (fence, complete, or unlock). On MPI < 3 builds the stub returns
 * MPI_ERR_OTHER.
 *
 * @param origin       Local destination buffer (write destination).
 * @param origin_count Number of elements in the origin buffer.
 * @param origin_dt_tag DatatypeTag for the origin element type.
 * @param target_rank  Source rank.
 * @param target_disp  Displacement in units of the window's disp_unit.
 * @param target_count Number of elements to read from the target.
 * @param target_dt_tag DatatypeTag for the target element type.
 * @param win_handle   Window handle.
 * @return MPI error code (MPI_SUCCESS = 0).
 */
int ferrompi_get(void* origin, int64_t origin_count, int32_t origin_dt_tag,
                 int32_t target_rank, int64_t target_disp, int64_t target_count,
                 int32_t target_dt_tag, int32_t win_handle);

/**
 * Request-returning one-sided read: copy data from a remote rank's window
 * memory into a local origin buffer and return a request handle for
 * local-buffer completion tracking.
 *
 * Wraps MPI_Rget. When the returned request completes (via MPI_Wait or
 * equivalent), the data IS guaranteed to be in the local origin buffer —
 * unlike Rput, no separate epoch-close is needed to observe the result
 * locally. On MPI < 3 builds the stub returns MPI_ERR_OTHER.
 *
 * Note: calling MPI_Cancel on an Rget request is implementation-defined
 * behavior and is discouraged by the MPI standard.
 *
 * @param origin          Local destination buffer (write destination).
 * @param origin_count    Number of elements in the origin buffer.
 * @param origin_dt_tag   DatatypeTag for the origin element type.
 * @param target_rank     Source rank.
 * @param target_disp     Displacement in units of the window's disp_unit.
 * @param target_count    Number of elements to read from the target.
 * @param target_dt_tag   DatatypeTag for the target element type.
 * @param win_handle      Window handle.
 * @param request_handle  Output: request table index for the new request.
 * @return MPI error code (MPI_SUCCESS = 0), or MPI_ERR_OTHER if the request
 *         table is exhausted.
 */
int ferrompi_rget(void* origin, int64_t origin_count, int32_t origin_dt_tag,
                  int32_t target_rank, int64_t target_disp, int64_t target_count,
                  int32_t target_dt_tag, int32_t win_handle, int64_t* request_handle);

/**
 * One-sided reduce: combine data from a local origin buffer with a remote
 * rank's window memory using a reduction operation.
 *
 * Wraps MPI_Accumulate. The operation completes when the surrounding epoch
 * closes (fence, complete, or unlock). On MPI < 3 builds the stub returns
 * MPI_ERR_OTHER.
 *
 * @param origin        Local origin buffer (read-only).
 * @param origin_count  Number of elements in the origin buffer.
 * @param origin_dt_tag DatatypeTag for the origin element type.
 * @param target_rank   Destination rank.
 * @param target_disp   Displacement in units of the window's disp_unit.
 * @param target_count  Number of elements to accumulate at the target.
 * @param target_dt_tag DatatypeTag for the target element type.
 * @param op_tag        ReduceOp discriminant (0=Sum, 12=Replace, 13=NoOp, ...).
 * @param win_handle    Window handle.
 * @return MPI error code (MPI_SUCCESS = 0).
 */
int ferrompi_accumulate(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                        int32_t target_rank, int64_t target_disp, int64_t target_count,
                        int32_t target_dt_tag, int32_t op_tag, int32_t win_handle);

/**
 * Request-returning one-sided reduce: combine data from a local origin buffer
 * with a remote rank's window memory using a reduction operation, returning a
 * Request handle that completes when the local origin buffer is safe to reuse.
 *
 * Wraps MPI_Raccumulate. Unlike MPI_Accumulate, local completion (i.e., the
 * origin buffer being safe to reuse) is signaled by the returned Request
 * rather than by the epoch boundary. Remote-side completion (visibility at the
 * target) still requires the surrounding epoch to close (fence, complete, or
 * unlock). On MPI < 3 builds the stub returns MPI_ERR_OTHER.
 *
 * @param origin         Local origin buffer (read-only).
 * @param origin_count   Number of elements in the origin buffer.
 * @param origin_dt_tag  DatatypeTag for the origin element type.
 * @param target_rank    Destination rank.
 * @param target_disp    Displacement in units of the window's disp_unit.
 * @param target_count   Number of elements to accumulate at the target.
 * @param target_dt_tag  DatatypeTag for the target element type.
 * @param op_tag         ReduceOp discriminant (0=Sum, 12=Replace, 13=NoOp, ...).
 * @param win_handle     Window handle.
 * @param request_handle Output: opaque handle for the returned Request.
 * @return MPI error code (MPI_SUCCESS = 0).
 */
int ferrompi_raccumulate(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                         int32_t target_rank, int64_t target_disp, int64_t target_count,
                         int32_t target_dt_tag, int32_t op_tag, int32_t win_handle,
                         int64_t* request_handle);

/**
 * Atomic read-modify-write: fetch the current target value into a local result
 * buffer, then apply a reduction of the origin buffer onto the target — all
 * atomically with respect to other RMA operations in the same epoch.
 *
 * Wraps MPI_Get_accumulate. The operation completes when the surrounding epoch
 * closes (fence, complete, or unlock). On MPI < 3 builds the stub returns
 * MPI_ERR_OTHER.
 *
 * @param origin         Local input buffer (read-only).
 * @param origin_count   Number of elements in the origin buffer.
 * @param origin_dt_tag  DatatypeTag for the origin element type.
 * @param result         Local destination buffer for the pre-update target value.
 * @param result_count   Number of elements in the result buffer.
 * @param result_dt_tag  DatatypeTag for the result element type.
 * @param target_rank    Target rank.
 * @param target_disp    Displacement in units of the window's disp_unit.
 * @param target_count   Number of elements to read-modify-write at the target.
 * @param target_dt_tag  DatatypeTag for the target element type.
 * @param op_tag         ReduceOp discriminant (0=Sum, 12=Replace, 13=NoOp, ...).
 * @param win_handle     Window handle.
 * @return MPI error code (MPI_SUCCESS = 0).
 */
int ferrompi_get_accumulate(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                            void* result, int64_t result_count, int32_t result_dt_tag,
                            int32_t target_rank, int64_t target_disp, int64_t target_count,
                            int32_t target_dt_tag, int32_t op_tag, int32_t win_handle);

/**
 * Single-element atomic fetch-and-update: read the current value at the
 * target displacement into `result`, then apply `op(origin, target)` at the
 * target — both atomically with respect to other RMA operations in the same
 * epoch.
 *
 * Wraps MPI_Fetch_and_op. Restricted to predefined MPI datatypes only
 * (no derived/custom datatypes). The operation completes when the surrounding
 * epoch closes (fence, complete, or unlock). On MPI < 3 builds the stub
 * returns MPI_ERR_OTHER.
 *
 * @param origin       Pointer to the single origin element (read-only).
 * @param result       Pointer to a single-element output buffer for the
 *                     pre-update target value.
 * @param dt_tag       DatatypeTag for the element type (predefined only).
 * @param target_rank  Target rank.
 * @param target_disp  Displacement in units of the window's disp_unit.
 * @param op_tag       ReduceOp discriminant (0=Sum, 12=Replace, 13=NoOp, ...).
 * @param win_handle   Window handle.
 * @return MPI error code (MPI_SUCCESS = 0).
 */
int ferrompi_fetch_and_op(const void* origin, void* result, int32_t dt_tag,
                          int32_t target_rank, int64_t target_disp,
                          int32_t op_tag, int32_t win_handle);

/**
 * Atomic compare-and-swap on a remote window element.
 *
 * Wraps `MPI_Compare_and_swap`. Atomically reads the remote element at
 * `(target_rank, target_disp)` into `result`, then conditionally replaces it
 * with `origin` if the remote value equals `compare`. The returned `result`
 * always holds the pre-CAS remote value regardless of whether the swap
 * succeeded.
 *
 * Only predefined integer and byte datatypes are valid for CAS per MPI 4.1
 * section 12.5.4 (`MPI_INT32_T`, `MPI_INT64_T`, `MPI_UINT32_T`,
 * `MPI_UINT64_T`, `MPI_UINT8_T` / `MPI_BYTE`). The Rust layer enforces this
 * at compile time via the `AtomicMpiDatatype` sealed trait.
 *
 * On MPI < 3 builds the function is present but returns `MPI_ERR_OTHER`.
 *
 * @param origin       Pointer to the new value to swap in if comparison succeeds.
 * @param compare      Pointer to the expected current value.
 * @param result       Pointer to a single-element output buffer for the
 *                     pre-CAS target value.
 * @param dt_tag       DatatypeTag for the element type (integer/byte types only).
 * @param target_rank  Target rank.
 * @param target_disp  Displacement in units of the window's disp_unit.
 * @param win_handle   Window handle.
 * @return MPI error code (MPI_SUCCESS = 0).
 */
int ferrompi_compare_and_swap(const void* origin, const void* compare, void* result,
                               int32_t dt_tag, int32_t target_rank, int64_t target_disp,
                               int32_t win_handle);

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
 * Group Operations
 * ============================================================ */

/** Sentinel value for slot 0: reserved for MPI_GROUP_EMPTY. */
#define FERROMPI_GROUP_EMPTY 0

/**
 * Return the MPI_UNDEFINED sentinel value for this MPI implementation.
 * Used by Group::rank() to expose the implementation's actual value.
 */
int32_t ferrompi_mpi_undefined(void);

/**
 * Get the group of a communicator (MPI_Comm_group).
 * @param comm_handle Communicator handle
 * @param group_handle Output: group handle
 * @return MPI error code
 */
int ferrompi_comm_group(int32_t comm_handle, int32_t* group_handle);

/**
 * Create a new group from a subset of an existing group (MPI_Group_incl).
 * @param group_handle Source group handle
 * @param n Number of ranks to include
 * @param ranks Array of n ranks from the source group
 * @param newgroup_handle Output: new group handle
 * @return MPI error code
 */
int ferrompi_group_incl(int32_t group_handle, int32_t n, const int32_t* ranks, int32_t* newgroup_handle);

/**
 * Create a new group excluding specified ranks (MPI_Group_excl).
 * @param group_handle Source group handle
 * @param n Number of ranks to exclude
 * @param ranks Array of n ranks to exclude from the source group
 * @param newgroup_handle Output: new group handle
 * @return MPI error code
 */
int ferrompi_group_excl(int32_t group_handle, int32_t n, const int32_t* ranks, int32_t* newgroup_handle);

/**
 * Free an MPI group (MPI_Group_free).
 * @param group_handle Group handle to free (no-op if slot 0 / MPI_GROUP_EMPTY)
 * @return MPI error code
 */
int ferrompi_group_free(int32_t group_handle);

/**
 * Get the size of a group (MPI_Group_size).
 * @param group_handle Group handle
 * @param size Output: number of processes in the group
 * @return MPI error code
 */
int ferrompi_group_size(int32_t group_handle, int32_t* size);

/**
 * Get the rank of the calling process in a group (MPI_Group_rank).
 * Returns MPI_UNDEFINED (-1) if the calling process is not in the group.
 * @param group_handle Group handle
 * @param rank Output: rank in group, or MPI_UNDEFINED
 * @return MPI error code
 */
int ferrompi_group_rank(int32_t group_handle, int32_t* rank);

/**
 * Compute the union of two groups (MPI_Group_union).
 * The result contains all ranks from group1 followed by ranks from group2 not in group1.
 * @param group1_handle First group handle
 * @param group2_handle Second group handle
 * @param newgroup_handle Output: new group handle
 * @return MPI error code
 */
int ferrompi_group_union(int32_t group1_handle, int32_t group2_handle, int32_t* newgroup_handle);

/**
 * Compute the intersection of two groups (MPI_Group_intersection).
 * The result contains ranks present in both groups, ordered as in group1.
 * @param group1_handle First group handle
 * @param group2_handle Second group handle
 * @param newgroup_handle Output: new group handle
 * @return MPI error code
 */
int ferrompi_group_intersection(int32_t group1_handle, int32_t group2_handle, int32_t* newgroup_handle);

/**
 * Compute the difference of two groups (MPI_Group_difference).
 * The result contains ranks in group1 that are not in group2, ordered as in group1.
 * @param group1_handle First group handle
 * @param group2_handle Second group handle
 * @param newgroup_handle Output: new group handle
 * @return MPI error code
 */
int ferrompi_group_difference(int32_t group1_handle, int32_t group2_handle, int32_t* newgroup_handle);

/**
 * Create a new group from rank-range triples (MPI_Group_range_incl).
 *
 * ranges_flat is a flattened array of length 3*n in
 * first, last, stride, first, last, stride, ... order.
 * The result group contains the union of all arithmetic progressions
 * described by the triples, in the order they appear.
 *
 * @param group_handle  Source group handle
 * @param n             Number of rank-range triples (must be >= 0)
 * @param ranges_flat   Flattened triples array of length 3*n
 * @param newgroup_handle Output: new group handle
 * @return MPI error code (MPI_ERR_ARG if n < 0 or group invalid;
 *         MPI_ERR_NO_MEM if n > 64 and malloc fails)
 */
int ferrompi_group_range_incl(int32_t group_handle, int32_t n,
                               const int32_t* ranges_flat,
                               int32_t* newgroup_handle);

/**
 * Create a new group by excluding rank-range triples (MPI_Group_range_excl).
 *
 * ranges_flat is a flattened array of length 3*n in
 * first, last, stride, first, last, stride, ... order.
 * The result group is the source group minus the union of the triples.
 *
 * @param group_handle  Source group handle
 * @param n             Number of rank-range triples (must be >= 0)
 * @param ranges_flat   Flattened triples array of length 3*n
 * @param newgroup_handle Output: new group handle
 * @return MPI error code (MPI_ERR_ARG if n < 0 or group invalid;
 *         MPI_ERR_NO_MEM if n > 64 and malloc fails)
 */
int ferrompi_group_range_excl(int32_t group_handle, int32_t n,
                               const int32_t* ranges_flat,
                               int32_t* newgroup_handle);

/**
 * Compare two groups (MPI_Group_compare).
 *
 * Normalises the MPI_IDENT / MPI_SIMILAR / MPI_UNEQUAL constants to
 * ferrompi-stable values (0 / 1 / 2) so the Rust #[repr(i32)] enum can
 * use fixed discriminants regardless of MPI implementation.
 *
 * @param group1_handle First group handle
 * @param group2_handle Second group handle
 * @param result        Output: 0=Identical, 1=Similar, 2=Unequal
 * @return MPI error code; MPI_ERR_INTERN if MPI returns an unexpected result
 */
int ferrompi_group_compare(int32_t group1_handle, int32_t group2_handle,
                           int32_t* result);

/**
 * Translate ranks from one group's rank space into another's
 * (MPI_Group_translate_ranks).
 *
 * Converts each entry in ranks1 (indices into group1) to the corresponding
 * rank in group2.  Ranks present in group1 but not in group2 are written as
 * -1 in ranks2 (normalised from MPI_UNDEFINED, whose integer value is not
 * standardised across implementations).
 *
 * @param group1_handle Source group handle
 * @param n             Number of ranks to translate (must be >= 0)
 * @param ranks1        Input array of n ranks in group1's rank space
 * @param group2_handle Target group handle
 * @param ranks2        Output array of n translated ranks (-1 for unmapped)
 * @return MPI error code; MPI_ERR_ARG if n < 0 or either group is invalid
 */
int ferrompi_group_translate_ranks(int32_t group1_handle, int32_t n,
                                   const int32_t* ranks1,
                                   int32_t group2_handle,
                                   int32_t* ranks2);

/* ============================================================
 * Custom Datatype Operations
 * ============================================================ */

/**
 * Create a contiguous derived datatype and commit it.
 *
 * Wraps MPI_Type_contiguous + MPI_Type_commit. The returned handle is
 * stored in the internal datatype_table and is always committed on return.
 *
 * @param count          Number of elements in the contiguous block
 * @param basetype_tag   Predefined datatype tag (FERROMPI_F32 … FERROMPI_BYTE)
 * @param newtype_handle Output: handle for the new committed datatype
 * @return MPI error code; MPI_ERR_OTHER if the datatype_table is full
 */
int ferrompi_type_contiguous(int32_t count, int32_t basetype_tag,
                              int32_t* newtype_handle);

/**
 * Create a strided vector derived datatype and commit it.
 *
 * Wraps MPI_Type_vector + MPI_Type_commit. The returned handle is stored in
 * the internal datatype_table and is always committed on return.
 *
 * @param count          Number of blocks
 * @param blocklength    Number of base elements per block
 * @param stride         Number of base elements between the start of
 *                       consecutive blocks (may be negative)
 * @param basetype_tag   Predefined datatype tag (FERROMPI_F32 … FERROMPI_BYTE)
 * @param newtype_handle Output: handle for the new committed datatype
 * @return MPI error code; MPI_ERR_OTHER if the datatype_table is full
 */
int ferrompi_type_vector(int32_t count, int32_t blocklength, int32_t stride,
                         int32_t basetype_tag, int32_t* newtype_handle);

/**
 * Create a heterogeneous struct derived datatype and commit it.
 *
 * Wraps MPI_Type_create_struct + MPI_Type_commit. Each field is described by
 * a (blocklength, displacement, basetype_tag) triple. The returned handle is
 * stored in the internal datatype_table and is always committed on return.
 *
 * Uses a 32-slot stack buffer for parallel MPI_Aint/MPI_Datatype arrays; falls
 * back to heap allocation for count > 32.
 *
 * @param count          Number of fields (must be >= 0; MPI requires >= 1 for success)
 * @param blocklengths   Array of count block lengths
 * @param displacements  Array of count byte displacements (cast to MPI_Aint)
 * @param basetype_tags  Array of count predefined datatype tags
 * @param newtype_handle Output: handle for the new committed datatype
 * @return MPI error code; MPI_ERR_ARG if count < 0; MPI_ERR_TYPE if any
 *         basetype tag is unresolvable; MPI_ERR_NO_MEM on heap allocation
 *         failure; MPI_ERR_OTHER if the datatype_table is full
 */
int ferrompi_type_create_struct(int32_t count,
                                const int32_t* blocklengths,
                                const int64_t* displacements,
                                const int32_t* basetype_tags,
                                int32_t* newtype_handle);

/**
 * Create a resized datatype with the same payload as an existing committed
 * datatype but with a new lower bound and extent (MPI_Type_create_resized).
 *
 * Common use case: fix an extent mismatch when an array of #[repr(C)] structs
 * has natural padding-to-alignment that MPI's auto-computed extent does not
 * match.  The original datatype (old_handle) remains valid and committed; this
 * function produces a wholly new handle in the datatype_table.
 *
 * @param old_handle     Handle of the existing committed datatype
 * @param lb             New lower bound in bytes (typically 0)
 * @param extent         New total extent in bytes between consecutive elements
 * @param newtype_handle Output: handle for the new committed datatype
 * @return MPI error code; MPI_ERR_TYPE if old_handle is invalid;
 *         MPI_ERR_ARG if extent is negative (implementation-defined);
 *         MPI_ERR_OTHER if the datatype_table is full
 */
int ferrompi_type_create_resized(int32_t old_handle,
                                 int64_t lb,
                                 int64_t extent,
                                 int32_t* newtype_handle);

/**
 * Free a committed custom datatype.
 *
 * Calls MPI_Type_free and releases the handle slot. No-op for already-freed
 * slots; returns MPI_ERR_ARG for out-of-range handles.
 *
 * @param type_handle Handle returned by ferrompi_type_contiguous (or future builders)
 * @return MPI error code
 */
int ferrompi_type_free(int32_t type_handle);

/* ============================================================
 * Custom-Datatype Point-to-Point
 * ============================================================ */

/**
 * Blocking send using a committed custom datatype.
 *
 * Identical to ferrompi_send but looks up the MPI_Datatype via
 * get_datatype_committed(datatype_handle) instead of get_datatype(tag).
 *
 * @param buf              Data buffer
 * @param count            Number of datatype elements
 * @param datatype_handle  Custom datatype handle (from ferrompi_type_*builders*)
 * @param dest             Destination rank
 * @param tag              Message tag
 * @param comm             Communicator handle
 * @return MPI error code
 */
int ferrompi_send_custom(
    const void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t dest,
    int32_t tag,
    int32_t comm
);

/**
 * Blocking receive using a committed custom datatype.
 *
 * Identical to ferrompi_recv but looks up the MPI_Datatype via
 * get_datatype_committed(datatype_handle) instead of get_datatype(tag).
 *
 * @param buf              Receive buffer
 * @param count            Maximum number of datatype elements
 * @param datatype_handle  Custom datatype handle
 * @param source           Source rank (or -1 for MPI_ANY_SOURCE)
 * @param tag              Message tag (or -1 for MPI_ANY_TAG)
 * @param comm             Communicator handle
 * @param actual_source    Output: actual source rank
 * @param actual_tag       Output: actual tag
 * @param actual_count     Output: actual count received
 * @return MPI error code
 */
int ferrompi_recv_custom(
    void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t source,
    int32_t tag,
    int32_t comm,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
);

/**
 * Nonblocking send using a committed custom datatype.
 *
 * Identical to ferrompi_isend but looks up the MPI_Datatype via
 * get_datatype_committed(datatype_handle) instead of get_datatype(tag).
 *
 * @param buf              Data buffer (must remain valid until request completes)
 * @param count            Number of datatype elements
 * @param datatype_handle  Custom datatype handle
 * @param dest             Destination rank
 * @param tag              Message tag
 * @param comm             Communicator handle
 * @param request          Output: request handle
 * @return MPI error code
 */
int ferrompi_isend_custom(
    const void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t dest,
    int32_t tag,
    int32_t comm,
    int64_t* request
);

/**
 * Nonblocking receive using a committed custom datatype.
 *
 * Identical to ferrompi_irecv but looks up the MPI_Datatype via
 * get_datatype_committed(datatype_handle) instead of get_datatype(tag).
 *
 * @param buf              Receive buffer (must remain valid until request completes)
 * @param count            Maximum number of datatype elements
 * @param datatype_handle  Custom datatype handle
 * @param source           Source rank (or -1 for MPI_ANY_SOURCE)
 * @param tag              Message tag (or -1 for MPI_ANY_TAG)
 * @param comm             Communicator handle
 * @param request          Output: request handle
 * @return MPI error code
 */
int ferrompi_irecv_custom(
    void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t source,
    int32_t tag,
    int32_t comm,
    int64_t* request
);

/* ============================================================
 * User-Defined Reduction Op (MPI_Op_create)
 * ============================================================
 *
 * The op-slot table supports MAX_OPS (16) concurrently live UserOp
 * instances.  Each slot is backed by a unique C trampoline function
 * so that MPI_Op_create can associate a single closure to an op
 * without needing a user-data side channel.
 */

/**
 * Allocate a free slot in the op-slot table.
 *
 * On success, writes the slot index (0..MAX_OPS-1) to *out_slot and
 * returns MPI_SUCCESS.  Returns MPI_ERR_OTHER if the table is full.
 */
int ferrompi_op_alloc_slot(int32_t* out_slot);

/**
 * Store the Rust fat-pointer (data + vtable) for the boxed closure
 * into the given slot.  Must be called before ferrompi_op_create_user.
 *
 * Both pointers are the two halves of a Rust *mut dyn Fn(...) fat pointer,
 * stored as void* to keep the C layer independent of Rust internals.
 */
void ferrompi_op_set_closure(int32_t slot, void* data, void* vtbl);

/**
 * Create an MPI_Op for the given slot.
 *
 * @param slot     Slot index returned by ferrompi_op_alloc_slot.
 * @param commute  1 if the operation is commutative, 0 otherwise.
 * @param out_handle Output: the handle (same as slot) to pass back to Rust.
 * @return MPI error code.
 */
int ferrompi_op_create_user(int32_t slot, int32_t commute, int32_t* out_handle);

/**
 * Free the MPI_Op and release the slot.
 *
 * Drop ordering (ADR-0005 Decision 3):
 *   1. MPI_Op_free  — MPI will not invoke the trampoline after this.
 *   2. ferrompi_op_drop_closure (Rust callback) — drops the boxed closure.
 *   3. free_op_slot — releases the C-side slot.
 *
 * @param handle  Slot/handle returned by ferrompi_op_create_user.
 * @return MPI error code.
 */
int ferrompi_op_free(int32_t handle);

/**
 * Release the op slot WITHOUT calling MPI_Op_free.
 *
 * Use this in rollback paths where ferrompi_op_create_user failed and the
 * slot holds MPI_OP_NULL.  Calling MPI_Op_free on MPI_OP_NULL is
 * implementation-defined; this shim avoids it entirely.
 *
 * The caller must have already dropped the Rust closure before calling this.
 * ferrompi_op_drop_closure is NOT invoked by this function.
 *
 * @param handle  Slot index allocated by ferrompi_op_alloc_slot.
 * @return MPI_SUCCESS, or MPI_ERR_ARG if handle is out of range.
 */
int ferrompi_op_free_slot_only(int32_t handle);

/**
 * MPI_Allreduce using a user-defined reduction op.
 *
 * @param sendbuf       Send buffer.
 * @param recvbuf       Receive buffer.
 * @param count         Number of elements.
 * @param datatype_tag  FERROMPI_* tag for the element type.
 * @param op_handle     Slot handle returned by ferrompi_op_create_user.
 * @param comm          Communicator handle.
 * @return MPI error code.
 */
int ferrompi_allreduce_user_op(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op_handle,
    int32_t comm
);

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
