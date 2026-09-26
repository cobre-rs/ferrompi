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
 * ============================================================ */

/* Must match the Rust DatatypeTag enum discriminants. */
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
 * ============================================================ */

/* Must match the Rust SplitType enum discriminants. */
#define FERROMPI_COMM_TYPE_SHARED 0

/* ============================================================
 * Initialization and Finalization
 * ============================================================ */

int ferrompi_init_thread(int required, int* provided);

int ferrompi_finalize(void);

int ferrompi_initialized(int* flag);

int ferrompi_finalized(int* flag);

/* ============================================================
 * Communicator Operations
 * ============================================================ */

int32_t ferrompi_comm_world(void);

int ferrompi_comm_rank(int32_t comm, int32_t* rank);

int ferrompi_comm_size(int32_t comm, int32_t* size);

int ferrompi_comm_dup(int32_t comm, int32_t* newcomm);

int ferrompi_comm_free(int32_t comm);

/** Split a communicator (MPI_Comm_split). color=-1 opts out (MPI_UNDEFINED); newcomm is set to -1. */
int ferrompi_comm_split(int32_t comm, int32_t color, int32_t key, int32_t* newcomm);

/** Split a communicator by type (MPI_Comm_split_type). newcomm is set to -1 if the split yields MPI_COMM_NULL. */
int ferrompi_comm_split_type(int32_t comm, int32_t split_type, int32_t key, int32_t* newcomm);

/**
 * Create a sub-communicator from a parent communicator and a group (MPI_Comm_create).
 * Collective over comm_handle: every rank in comm_handle must call, even ranks
 * not in group_handle. Ranks not in group_handle receive newcomm_handle = -1.
 */
int ferrompi_comm_create_from_group_parent(int32_t comm_handle,
                                           int32_t group_handle,
                                           int32_t* newcomm_handle);

/**
 * Create a communicator from a group without a parent communicator
 * (MPI_Comm_create_from_group, MPI 4.0+). Collective only over the processes
 * sharing the same group and stringtag, not over an existing communicator.
 * Returns MPI_ERR_OTHER on MPI < 4.0.
 */
int ferrompi_comm_create_from_group(int32_t group_handle,
                                    const char* stringtag,
                                    int32_t* newcomm_handle);

/* ============================================================
 * Synchronization
 * ============================================================ */

int ferrompi_barrier(int32_t comm);

/* ============================================================
 * Generic Point-to-Point Communication
 * ============================================================ */

int ferrompi_send(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm
);

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

int ferrompi_isend(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm,
    int64_t* request
);

int ferrompi_irecv(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm,
    int64_t* request
);

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

int ferrompi_probe(
    int32_t source,
    int32_t tag,
    int32_t comm,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* count,
    int32_t datatype_tag
);

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

int ferrompi_bcast(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm);

/**
 * In-place collectives (generic): reduce, allreduce, gather, allgather,
 * alltoall, igather, iallgather, ialltoall, allreduce_init, gather_init,
 * allgather_init and alltoall_init substitute MPI_IN_PLACE for sendbuf when
 * sendbuf == NULL; scatter, iscatter and scatter_init substitute MPI_IN_PLACE
 * for recvbuf when recvbuf == NULL. ireduce, iallreduce and reduce_init pass
 * sendbuf through unchanged.
 * Rust slices never produce a NULL pointer, so NULL is an unambiguous in-place
 * marker here.
 */

/** Reduce (generic). NULL sendbuf maps to MPI_IN_PLACE. */
int ferrompi_reduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t root, int32_t comm);

/** All-reduce (generic). NULL sendbuf maps to MPI_IN_PLACE. */
int ferrompi_allreduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm);

int ferrompi_scan(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm);

int ferrompi_exscan(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm);

/** Gather (generic). NULL sendbuf maps to MPI_IN_PLACE (valid only at root). */
int ferrompi_gather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm);

/** All-gather (generic). NULL sendbuf maps to MPI_IN_PLACE. */
int ferrompi_allgather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm);

/** Scatter (generic). NULL recvbuf maps to MPI_IN_PLACE (valid only at root). */
int ferrompi_scatter(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm);

/** All-to-all (generic, MPI_Alltoall). NULL sendbuf maps to MPI_IN_PLACE. */
int ferrompi_alltoall(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm);

int ferrompi_reduce_scatter_block(const void* sendbuf, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t op, int32_t comm);

/* ============================================================
 * Generic V-Collectives (variable-count)
 * ============================================================ */

int ferrompi_gatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t root, int32_t comm
);

int ferrompi_scatterv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* displs,
    void* recvbuf, int64_t recvcount,
    int32_t datatype_tag, int32_t root, int32_t comm
);

int ferrompi_allgatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t comm
);

int ferrompi_alltoallv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls,
    void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls,
    int32_t datatype_tag, int32_t comm
);

/* ============================================================
 * Generic Collective Operations - Nonblocking
 * ============================================================ */

int ferrompi_ibcast(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

int ferrompi_iallreduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

int ferrompi_ireduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t root, int32_t comm, int64_t* request);

/** Nonblocking gather (generic). NULL sendbuf maps to MPI_IN_PLACE (valid only at root). */
int ferrompi_igather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Nonblocking all-gather (generic). NULL sendbuf maps to MPI_IN_PLACE. */
int ferrompi_iallgather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm, int64_t* request);

/** Nonblocking scatter (generic). NULL recvbuf maps to MPI_IN_PLACE (valid only at root). */
int ferrompi_iscatter(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

int ferrompi_ibarrier(int32_t comm, int64_t* request);

int ferrompi_iscan(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

int ferrompi_iexscan(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Nonblocking all-to-all (generic). NULL sendbuf maps to MPI_IN_PLACE. */
int ferrompi_ialltoall(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm, int64_t* request);

int ferrompi_igatherv(const void* sendbuf, int64_t sendcount, void* recvbuf, const int32_t* recvcounts, const int32_t* displs, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

int ferrompi_iscatterv(const void* sendbuf, const int32_t* sendcounts, const int32_t* displs, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

int ferrompi_iallgatherv(const void* sendbuf, int64_t sendcount, void* recvbuf, const int32_t* recvcounts, const int32_t* displs, int32_t datatype_tag, int32_t comm, int64_t* request);

int ferrompi_ialltoallv(const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls, void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls, int32_t datatype_tag, int32_t comm, int64_t* request);

int ferrompi_ireduce_scatter_block(const void* sendbuf, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/* ============================================================
 * Persistent Point-to-Point (MPI 1.1+)
 * ============================================================ */

int ferrompi_send_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
);

int ferrompi_recv_init(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
);

int ferrompi_rsend_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
);

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
 * Attach a user buffer for buffered sends (MPI_Buffer_attach). Only one
 * buffer may be attached at a time; it must remain valid until detach.
 */
int ferrompi_buffer_attach(void* buffer, int64_t size);

/** Detach the previously attached buffer (MPI_Buffer_detach). Blocks until all buffered sends using it complete. */
int ferrompi_buffer_detach(void** buffer, int64_t* size);

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

int ferrompi_bcast_init(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent all-reduce (generic). NULL sendbuf maps to MPI_IN_PLACE. */
int ferrompi_allreduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent gather (generic). NULL sendbuf maps to MPI_IN_PLACE (valid only at root). */
int ferrompi_gather_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

int ferrompi_reduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent scatter (generic). NULL recvbuf maps to MPI_IN_PLACE (valid only at root). */
int ferrompi_scatter_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent all-gather (generic). NULL sendbuf maps to MPI_IN_PLACE. */
int ferrompi_allgather_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm, int64_t* request);

int ferrompi_scan_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

int ferrompi_exscan_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent all-to-all (generic). NULL sendbuf maps to MPI_IN_PLACE. */
int ferrompi_alltoall_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm, int64_t* request);

int ferrompi_gatherv_init(const void* sendbuf, int64_t sendcount, void* recvbuf, const int32_t* recvcounts, const int32_t* displs, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

int ferrompi_scatterv_init(const void* sendbuf, const int32_t* sendcounts, const int32_t* displs, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

int ferrompi_allgatherv_init(const void* sendbuf, int64_t sendcount, void* recvbuf, const int32_t* recvcounts, const int32_t* displs, int32_t datatype_tag, int32_t comm, int64_t* request);

int ferrompi_alltoallv_init(const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls, void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls, int32_t datatype_tag, int32_t comm, int64_t* request);

int ferrompi_reduce_scatter_block_init(const void* sendbuf, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/* ============================================================
 * Info Object Operations
 * ============================================================ */

int ferrompi_info_create(int32_t* info_handle);

/** Free an MPI_Info object (MPI_Info_free). No-op on an invalid or already-freed handle. */
int ferrompi_info_free(int32_t info_handle);

int ferrompi_info_set(int32_t info_handle, const char* key, const char* value);

int ferrompi_info_get(int32_t info_handle, const char* key, char* value, int32_t* valuelen, int32_t* flag);

/* ============================================================
 * Error Information
 * ============================================================ */

int ferrompi_error_info(int code, int32_t* error_class, char* message, int32_t* msg_len);

/* ============================================================
 * Request Management
 * ============================================================ */

int ferrompi_wait(int64_t request);

int ferrompi_test(int64_t request, int32_t* flag);

int ferrompi_waitall(int64_t count, int64_t* requests);

int ferrompi_request_free(int64_t request);

/** Non-destructive status query (MPI_Request_get_status). Does not free the request handle. */
int ferrompi_request_get_status(int64_t request, int32_t* flag);

/** Request cancellation of a pending operation (MPI_Cancel). Does not free the request handle. */
int ferrompi_cancel(int64_t request);

int ferrompi_waitany(int64_t count, int64_t* requests, int32_t* index);

int ferrompi_waitsome(int64_t count, int64_t* requests, int64_t* outcount, int32_t* indices);

int ferrompi_testany(int64_t count, int64_t* requests, int32_t* index, int32_t* flag);

int ferrompi_testsome(int64_t count, int64_t* requests, int64_t* outcount, int32_t* indices);

int ferrompi_start(int64_t request);

int ferrompi_startall(int64_t count, int64_t* requests);

/* ============================================================
 * RMA Window Operations (MPI 3.0+)
 * ============================================================ */

/* Lock type constants for MPI_Win_lock / MPI_Win_lock_all. Must match the LockType -> FERROMPI_LOCK_* mapping in src/window.rs. */
#define FERROMPI_LOCK_EXCLUSIVE 0
#define FERROMPI_LOCK_SHARED    1

int ferrompi_win_allocate_shared(int64_t size, int32_t disp_unit, int32_t info,
                                  int32_t comm, void** baseptr, int32_t* win);

int ferrompi_win_create(void* base, int64_t size, int32_t disp_unit, int32_t info,
                         int32_t comm, int32_t* win);

int ferrompi_win_allocate(int64_t size, int32_t disp_unit, int32_t info,
                           int32_t comm, void** baseptr, int32_t* win);

int ferrompi_win_shared_query(int32_t win, int32_t rank,
                               int64_t* size, int32_t* disp_unit, void** baseptr);

/** Free an MPI window (MPI_Win_free). No-op on an invalid or already-freed handle. */
int ferrompi_win_free(int32_t win);

int ferrompi_win_fence(int32_t assert_val, int32_t win);

void ferrompi_win_fence_mode_values(int32_t* out);

int ferrompi_win_lock(int32_t lock_type, int32_t rank, int32_t assert_val, int32_t win);

int ferrompi_win_unlock(int32_t rank, int32_t win);

int ferrompi_win_lock_all(int32_t assert_val, int32_t win);

int ferrompi_win_unlock_all(int32_t win);

int ferrompi_win_flush(int32_t rank, int32_t win);

int ferrompi_win_flush_all(int32_t win);

int ferrompi_win_flush_local(int32_t rank, int32_t win);

int ferrompi_win_flush_local_all(int32_t win);

int ferrompi_win_sync(int32_t win);

int ferrompi_win_post(int32_t group, int32_t assert_val, int32_t win);

int ferrompi_win_start(int32_t group, int32_t assert_val, int32_t win);

int ferrompi_win_complete(int32_t win);

int ferrompi_win_wait(int32_t win);

int ferrompi_win_test(int32_t win, int32_t* flag);

void ferrompi_win_pscw_mode_values(int32_t* out);

int ferrompi_put(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                 int32_t target_rank, int64_t target_disp, int64_t target_count,
                 int32_t target_dt_tag, int32_t win_handle);

int ferrompi_rput(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                  int32_t target_rank, int64_t target_disp, int64_t target_count,
                  int32_t target_dt_tag, int32_t win_handle, int64_t* request_handle);

int ferrompi_get(void* origin, int64_t origin_count, int32_t origin_dt_tag,
                 int32_t target_rank, int64_t target_disp, int64_t target_count,
                 int32_t target_dt_tag, int32_t win_handle);

int ferrompi_rget(void* origin, int64_t origin_count, int32_t origin_dt_tag,
                  int32_t target_rank, int64_t target_disp, int64_t target_count,
                  int32_t target_dt_tag, int32_t win_handle, int64_t* request_handle);

int ferrompi_accumulate(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                        int32_t target_rank, int64_t target_disp, int64_t target_count,
                        int32_t target_dt_tag, int32_t op_tag, int32_t win_handle);

int ferrompi_raccumulate(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                         int32_t target_rank, int64_t target_disp, int64_t target_count,
                         int32_t target_dt_tag, int32_t op_tag, int32_t win_handle,
                         int64_t* request_handle);

int ferrompi_get_accumulate(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                            void* result, int64_t result_count, int32_t result_dt_tag,
                            int32_t target_rank, int64_t target_disp, int64_t target_count,
                            int32_t target_dt_tag, int32_t op_tag, int32_t win_handle);

int ferrompi_fetch_and_op(const void* origin, void* result, int32_t dt_tag,
                          int32_t target_rank, int64_t target_disp,
                          int32_t op_tag, int32_t win_handle);

int ferrompi_compare_and_swap(const void* origin, const void* compare, void* result,
                               int32_t dt_tag, int32_t target_rank, int64_t target_disp,
                               int32_t win_handle);

/* ============================================================
 * Utility Functions
 * ============================================================ */

int ferrompi_get_library_version(char* buf, int32_t* len);

int ferrompi_get_version(char* version, int32_t* len);

int ferrompi_get_processor_name(char* name, int32_t* len);

double ferrompi_wtime(void);

int ferrompi_abort(int32_t comm, int32_t errorcode);

/* ============================================================
 * Group Operations
 * ============================================================ */

#define FERROMPI_GROUP_EMPTY 0

int ferrompi_comm_group(int32_t comm_handle, int32_t* group_handle);

int ferrompi_group_incl(int32_t group_handle, int32_t n, const int32_t* ranks, int32_t* newgroup_handle);

int ferrompi_group_excl(int32_t group_handle, int32_t n, const int32_t* ranks, int32_t* newgroup_handle);

int ferrompi_group_free(int32_t group_handle);

int ferrompi_group_size(int32_t group_handle, int32_t* size);

/** Get the calling process's rank in a group (MPI_Group_rank). Returns MPI_UNDEFINED (-1) if not a member. */
int ferrompi_group_rank(int32_t group_handle, int32_t* rank);

int ferrompi_group_union(int32_t group1_handle, int32_t group2_handle, int32_t* newgroup_handle);

int ferrompi_group_intersection(int32_t group1_handle, int32_t group2_handle, int32_t* newgroup_handle);

int ferrompi_group_difference(int32_t group1_handle, int32_t group2_handle, int32_t* newgroup_handle);

int ferrompi_group_range_incl(int32_t group_handle, int32_t n,
                               const int32_t* ranges_flat,
                               int32_t* newgroup_handle);

int ferrompi_group_range_excl(int32_t group_handle, int32_t n,
                               const int32_t* ranges_flat,
                               int32_t* newgroup_handle);

int ferrompi_group_compare(int32_t group1_handle, int32_t group2_handle,
                           int32_t* result);

/**
 * Translate ranks between groups (MPI_Group_translate_ranks). Ranks present in
 * group1 but not group2 are written as -1 (normalised from MPI_UNDEFINED).
 */
int ferrompi_group_translate_ranks(int32_t group1_handle, int32_t n,
                                   const int32_t* ranks1,
                                   int32_t group2_handle,
                                   int32_t* ranks2);

/* ============================================================
 * Custom Datatype Operations
 * ============================================================ */

/**
 * ferrompi_type_contiguous, ferrompi_type_vector and ferrompi_type_create_struct
 * store the returned handle in the internal datatype table, committed on return.
 */
int ferrompi_type_contiguous(int32_t count, int32_t basetype_tag,
                              int32_t* newtype_handle);

int ferrompi_type_vector(int32_t count, int32_t blocklength, int32_t stride,
                         int32_t basetype_tag, int32_t* newtype_handle);

int ferrompi_type_create_struct(int32_t count,
                                const int32_t* blocklengths,
                                const int64_t* displacements,
                                const int32_t* basetype_tags,
                                int32_t* newtype_handle);

int ferrompi_type_create_resized(int32_t old_handle,
                                 int64_t lb,
                                 int64_t extent,
                                 int32_t* newtype_handle);

/**
 * Query the extent and true extent of a committed custom datatype
 * (MPI_Type_get_extent + MPI_Type_get_true_extent). Writes the three
 * outputs only on success.
 */
int ferrompi_type_get_extents(int32_t type_handle, int64_t* extent,
                              int64_t* true_lb, int64_t* true_extent);

int ferrompi_type_free(int32_t type_handle);

/* ============================================================
 * Custom-Datatype Point-to-Point
 * ============================================================ */

int ferrompi_send_custom(
    const void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t dest,
    int32_t tag,
    int32_t comm
);

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

int ferrompi_isend_custom(
    const void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t dest,
    int32_t tag,
    int32_t comm,
    int64_t* request
);

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
 * ============================================================ */

/** Call ferrompi_op_alloc_slot and publish the Rust closure for that slot before ferrompi_op_create_user. */
int ferrompi_op_alloc_slot(int32_t* out_slot);

int ferrompi_op_create_user(int32_t slot, int32_t commute, int32_t* out_handle);

int ferrompi_op_free(int32_t handle);

/** Release the op slot without calling MPI_Op_free. The caller must already have dropped the Rust closure. */
int ferrompi_op_free_slot_only(int32_t handle);

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

int32_t ferrompi_err_file(void);
int32_t ferrompi_err_info(void);
int32_t ferrompi_err_win(void);

#ifdef __cplusplus
}
#endif

#endif /* ferrompi_H */
