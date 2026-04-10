/**
 * ferrompi.c - Thin C wrapper for MPI 4.x features
 * 
 * Implementation of the FFI layer for Rust interop.
 */

#include "ferrompi.h"
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>

/* ============================================================
 * Internal State and Handle Management
 * ============================================================ */

// Maximum number of concurrent communicators (excluding COMM_WORLD)
#define MAX_COMMS 256

// Maximum number of concurrent requests
#define MAX_REQUESTS 16384

// Split type constants (must match Rust SplitType enum and header defines)
#define FERROMPI_COMM_TYPE_SHARED 0

// Maximum number of concurrent MPI_Info objects
#define MAX_INFOS 64

// Maximum number of concurrent RMA windows
#define MAX_WINDOWS 256

// Communicator table (index 0 is always COMM_WORLD)
static MPI_Comm comm_table[MAX_COMMS];
static int comm_count = 1;  // Start at 1, 0 is COMM_WORLD

// Request table
static MPI_Request request_table[MAX_REQUESTS];
static int request_used[MAX_REQUESTS];  // 1 if slot is in use
static int next_request_hint = 0;

// Window table
static MPI_Win win_table[MAX_WINDOWS];
static int win_used[MAX_WINDOWS];  // 1 if slot is in use
static int next_win_hint = 0;

// Info table
static MPI_Info info_table[MAX_INFOS];
static int info_used[MAX_INFOS];  // 1 if slot is in use
static int next_info_hint = 0;

// Initialize tables
static int tables_initialized = 0;

static void init_tables(void) {
    if (tables_initialized) return;
    
    // Note: MPI_COMM_WORLD must be set AFTER MPI_Init is called
    // comm_table[0] will be set in ferrompi_init* functions
    for (int i = 0; i < MAX_COMMS; i++) {
        comm_table[i] = MPI_COMM_NULL;
    }
    for (int i = 0; i < MAX_REQUESTS; i++) {
        request_table[i] = MPI_REQUEST_NULL;
        request_used[i] = 0;
    }
    for (int i = 0; i < MAX_WINDOWS; i++) {
        win_table[i] = MPI_WIN_NULL;
        win_used[i] = 0;
    }
    for (int i = 0; i < MAX_INFOS; i++) {
        info_table[i] = MPI_INFO_NULL;
        info_used[i] = 0;
    }
    tables_initialized = 1;
}

// Get MPI_Comm from handle
static MPI_Comm get_comm(int32_t handle) {
    if (handle < 0 || handle >= MAX_COMMS) {
        return MPI_COMM_NULL;
    }
    return comm_table[handle];
}

// Allocate a communicator handle
static int32_t alloc_comm(MPI_Comm comm) {
    for (int i = 1; i < MAX_COMMS; i++) {
        if (comm_table[i] == MPI_COMM_NULL) {
            comm_table[i] = comm;
            if (i >= comm_count) comm_count = i + 1;
            return i;
        }
    }
    return -1;  // No space
}

// Allocate a request handle
static int64_t alloc_request(MPI_Request req) {
    // Start searching from hint for O(1) common case
    for (int i = 0; i < MAX_REQUESTS; i++) {
        int idx = (next_request_hint + i) % MAX_REQUESTS;
        if (!request_used[idx]) {
            request_table[idx] = req;
            request_used[idx] = 1;
            next_request_hint = (idx + 1) % MAX_REQUESTS;
            return (int64_t)idx;
        }
    }
    return -1;  // No space
}

// Get MPI_Request pointer from handle
static MPI_Request* get_request_ptr(int64_t handle) {
    if (handle < 0 || handle >= MAX_REQUESTS || !request_used[handle]) {
        return NULL;
    }
    return &request_table[handle];
}

// Free a request handle
static void free_request(int64_t handle) {
    if (handle >= 0 && handle < MAX_REQUESTS) {
        request_table[handle] = MPI_REQUEST_NULL;
        request_used[handle] = 0;
    }
}

// Allocate a window handle
static int32_t alloc_win(MPI_Win win) {
    for (int i = 0; i < MAX_WINDOWS; i++) {
        int idx = (next_win_hint + i) % MAX_WINDOWS;
        if (!win_used[idx]) {
            win_table[idx] = win;
            win_used[idx] = 1;
            next_win_hint = (idx + 1) % MAX_WINDOWS;
            return (int32_t)idx;
        }
    }
    return -1;  // No space
}

// Get MPI_Win from handle
static MPI_Win get_win(int32_t handle) {
    if (handle < 0 || handle >= MAX_WINDOWS || !win_used[handle]) {
        return MPI_WIN_NULL;
    }
    return win_table[handle];
}

// Get MPI_Win pointer from handle (for operations that modify the win)
static MPI_Win* get_win_ptr(int32_t handle) {
    if (handle < 0 || handle >= MAX_WINDOWS || !win_used[handle]) {
        return NULL;
    }
    return &win_table[handle];
}

// Free a window handle
static void free_win(int32_t handle) {
    if (handle >= 0 && handle < MAX_WINDOWS) {
        win_table[handle] = MPI_WIN_NULL;
        win_used[handle] = 0;
    }
}

// Allocate an info handle
static int32_t alloc_info(MPI_Info info) {
    for (int i = 0; i < MAX_INFOS; i++) {
        int idx = (next_info_hint + i) % MAX_INFOS;
        if (!info_used[idx]) {
            info_table[idx] = info;
            info_used[idx] = 1;
            next_info_hint = (idx + 1) % MAX_INFOS;
            return (int32_t)idx;
        }
    }
    return -1;  // No space
}

// Get MPI_Info from handle
static MPI_Info get_info(int32_t handle) {
    if (handle < 0 || handle >= MAX_INFOS || !info_used[handle]) {
        return MPI_INFO_NULL;
    }
    return info_table[handle];
}

// Free an info handle
static void free_info(int32_t handle) {
    if (handle >= 0 && handle < MAX_INFOS) {
        info_table[handle] = MPI_INFO_NULL;
        info_used[handle] = 0;
    }
}

// Map operation code to MPI_Op
static MPI_Op get_op(int32_t op) {
    switch (op) {
        case 0: return MPI_SUM;
        case 1: return MPI_MAX;
        case 2: return MPI_MIN;
        case 3: return MPI_PROD;
        default: return MPI_OP_NULL;
    }
}

// Map datatype tag to MPI_Datatype
static MPI_Datatype get_datatype(int32_t tag) {
    switch (tag) {
        case FERROMPI_F32: return MPI_FLOAT;
        case FERROMPI_F64: return MPI_DOUBLE;
        case FERROMPI_I32: return MPI_INT32_T;
        case FERROMPI_I64: return MPI_INT64_T;
        case FERROMPI_U8:  return MPI_UINT8_T;
        case FERROMPI_U32: return MPI_UINT32_T;
        case FERROMPI_U64: return MPI_UINT64_T;
        default: return MPI_DATATYPE_NULL;
    }
}

/* ============================================================
 * Initialization and Finalization
 * ============================================================ */

int ferrompi_init_thread(int required, int* provided) {
    init_tables();
    int mpi_required;
    switch (required) {
        case 0: mpi_required = MPI_THREAD_SINGLE; break;
        case 1: mpi_required = MPI_THREAD_FUNNELED; break;
        case 2: mpi_required = MPI_THREAD_SERIALIZED; break;
        case 3: mpi_required = MPI_THREAD_MULTIPLE; break;
        default: mpi_required = MPI_THREAD_SINGLE; break;
    }
    
    int mpi_provided;
    int ret = MPI_Init_thread(NULL, NULL, mpi_required, &mpi_provided);
    
    // Initialize COMM_WORLD after MPI_Init
    if (ret == MPI_SUCCESS) {
        comm_table[0] = MPI_COMM_WORLD;
    }
    
    if (provided) {
        switch (mpi_provided) {
            case MPI_THREAD_SINGLE: *provided = 0; break;
            case MPI_THREAD_FUNNELED: *provided = 1; break;
            case MPI_THREAD_SERIALIZED: *provided = 2; break;
            case MPI_THREAD_MULTIPLE: *provided = 3; break;
            default: *provided = 0; break;
        }
    }
    
    return ret;
}

int ferrompi_init(void) {
    init_tables();
    int ret = MPI_Init(NULL, NULL);
    
    // Initialize COMM_WORLD after MPI_Init
    if (ret == MPI_SUCCESS) {
        comm_table[0] = MPI_COMM_WORLD;
    }
    
    return ret;
}

int ferrompi_finalize(void) {
    // Clean up requests first (they may reference communicators)
    for (int i = 0; i < MAX_REQUESTS; i++) {
        if (request_used[i] && request_table[i] != MPI_REQUEST_NULL) {
            MPI_Request_free(&request_table[i]);
        }
        request_used[i] = 0;
    }

    // Clean up any remaining info objects
    for (int i = 0; i < MAX_INFOS; i++) {
        if (info_used[i] && info_table[i] != MPI_INFO_NULL) {
            MPI_Info_free(&info_table[i]);
        }
        info_used[i] = 0;
    }

    // Clean up any remaining windows (before communicators, since windows reference comms)
    for (int i = 0; i < MAX_WINDOWS; i++) {
        if (win_used[i] && win_table[i] != MPI_WIN_NULL) {
            MPI_Win_free(&win_table[i]);
        }
        win_used[i] = 0;
    }

    // Clean up any remaining communicators
    for (int i = 1; i < comm_count; i++) {
        if (comm_table[i] != MPI_COMM_NULL) {
            MPI_Comm_free(&comm_table[i]);
            comm_table[i] = MPI_COMM_NULL;
        }
    }
    
    tables_initialized = 0;
    return MPI_Finalize();
}

int ferrompi_initialized(int* flag) {
    return MPI_Initialized(flag);
}

int ferrompi_finalized(int* flag) {
    return MPI_Finalized(flag);
}

/* ============================================================
 * Communicator Operations
 * ============================================================ */

int32_t ferrompi_comm_world(void) {
    return 0;  // COMM_WORLD is always handle 0
}

int ferrompi_comm_rank(int32_t comm_handle, int32_t* rank) {
    MPI_Comm comm = get_comm(comm_handle);
    int r;
    int ret = MPI_Comm_rank(comm, &r);
    *rank = (int32_t)r;
    return ret;
}

int ferrompi_comm_size(int32_t comm_handle, int32_t* size) {
    MPI_Comm comm = get_comm(comm_handle);
    int s;
    int ret = MPI_Comm_size(comm, &s);
    *size = (int32_t)s;
    return ret;
}

int ferrompi_comm_dup(int32_t comm_handle, int32_t* newcomm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Comm newcomm;
    int ret = MPI_Comm_dup(comm, &newcomm);
    if (ret == MPI_SUCCESS) {
        *newcomm_handle = alloc_comm(newcomm);
        if (*newcomm_handle < 0) {
            MPI_Comm_free(&newcomm);
            return MPI_ERR_OTHER;
        }
    }
    return ret;
}

int ferrompi_comm_free(int32_t comm_handle) {
    if (comm_handle == 0) {
        return MPI_ERR_COMM;  // Cannot free COMM_WORLD
    }
    if (comm_handle < 0 || comm_handle >= MAX_COMMS) {
        return MPI_ERR_COMM;
    }
    if (comm_table[comm_handle] == MPI_COMM_NULL) {
        return MPI_SUCCESS;  // Already freed
    }
    int ret = MPI_Comm_free(&comm_table[comm_handle]);
    comm_table[comm_handle] = MPI_COMM_NULL;
    return ret;
}

int ferrompi_comm_split(int32_t comm_handle, int32_t color, int32_t key, int32_t* newcomm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    int mpi_color = (color == -1) ? MPI_UNDEFINED : color;
    MPI_Comm newcomm;
    int ret = MPI_Comm_split(comm, mpi_color, key, &newcomm);
    if (ret == MPI_SUCCESS) {
        if (newcomm == MPI_COMM_NULL) {
            *newcomm_handle = -1;  // Process opted out
        } else {
            *newcomm_handle = alloc_comm(newcomm);
            if (*newcomm_handle < 0) {
                MPI_Comm_free(&newcomm);
                return MPI_ERR_OTHER;
            }
        }
    }
    return ret;
}

int ferrompi_comm_split_type(int32_t comm_handle, int32_t split_type, int32_t key, int32_t* newcomm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    int mpi_split_type;
    switch (split_type) {
        case FERROMPI_COMM_TYPE_SHARED:
            mpi_split_type = MPI_COMM_TYPE_SHARED;
            break;
        default:
            return MPI_ERR_ARG;
    }
    MPI_Comm newcomm;
    int ret = MPI_Comm_split_type(comm, mpi_split_type, key, MPI_INFO_NULL, &newcomm);
    if (ret == MPI_SUCCESS) {
        if (newcomm == MPI_COMM_NULL) {
            *newcomm_handle = -1;
        } else {
            *newcomm_handle = alloc_comm(newcomm);
            if (*newcomm_handle < 0) {
                MPI_Comm_free(&newcomm);
                return MPI_ERR_OTHER;
            }
        }
    }
    return ret;
}

/* ============================================================
 * Synchronization
 * ============================================================ */

int ferrompi_barrier(int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    return MPI_Barrier(comm);
}

/* ============================================================
 * Generic Point-to-Point Communication
 * ============================================================ */

int ferrompi_send(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Send_c(buf, (MPI_Count)count, dt, dest, tag, comm);
    }
#endif
    return MPI_Send(buf, (int)count, dt, dest, tag, comm);
}

int ferrompi_recv(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Status status;
    
    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;
    
    int ret;
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Recv_c(buf, (MPI_Count)count, dt, mpi_source, mpi_tag, comm, &status);
    } else
#endif
    {
        ret = MPI_Recv(buf, (int)count, dt, mpi_source, mpi_tag, comm, &status);
    }
    
    if (ret == MPI_SUCCESS) {
        *actual_source = status.MPI_SOURCE;
        *actual_tag = status.MPI_TAG;
#if MPI_VERSION >= 4
        MPI_Count cnt;
        MPI_Get_count_c(&status, dt, &cnt);
        *actual_count = (int64_t)cnt;
#else
        int cnt;
        MPI_Get_count(&status, dt, &cnt);
        *actual_count = (int64_t)cnt;
#endif
    }

    return ret;
}

int ferrompi_isend(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Isend_c(buf, (MPI_Count)count, dt, dest, tag, comm, &req);
    } else
#endif
    {
        ret = MPI_Isend(buf, (int)count, dt, dest, tag, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_irecv(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;

    int ret;
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Irecv_c(buf, (MPI_Count)count, dt, mpi_source, mpi_tag, comm, &req);
    } else
#endif
    {
        ret = MPI_Irecv(buf, (int)count, dt, mpi_source, mpi_tag, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

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
    int32_t comm_handle,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype send_dt = get_datatype(send_datatype_tag);
    MPI_Datatype recv_dt = get_datatype(recv_datatype_tag);
    if (send_dt == MPI_DATATYPE_NULL || recv_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Status status;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_recvtag = (recvtag == -1) ? MPI_ANY_TAG : recvtag;

    int ret;
#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        ret = MPI_Sendrecv_c(sendbuf, (MPI_Count)sendcount, send_dt, dest, sendtag,
                             recvbuf, (MPI_Count)recvcount, recv_dt, mpi_source, mpi_recvtag,
                             comm, &status);
    } else
#endif
    {
        ret = MPI_Sendrecv(sendbuf, (int)sendcount, send_dt, dest, sendtag,
                           recvbuf, (int)recvcount, recv_dt, mpi_source, mpi_recvtag,
                           comm, &status);
    }

    if (ret == MPI_SUCCESS) {
        *actual_source = status.MPI_SOURCE;
        *actual_tag = status.MPI_TAG;
#if MPI_VERSION >= 4
        MPI_Count cnt;
        MPI_Get_count_c(&status, recv_dt, &cnt);
        *actual_count = (int64_t)cnt;
#else
        int cnt;
        MPI_Get_count(&status, recv_dt, &cnt);
        *actual_count = (int64_t)cnt;
#endif
    }

    return ret;
}

/* ============================================================
 * Message Probing
 * ============================================================ */

int ferrompi_probe(int32_t source, int32_t tag, int32_t comm_handle,
                   int32_t* actual_source, int32_t* actual_tag,
                   int64_t* count, int32_t datatype_tag) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;

    MPI_Status status;
    int ret = MPI_Probe(mpi_source, mpi_tag, comm, &status);
    if (ret == MPI_SUCCESS) {
        *actual_source = status.MPI_SOURCE;
        *actual_tag = status.MPI_TAG;
#if MPI_VERSION >= 4
        MPI_Count cnt;
        MPI_Get_count_c(&status, dt, &cnt);
        *count = (int64_t)cnt;
#else
        int cnt;
        MPI_Get_count(&status, dt, &cnt);
        *count = (int64_t)cnt;
#endif
    }
    return ret;
}

int ferrompi_iprobe(int32_t source, int32_t tag, int32_t comm_handle,
                    int32_t* flag, int32_t* actual_source, int32_t* actual_tag,
                    int64_t* count, int32_t datatype_tag) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;

    MPI_Status status;
    int f;
    int ret = MPI_Iprobe(mpi_source, mpi_tag, comm, &f, &status);
    if (ret == MPI_SUCCESS) {
        *flag = f;
        if (f) {
            *actual_source = status.MPI_SOURCE;
            *actual_tag = status.MPI_TAG;
#if MPI_VERSION >= 4
            MPI_Count cnt;
            MPI_Get_count_c(&status, dt, &cnt);
            *count = (int64_t)cnt;
#else
            int cnt;
            MPI_Get_count(&status, dt, &cnt);
            *count = (int64_t)cnt;
#endif
        }
    }
    return ret;
}

/* ============================================================
 * Generic Collective Operations - Blocking
 * ============================================================ */

int ferrompi_bcast(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Bcast_c(buf, (MPI_Count)count, dt, root, comm);
    }
#endif
    return MPI_Bcast(buf, (int)count, dt, root, comm);
}

int ferrompi_reduce(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t root,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Reduce_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, root, comm);
    }
#endif
    return MPI_Reduce(sendbuf, recvbuf, (int)count, dt, mpi_op, root, comm);
}

int ferrompi_allreduce(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Allreduce_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm);
    }
#endif
    return MPI_Allreduce(sendbuf, recvbuf, (int)count, dt, mpi_op, comm);
}

int ferrompi_reduce_inplace(void* buf, int64_t count, int32_t datatype_tag,
                            int32_t op, int32_t root, int32_t is_root,
                            int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    MPI_Op mpi_op = get_op(op);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;

    if (is_root) {
        /* Root uses MPI_IN_PLACE as sendbuf; buf is both input and output */
#if MPI_VERSION >= 4
        if (count > INT_MAX) {
            return MPI_Reduce_c(MPI_IN_PLACE, buf, (MPI_Count)count, dt, mpi_op, root, comm);
        }
#endif
        return MPI_Reduce(MPI_IN_PLACE, buf, (int)count, dt, mpi_op, root, comm);
    } else {
        /* Non-root sends buf, recvbuf is ignored */
#if MPI_VERSION >= 4
        if (count > INT_MAX) {
            return MPI_Reduce_c(buf, NULL, (MPI_Count)count, dt, mpi_op, root, comm);
        }
#endif
        return MPI_Reduce(buf, NULL, (int)count, dt, mpi_op, root, comm);
    }
}

int ferrompi_allreduce_inplace(void* buf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Allreduce_c(MPI_IN_PLACE, buf, (MPI_Count)count, dt, mpi_op, comm);
    }
#endif
    return MPI_Allreduce(MPI_IN_PLACE, buf, (int)count, dt, mpi_op, comm);
}

int ferrompi_scan(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Scan_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm);
    }
#endif
    return MPI_Scan(sendbuf, recvbuf, (int)count, dt, mpi_op, comm);
}

int ferrompi_exscan(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Exscan_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm);
    }
#endif
    return MPI_Exscan(sendbuf, recvbuf, (int)count, dt, mpi_op, comm);
}

int ferrompi_gather(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        return MPI_Gather_c(sendbuf, (MPI_Count)sendcount, dt,
                           recvbuf, (MPI_Count)recvcount, dt,
                           root, comm);
    }
#endif
    return MPI_Gather(sendbuf, (int)sendcount, dt,
                      recvbuf, (int)recvcount, dt,
                      root, comm);
}

int ferrompi_allgather(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        return MPI_Allgather_c(sendbuf, (MPI_Count)sendcount, dt,
                               recvbuf, (MPI_Count)recvcount, dt,
                               comm);
    }
#endif
    return MPI_Allgather(sendbuf, (int)sendcount, dt,
                         recvbuf, (int)recvcount, dt,
                         comm);
}

int ferrompi_scatter(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        return MPI_Scatter_c(sendbuf, (MPI_Count)sendcount, dt,
                            recvbuf, (MPI_Count)recvcount, dt,
                            root, comm);
    }
#endif
    return MPI_Scatter(sendbuf, (int)sendcount, dt,
                       recvbuf, (int)recvcount, dt,
                       root, comm);
}

int ferrompi_alltoall(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        return MPI_Alltoall_c(sendbuf, (MPI_Count)sendcount, dt,
                              recvbuf, (MPI_Count)recvcount, dt, comm);
    }
#endif
    return MPI_Alltoall(sendbuf, (int)sendcount, dt,
                        recvbuf, (int)recvcount, dt, comm);
}

int ferrompi_reduce_scatter_block(
    const void* sendbuf,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    MPI_Op mpi_op = get_op(op);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
#if MPI_VERSION >= 4
    if (recvcount > INT_MAX) {
        return MPI_Reduce_scatter_block_c(sendbuf, recvbuf, (MPI_Count)recvcount, dt, mpi_op, comm);
    }
#endif
    return MPI_Reduce_scatter_block(sendbuf, recvbuf, (int)recvcount, dt, mpi_op, comm);
}

/* ============================================================
 * Generic V-Collectives (variable-count)
 * ============================================================ */

int ferrompi_gatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t root, int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    /* Cast int32_t* to int* — safe since int is at least 32 bits on all MPI platforms */
    return MPI_Gatherv(sendbuf, (int)sendcount, dt,
                       recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                       root, comm);
}

int ferrompi_scatterv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* displs,
    void* recvbuf, int64_t recvcount,
    int32_t datatype_tag, int32_t root, int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    /* Cast int32_t* to int* — safe since int is at least 32 bits on all MPI platforms */
    return MPI_Scatterv(sendbuf, (const int*)sendcounts, (const int*)displs, dt,
                        recvbuf, (int)recvcount, dt,
                        root, comm);
}

int ferrompi_allgatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    /* Cast int32_t* to int* — safe since int is at least 32 bits on all MPI platforms */
    return MPI_Allgatherv(sendbuf, (int)sendcount, dt,
                          recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                          comm);
}

int ferrompi_alltoallv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls,
    void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls,
    int32_t datatype_tag, int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    /* Cast int32_t* to int* — safe since int is at least 32 bits on all MPI platforms */
    return MPI_Alltoallv(sendbuf, (const int*)sendcounts, (const int*)sdispls, dt,
                         recvbuf, (const int*)recvcounts, (const int*)rdispls, dt,
                         comm);
}

/* ============================================================
 * Generic Collective Operations - Nonblocking
 * ============================================================ */

int ferrompi_ibcast(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;
    
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Ibcast_c(buf, (MPI_Count)count, dt, root, comm, &req);
    } else
#endif
    {
        ret = MPI_Ibcast(buf, (int)count, dt, root, comm, &req);
    }
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_iallreduce(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;
    
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Iallreduce_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm, &req);
    } else
#endif
    {
        ret = MPI_Iallreduce(sendbuf, recvbuf, (int)count, dt, mpi_op, comm, &req);
    }
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_ireduce(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Ireduce_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, root, comm, &req);
    } else
#endif
    {
        ret = MPI_Ireduce(sendbuf, recvbuf, (int)count, dt, mpi_op, root, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_igather(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        ret = MPI_Igather_c(sendbuf, (MPI_Count)sendcount, dt,
                            recvbuf, (MPI_Count)recvcount, dt,
                            root, comm, &req);
    } else
#endif
    {
        ret = MPI_Igather(sendbuf, (int)sendcount, dt,
                          recvbuf, (int)recvcount, dt,
                          root, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_iallgather(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        ret = MPI_Iallgather_c(sendbuf, (MPI_Count)sendcount, dt,
                                recvbuf, (MPI_Count)recvcount, dt,
                                comm, &req);
    } else
#endif
    {
        ret = MPI_Iallgather(sendbuf, (int)sendcount, dt,
                             recvbuf, (int)recvcount, dt,
                             comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_iscatter(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        ret = MPI_Iscatter_c(sendbuf, (MPI_Count)sendcount, dt,
                             recvbuf, (MPI_Count)recvcount, dt,
                             root, comm, &req);
    } else
#endif
    {
        ret = MPI_Iscatter(sendbuf, (int)sendcount, dt,
                           recvbuf, (int)recvcount, dt,
                           root, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_ibarrier(
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Request req;
    int ret = MPI_Ibarrier(comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_iscan(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Iscan_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm, &req);
    } else
#endif
    {
        ret = MPI_Iscan(sendbuf, recvbuf, (int)count, dt, mpi_op, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_iexscan(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Iexscan_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm, &req);
    } else
#endif
    {
        ret = MPI_Iexscan(sendbuf, recvbuf, (int)count, dt, mpi_op, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_ialltoall(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        ret = MPI_Ialltoall_c(sendbuf, (MPI_Count)sendcount, dt,
                               recvbuf, (MPI_Count)recvcount, dt, comm, &req);
    } else
#endif
    {
        ret = MPI_Ialltoall(sendbuf, (int)sendcount, dt,
                            recvbuf, (int)recvcount, dt, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_igatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t root, int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    /* Cast int32_t* to int* — safe since int is at least 32 bits on all MPI platforms */
    int ret = MPI_Igatherv(sendbuf, (int)sendcount, dt,
                           recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                           root, comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_iscatterv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* displs,
    void* recvbuf, int64_t recvcount,
    int32_t datatype_tag, int32_t root, int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    /* Cast int32_t* to int* — safe since int is at least 32 bits on all MPI platforms */
    int ret = MPI_Iscatterv(sendbuf, (const int*)sendcounts, (const int*)displs, dt,
                            recvbuf, (int)recvcount, dt,
                            root, comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_iallgatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    /* Cast int32_t* to int* — safe since int is at least 32 bits on all MPI platforms */
    int ret = MPI_Iallgatherv(sendbuf, (int)sendcount, dt,
                              recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                              comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_ialltoallv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls,
    void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls,
    int32_t datatype_tag, int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    /* Cast int32_t* to int* — safe since int is at least 32 bits on all MPI platforms */
    int ret = MPI_Ialltoallv(sendbuf, (const int*)sendcounts, (const int*)sdispls, dt,
                             recvbuf, (const int*)recvcounts, (const int*)rdispls, dt,
                             comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_ireduce_scatter_block(
    const void* sendbuf,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

#if MPI_VERSION >= 4
    if (recvcount > INT_MAX) {
        ret = MPI_Ireduce_scatter_block_c(sendbuf, recvbuf, (MPI_Count)recvcount, dt, mpi_op, comm, &req);
    } else
#endif
    {
        ret = MPI_Ireduce_scatter_block(sendbuf, recvbuf, (int)recvcount, dt, mpi_op, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

/* ============================================================
 * Generic Persistent Collectives (MPI 4.0+)
 * ============================================================ */

#if MPI_VERSION >= 4

int ferrompi_bcast_init(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    
    int ret = MPI_Bcast_init(buf, (int)count, dt, root, comm, MPI_INFO_NULL, &req);
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_allreduce_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    
    int ret = MPI_Allreduce_init(sendbuf, recvbuf, (int)count, dt,
                                  mpi_op, comm, MPI_INFO_NULL, &req);
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_allreduce_init_inplace(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    
    int ret = MPI_Allreduce_init(MPI_IN_PLACE, buf, (int)count, dt,
                                  mpi_op, comm, MPI_INFO_NULL, &req);
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_gather_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    
    int ret = MPI_Gather_init(sendbuf, (int)sendcount, dt,
                              recvbuf, (int)recvcount, dt,
                              root, comm, MPI_INFO_NULL, &req);
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_reduce_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;

    int ret = MPI_Reduce_init(sendbuf, recvbuf, (int)count, dt, mpi_op, root, comm,
                              MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_scatter_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int ret = MPI_Scatter_init(sendbuf, (int)sendcount, dt,
                               recvbuf, (int)recvcount, dt,
                               root, comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_allgather_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int ret = MPI_Allgather_init(sendbuf, (int)sendcount, dt,
                                 recvbuf, (int)recvcount, dt,
                                 comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_scan_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;

    int ret = MPI_Scan_init(sendbuf, recvbuf, (int)count, dt, mpi_op, comm,
                            MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_exscan_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;

    int ret = MPI_Exscan_init(sendbuf, recvbuf, (int)count, dt, mpi_op, comm,
                              MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_alltoall_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int ret = MPI_Alltoall_init(sendbuf, (int)sendcount, dt,
                                recvbuf, (int)recvcount, dt,
                                comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_gatherv_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    const int32_t* recvcounts,
    const int32_t* displs,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int ret = MPI_Gatherv_init(sendbuf, (int)sendcount, dt,
                               recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                               root, comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_scatterv_init(
    const void* sendbuf,
    const int32_t* sendcounts,
    const int32_t* displs,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int ret = MPI_Scatterv_init(sendbuf, (const int*)sendcounts, (const int*)displs, dt,
                                recvbuf, (int)recvcount, dt,
                                root, comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_allgatherv_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    const int32_t* recvcounts,
    const int32_t* displs,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int ret = MPI_Allgatherv_init(sendbuf, (int)sendcount, dt,
                                  recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                                  comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_alltoallv_init(
    const void* sendbuf,
    const int32_t* sendcounts,
    const int32_t* sdispls,
    void* recvbuf,
    const int32_t* recvcounts,
    const int32_t* rdispls,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int ret = MPI_Alltoallv_init(sendbuf, (const int*)sendcounts, (const int*)sdispls, dt,
                                 recvbuf, (const int*)recvcounts, (const int*)rdispls, dt,
                                 comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

int ferrompi_reduce_scatter_block_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;

    int ret = MPI_Reduce_scatter_block_init(sendbuf, recvbuf, (int)recvcount, dt, mpi_op, comm,
                                            MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }

    return ret;
}

#else /* MPI_VERSION < 4 */

int ferrompi_bcast_init(void* buf, int64_t count, int32_t datatype_tag, int32_t root,
                        int32_t comm_handle, int64_t* request_handle) {
    (void)buf; (void)count; (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_allreduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag,
                            int32_t op, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)datatype_tag; (void)op; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_allreduce_init_inplace(void* buf, int64_t count, int32_t datatype_tag, int32_t op,
                                    int32_t comm_handle, int64_t* request_handle) {
    (void)buf; (void)count; (void)datatype_tag; (void)op; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_gather_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount,
                         int32_t datatype_tag, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_reduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag,
                         int32_t op, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)datatype_tag;
    (void)op; (void)root; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_scatter_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount,
                          int32_t datatype_tag, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_allgather_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount,
                            int32_t datatype_tag, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_scan_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag,
                       int32_t op, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)datatype_tag;
    (void)op; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_exscan_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag,
                         int32_t op, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)datatype_tag;
    (void)op; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_alltoall_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount,
                           int32_t datatype_tag, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_gatherv_init(const void* sendbuf, int64_t sendcount, void* recvbuf,
                          const int32_t* recvcounts, const int32_t* displs,
                          int32_t datatype_tag, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcounts; (void)displs;
    (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_scatterv_init(const void* sendbuf, const int32_t* sendcounts, const int32_t* displs,
                           void* recvbuf, int64_t recvcount,
                           int32_t datatype_tag, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcounts; (void)displs; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_allgatherv_init(const void* sendbuf, int64_t sendcount, void* recvbuf,
                             const int32_t* recvcounts, const int32_t* displs,
                             int32_t datatype_tag, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcounts; (void)displs;
    (void)datatype_tag; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_alltoallv_init(const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls,
                            void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls,
                            int32_t datatype_tag, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcounts; (void)sdispls; (void)recvbuf; (void)recvcounts; (void)rdispls;
    (void)datatype_tag; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_reduce_scatter_block_init(const void* sendbuf, void* recvbuf, int64_t recvcount,
                                       int32_t datatype_tag, int32_t op, int32_t comm_handle,
                                       int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)recvcount; (void)datatype_tag;
    (void)op; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

#endif /* MPI_VERSION >= 4 */

/* ============================================================
 * Info Object Operations
 * ============================================================ */

int ferrompi_info_create(int32_t* info_handle) {
    MPI_Info info;
    int ret = MPI_Info_create(&info);
    if (ret == MPI_SUCCESS) {
        *info_handle = alloc_info(info);
        if (*info_handle < 0) {
            MPI_Info_free(&info);
            return MPI_ERR_OTHER;
        }
    }
    return ret;
}

int ferrompi_info_free(int32_t info_handle) {
    if (info_handle < 0 || info_handle >= MAX_INFOS || !info_used[info_handle]) {
        return MPI_SUCCESS;
    }
    int ret = MPI_Info_free(&info_table[info_handle]);
    free_info(info_handle);
    return ret;
}

int ferrompi_info_set(int32_t info_handle, const char* key, const char* value) {
    MPI_Info info = get_info(info_handle);
    if (info == MPI_INFO_NULL) return MPI_ERR_INFO;
    return MPI_Info_set(info, key, value);
}

int ferrompi_info_get(int32_t info_handle, const char* key, char* value, int32_t* valuelen, int32_t* flag) {
    MPI_Info info = get_info(info_handle);
    if (info == MPI_INFO_NULL) return MPI_ERR_INFO;
    int f;
#if MPI_VERSION >= 4
    int ret = MPI_Info_get_string(info, key, valuelen, value, &f);
#else
    /* MPI 3.x fallback: MPI_Info_get uses (info, key, valuelen, value, &flag) */
    /* where valuelen is input max length, value is output buffer */
    int vlen = *valuelen - 1;  /* MPI_Info_get expects max value length excluding null */
    if (vlen < 0) vlen = 0;
    int ret = MPI_Info_get(info, key, vlen, value, &f);
    if (f) {
        *valuelen = (int32_t)strlen(value);
    }
#endif
    *flag = f;
    return ret;
}

/* ============================================================
 * Error Information
 * ============================================================ */

int ferrompi_error_info(int code, int32_t* error_class, char* message, int32_t* msg_len) {
    int cls;
    int ret = MPI_Error_class(code, &cls);
    if (ret != MPI_SUCCESS) return ret;
    *error_class = (int32_t)cls;

    int len;
    ret = MPI_Error_string(code, message, &len);
    if (ret != MPI_SUCCESS) return ret;
    *msg_len = (int32_t)len;
    return MPI_SUCCESS;
}

/* ============================================================
 * Request Management
 * ============================================================ */

int ferrompi_wait(int64_t request_handle) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_ERR_REQUEST;
    }
    int ret = MPI_Wait(req, MPI_STATUS_IGNORE);
    // Don't free persistent requests automatically
    if (*req == MPI_REQUEST_NULL) {
        free_request(request_handle);
    }
    return ret;
}

int ferrompi_test(int64_t request_handle, int32_t* flag) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_ERR_REQUEST;
    }
    int f;
    int ret = MPI_Test(req, &f, MPI_STATUS_IGNORE);
    *flag = f;
    if (f && *req == MPI_REQUEST_NULL) {
        free_request(request_handle);
    }
    return ret;
}

int ferrompi_waitall(int64_t count, int64_t* request_handles) {
    if (count <= 0) return MPI_SUCCESS;
    
    // Allocate temporary array of MPI_Request pointers
    MPI_Request* reqs = (MPI_Request*)malloc(count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_OTHER;
    
    for (int64_t i = 0; i < count; i++) {
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) {
            free(reqs);
            return MPI_ERR_REQUEST;
        }
        reqs[i] = *req;
    }
    
    if (count > INT_MAX) {
        free(reqs);
        return MPI_ERR_COUNT;
    }
    int ret = MPI_Waitall((int)count, reqs, MPI_STATUSES_IGNORE);
    
    // Only update handles and free slots on success
    if (ret == MPI_SUCCESS) {
        for (int64_t i = 0; i < count; i++) {
            MPI_Request* req = get_request_ptr(request_handles[i]);
            if (req) {
                *req = reqs[i];
                if (*req == MPI_REQUEST_NULL) {
                    free_request(request_handles[i]);
                }
            }
        }
    }

    free(reqs);
    return ret;
}

int ferrompi_start(int64_t request_handle) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) return MPI_ERR_REQUEST;
    return MPI_Start(req);
}

int ferrompi_startall(int64_t count, int64_t* request_handles) {
    if (count <= 0) return MPI_SUCCESS;

    MPI_Request* reqs = (MPI_Request*)malloc(count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_NO_MEM;

    for (int64_t i = 0; i < count; i++) {
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) {
            free(reqs);
            return MPI_ERR_REQUEST;
        }
        reqs[i] = *req;
    }

    if (count > INT_MAX) {
        free(reqs);
        return MPI_ERR_COUNT;
    }
    int ret = MPI_Startall((int)count, reqs);

    // Update handles with post-start state
    for (int64_t i = 0; i < count; i++) {
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (req) {
            *req = reqs[i];
        }
    }

    free(reqs);
    return ret;
}

int ferrompi_request_free(int64_t request_handle) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_SUCCESS;  // Already freed
    }
    if (*req != MPI_REQUEST_NULL) {
        int ret = MPI_Request_free(req);
        if (ret != MPI_SUCCESS) return ret;
    }
    free_request(request_handle);
    return MPI_SUCCESS;
}

/* ============================================================
 * RMA Window Operations (MPI 3.0+)
 * ============================================================ */

#if MPI_VERSION >= 3

int ferrompi_win_allocate_shared(int64_t size, int32_t disp_unit, int32_t info_handle,
                                  int32_t comm_handle, void** baseptr, int32_t* win_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Info info = (info_handle < 0) ? MPI_INFO_NULL : get_info(info_handle);
    MPI_Win win;
    int ret = MPI_Win_allocate_shared((MPI_Aint)size, disp_unit, info, comm, baseptr, &win);
    if (ret == MPI_SUCCESS) {
        *win_handle = alloc_win(win);
        if (*win_handle < 0) {
            MPI_Win_free(&win);
            return MPI_ERR_OTHER;
        }
    }
    return ret;
}

int ferrompi_win_shared_query(int32_t win_handle, int32_t rank,
                               int64_t* size, int32_t* disp_unit, void** baseptr) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Aint sz;
    int du;
    int ret = MPI_Win_shared_query(win, rank, &sz, &du, baseptr);
    if (ret == MPI_SUCCESS) {
        *size = (int64_t)sz;
        *disp_unit = (int32_t)du;
    }
    return ret;
}

int ferrompi_win_free(int32_t win_handle) {
    MPI_Win* winp = get_win_ptr(win_handle);
    if (!winp) return MPI_SUCCESS;
    int ret = MPI_Win_free(winp);
    free_win(win_handle);
    return ret;
}

int ferrompi_win_fence(int32_t assert_val, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_fence(assert_val, win);
}

int ferrompi_win_lock(int32_t lock_type, int32_t rank, int32_t assert_val, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    int mpi_lock = (lock_type == FERROMPI_LOCK_SHARED) ? MPI_LOCK_SHARED : MPI_LOCK_EXCLUSIVE;
    return MPI_Win_lock(mpi_lock, rank, assert_val, win);
}

int ferrompi_win_unlock(int32_t rank, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_unlock(rank, win);
}

int ferrompi_win_lock_all(int32_t assert_val, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_lock_all(assert_val, win);
}

int ferrompi_win_unlock_all(int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_unlock_all(win);
}

int ferrompi_win_flush(int32_t rank, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_flush(rank, win);
}

int ferrompi_win_flush_all(int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_flush_all(win);
}

#else /* MPI_VERSION < 3 */

int ferrompi_win_allocate_shared(int64_t size, int32_t disp_unit, int32_t info_handle,
                                  int32_t comm_handle, void** baseptr, int32_t* win_handle) {
    (void)size; (void)disp_unit; (void)info_handle; (void)comm_handle; (void)baseptr; (void)win_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_win_shared_query(int32_t win_handle, int32_t rank,
                               int64_t* size, int32_t* disp_unit, void** baseptr) {
    (void)win_handle; (void)rank; (void)size; (void)disp_unit; (void)baseptr;
    return MPI_ERR_OTHER;
}

int ferrompi_win_free(int32_t win_handle) {
    (void)win_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_win_fence(int32_t assert_val, int32_t win_handle) {
    (void)assert_val; (void)win_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_win_lock(int32_t lock_type, int32_t rank, int32_t assert_val, int32_t win_handle) {
    (void)lock_type; (void)rank; (void)assert_val; (void)win_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_win_unlock(int32_t rank, int32_t win_handle) {
    (void)rank; (void)win_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_win_lock_all(int32_t assert_val, int32_t win_handle) {
    (void)assert_val; (void)win_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_win_unlock_all(int32_t win_handle) {
    (void)win_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_win_flush(int32_t rank, int32_t win_handle) {
    (void)rank; (void)win_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_win_flush_all(int32_t win_handle) {
    (void)win_handle;
    return MPI_ERR_OTHER;
}

#endif /* MPI_VERSION >= 3 */

/* ============================================================
 * Utility Functions
 * ============================================================ */

int ferrompi_get_library_version(char* buf, int32_t* len) {
    int l = 0;
    int ret = MPI_Get_library_version(buf, &l);
    if (ret == MPI_SUCCESS) {
        *len = l;
    } else {
        *len = 0;
    }
    return ret;
}

int ferrompi_get_version(char* version, int32_t* len) {
    int version_num, subversion_num;
    int ret = MPI_Get_version(&version_num, &subversion_num);
    if (ret == MPI_SUCCESS) {
        int n = snprintf(version, 256, "MPI %d.%d", version_num, subversion_num);
        /* Clamp: snprintf can return a value > buffer size per C standard */
        *len = (n < 0) ? 0 : (n > 255 ? 255 : n);
    }
    return ret;
}

int ferrompi_get_processor_name(char* name, int32_t* len) {
    int l = 0;
    int ret = MPI_Get_processor_name(name, &l);
    if (ret == MPI_SUCCESS) {
        *len = l;
    } else {
        *len = 0;
    }
    return ret;
}

double ferrompi_wtime(void) {
    return MPI_Wtime();
}

int ferrompi_abort(int32_t comm_handle, int32_t errorcode) {
    MPI_Comm comm = get_comm(comm_handle);
    return MPI_Abort(comm, errorcode);
}

/* ============================================================
 * Error Class Constants
 * ============================================================ */

int32_t ferrompi_err_file(void)  { return MPI_ERR_FILE; }
int32_t ferrompi_err_info(void)  { return MPI_ERR_INFO; }
int32_t ferrompi_err_win(void)   { return MPI_ERR_WIN; }
