//! Raw FFI bindings to the C wrapper layer.
//!
//! These are low-level unsafe functions. Use the safe wrappers in the parent module.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_double, c_int, c_void};

// Type aliases matching the C header
pub type int32_t = i32;
pub type int64_t = i64;

// Lock type constants matching the C header defines
#[cfg(feature = "rma")]
pub const FERROMPI_LOCK_EXCLUSIVE: int32_t = 0;
#[cfg(feature = "rma")]
pub const FERROMPI_LOCK_SHARED: int32_t = 1;

extern "C" {
    // ============================================================
    // Initialization and Finalization
    // ============================================================

    pub fn ferrompi_init_thread(required: c_int, provided: *mut c_int) -> c_int;
    pub fn ferrompi_finalize() -> c_int;
    pub fn ferrompi_initialized(flag: *mut c_int) -> c_int;
    pub fn ferrompi_finalized(flag: *mut c_int) -> c_int;

    // ============================================================
    // Communicator Operations
    // ============================================================

    pub fn ferrompi_comm_world() -> int32_t;
    pub fn ferrompi_comm_rank(comm: int32_t, rank: *mut int32_t) -> c_int;
    pub fn ferrompi_comm_size(comm: int32_t, size: *mut int32_t) -> c_int;
    pub fn ferrompi_comm_dup(comm: int32_t, newcomm: *mut int32_t) -> c_int;
    pub fn ferrompi_comm_free(comm: int32_t) -> c_int;
    pub fn ferrompi_comm_split(
        comm: int32_t,
        color: int32_t,
        key: int32_t,
        newcomm: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_comm_split_type(
        comm: int32_t,
        split_type: int32_t,
        key: int32_t,
        newcomm: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_comm_create_from_group_parent(
        comm: int32_t,
        group: int32_t,
        newcomm: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_comm_create_from_group(
        group: int32_t,
        stringtag: *const c_char,
        newcomm: *mut int32_t,
    ) -> c_int;

    // ============================================================
    // Group Operations
    // ============================================================

    pub fn ferrompi_mpi_undefined() -> int32_t;
    pub fn ferrompi_comm_group(comm: int32_t, group_handle: *mut int32_t) -> c_int;
    pub fn ferrompi_group_incl(
        group_handle: int32_t,
        n: int32_t,
        ranks: *const int32_t,
        newgroup_handle: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_group_excl(
        group_handle: int32_t,
        n: int32_t,
        ranks: *const int32_t,
        newgroup_handle: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_group_free(group_handle: int32_t) -> c_int;
    pub fn ferrompi_group_size(group_handle: int32_t, size: *mut int32_t) -> c_int;
    pub fn ferrompi_group_rank(group_handle: int32_t, rank: *mut int32_t) -> c_int;
    pub fn ferrompi_group_union(
        group1_handle: int32_t,
        group2_handle: int32_t,
        newgroup_handle: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_group_intersection(
        group1_handle: int32_t,
        group2_handle: int32_t,
        newgroup_handle: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_group_difference(
        group1_handle: int32_t,
        group2_handle: int32_t,
        newgroup_handle: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_group_range_incl(
        group_handle: int32_t,
        n: int32_t,
        ranges_flat: *const int32_t,
        newgroup_handle: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_group_range_excl(
        group_handle: int32_t,
        n: int32_t,
        ranges_flat: *const int32_t,
        newgroup_handle: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_group_compare(
        group1_handle: int32_t,
        group2_handle: int32_t,
        result: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_group_translate_ranks(
        group1_handle: int32_t,
        n: int32_t,
        ranks1: *const int32_t,
        group2_handle: int32_t,
        ranks2: *mut int32_t,
    ) -> c_int;

    // ============================================================
    // Synchronization
    // ============================================================

    pub fn ferrompi_barrier(comm: int32_t) -> c_int;

    // ============================================================
    // Generic Point-to-Point Communication
    // ============================================================

    pub fn ferrompi_send(
        buf: *const c_void,
        count: int64_t,
        datatype_tag: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_recv(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        actual_source: *mut int32_t,
        actual_tag: *mut int32_t,
        actual_count: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_isend(
        buf: *const c_void,
        count: int64_t,
        datatype_tag: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_irecv(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_sendrecv(
        sendbuf: *const c_void,
        sendcount: int64_t,
        send_datatype_tag: int32_t,
        dest: int32_t,
        sendtag: int32_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        recv_datatype_tag: int32_t,
        source: int32_t,
        recvtag: int32_t,
        comm: int32_t,
        actual_source: *mut int32_t,
        actual_tag: *mut int32_t,
        actual_count: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // Message Probing
    // ============================================================

    pub fn ferrompi_probe(
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        actual_source: *mut int32_t,
        actual_tag: *mut int32_t,
        count: *mut int64_t,
        datatype_tag: int32_t,
    ) -> c_int;

    pub fn ferrompi_iprobe(
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        flag: *mut int32_t,
        actual_source: *mut int32_t,
        actual_tag: *mut int32_t,
        count: *mut int64_t,
        datatype_tag: int32_t,
    ) -> c_int;

    // ============================================================
    // Generic Collective Operations - Blocking
    // ============================================================

    pub fn ferrompi_bcast(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_reduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_reduce_inplace(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        root: int32_t,
        is_root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allreduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_inplace(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_gather_inplace(
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        is_root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allgather_inplace(
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_scatter_inplace(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        is_root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_alltoall_inplace(
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_scan(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_exscan(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_gather(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allgather(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_scatter(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_alltoall(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_reduce_scatter_block(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    // ============================================================
    // Generic V-Collectives (variable-count)
    // ============================================================

    pub fn ferrompi_gatherv(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        displs: *const int32_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_scatterv(
        sendbuf: *const c_void,
        sendcounts: *const int32_t,
        displs: *const int32_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allgatherv(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        displs: *const int32_t,
        datatype_tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_alltoallv(
        sendbuf: *const c_void,
        sendcounts: *const int32_t,
        sdispls: *const int32_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        rdispls: *const int32_t,
        datatype_tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    // ============================================================
    // Generic Collective Operations - Nonblocking
    // ============================================================

    pub fn ferrompi_ibcast(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iallreduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_ireduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_igather(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iallgather(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iscatter(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_ibarrier(comm: int32_t, request: *mut int64_t) -> c_int;

    pub fn ferrompi_iscan(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iexscan(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_ialltoall(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_igather_inplace(
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        is_root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iallgather_inplace(
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iscatter_inplace(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        is_root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_ialltoall_inplace(
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_igatherv(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        displs: *const int32_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iscatterv(
        sendbuf: *const c_void,
        sendcounts: *const int32_t,
        displs: *const int32_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iallgatherv(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        displs: *const int32_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_ialltoallv(
        sendbuf: *const c_void,
        sendcounts: *const int32_t,
        sdispls: *const int32_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        rdispls: *const int32_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_ireduce_scatter_block(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // Persistent Point-to-Point (MPI 1.1+)
    // ============================================================

    pub fn ferrompi_send_init(
        buf: *const c_void,
        count: int64_t,
        datatype_tag: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_recv_init(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_rsend_init(
        buf: *const c_void,
        count: int64_t,
        datatype_tag: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_ssend_init(
        buf: *const c_void,
        count: int64_t,
        datatype_tag: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_buffer_attach(buffer: *mut c_void, size: int64_t) -> c_int;

    pub fn ferrompi_buffer_detach(buffer: *mut *mut c_void, size: *mut int64_t) -> c_int;

    pub fn ferrompi_bsend_init(
        buf: *const c_void,
        count: int64_t,
        datatype_tag: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // Generic Persistent Collectives (MPI 4.0+)
    // ============================================================

    pub fn ferrompi_bcast_init(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_init(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_init_inplace(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_gather_init(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_reduce_init(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_scatter_init(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_allgather_init(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_scan_init(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_exscan_init(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_alltoall_init(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_gather_init_inplace(
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        is_root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_allgather_init_inplace(
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_scatter_init_inplace(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        is_root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_alltoall_init_inplace(
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_gatherv_init(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        displs: *const int32_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_scatterv_init(
        sendbuf: *const c_void,
        sendcounts: *const int32_t,
        displs: *const int32_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_allgatherv_init(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        displs: *const int32_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_alltoallv_init(
        sendbuf: *const c_void,
        sendcounts: *const int32_t,
        sdispls: *const int32_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        rdispls: *const int32_t,
        datatype_tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_reduce_scatter_block_init(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // Info Object Operations
    // ============================================================

    pub fn ferrompi_info_create(info_handle: *mut int32_t) -> c_int;
    pub fn ferrompi_info_free(info_handle: int32_t) -> c_int;
    pub fn ferrompi_info_set(
        info_handle: int32_t,
        key: *const c_char,
        value: *const c_char,
    ) -> c_int;
    pub fn ferrompi_info_get(
        info_handle: int32_t,
        key: *const c_char,
        value: *mut c_char,
        valuelen: *mut int32_t,
        flag: *mut int32_t,
    ) -> c_int;

    // ============================================================
    // Error Information
    // ============================================================

    pub fn ferrompi_error_info(
        code: c_int,
        error_class: *mut int32_t,
        message: *mut c_char,
        msg_len: *mut int32_t,
    ) -> c_int;

    // ============================================================
    // Request Management
    // ============================================================

    pub fn ferrompi_wait(request: int64_t) -> c_int;
    pub fn ferrompi_test(request: int64_t, flag: *mut int32_t) -> c_int;
    pub fn ferrompi_waitall(count: int64_t, requests: *mut int64_t) -> c_int;
    pub fn ferrompi_request_free(request: int64_t) -> c_int;
    pub fn ferrompi_request_get_status(request: int64_t, flag: *mut int32_t) -> c_int;
    pub fn ferrompi_cancel(request: int64_t) -> c_int;
    pub fn ferrompi_waitany(count: int64_t, requests: *mut int64_t, index: *mut int32_t) -> c_int;
    pub fn ferrompi_waitsome(
        count: int64_t,
        requests: *mut int64_t,
        outcount: *mut int64_t,
        indices: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_testany(
        count: int64_t,
        requests: *mut int64_t,
        index: *mut int32_t,
        flag: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_testsome(
        count: int64_t,
        requests: *mut int64_t,
        outcount: *mut int64_t,
        indices: *mut int32_t,
    ) -> c_int;

    // ============================================================
    // Persistent Request Management
    // ============================================================

    pub fn ferrompi_start(request: int64_t) -> c_int;
    pub fn ferrompi_startall(count: int64_t, requests: *mut int64_t) -> c_int;

    // ============================================================
    // Utility Functions
    // ============================================================

    pub fn ferrompi_get_library_version(buf: *mut c_char, len: *mut int32_t) -> c_int;
    pub fn ferrompi_get_version(version: *mut c_char, len: *mut int32_t) -> c_int;
    pub fn ferrompi_get_processor_name(name: *mut c_char, len: *mut int32_t) -> c_int;
    pub fn ferrompi_wtime() -> c_double;
    pub fn ferrompi_abort(comm: int32_t, errorcode: int32_t) -> c_int;

    // ============================================================
    // Custom Datatype Operations
    // ============================================================

    pub fn ferrompi_type_contiguous(
        count: int32_t,
        basetype_tag: int32_t,
        newtype_handle: *mut int32_t,
    ) -> c_int;

    pub fn ferrompi_type_vector(
        count: int32_t,
        blocklength: int32_t,
        stride: int32_t,
        basetype_tag: int32_t,
        newtype_handle: *mut int32_t,
    ) -> c_int;

    pub fn ferrompi_type_create_struct(
        count: int32_t,
        blocklengths: *const int32_t,
        displacements: *const int64_t,
        basetype_tags: *const int32_t,
        newtype_handle: *mut int32_t,
    ) -> c_int;

    pub fn ferrompi_type_create_resized(
        old_handle: int32_t,
        lb: int64_t,
        extent: int64_t,
        newtype_handle: *mut int32_t,
    ) -> c_int;

    pub fn ferrompi_type_free(type_handle: int32_t) -> c_int;

    // ============================================================
    // Custom-Datatype Point-to-Point
    // ============================================================

    pub fn ferrompi_send_custom(
        buf: *const c_void,
        count: int64_t,
        datatype_handle: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_recv_custom(
        buf: *mut c_void,
        count: int64_t,
        datatype_handle: int32_t,
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        actual_source: *mut int32_t,
        actual_tag: *mut int32_t,
        actual_count: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_isend_custom(
        buf: *const c_void,
        count: int64_t,
        datatype_handle: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_irecv_custom(
        buf: *mut c_void,
        count: int64_t,
        datatype_handle: int32_t,
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // User-Defined Reduction Op (MPI_Op_create)
    // ============================================================

    /// Allocate a free slot in the op-slot table.
    /// Writes the slot index to `*out_slot`.
    /// Returns MPI_SUCCESS on success, MPI_ERR_OTHER if the table is full.
    pub fn ferrompi_op_alloc_slot(out_slot: *mut int32_t) -> c_int;

    /// Store the Rust fat-pointer halves (data + vtable) for the given slot.
    /// Must be called after `ferrompi_op_alloc_slot` and before
    /// `ferrompi_op_create_user`.
    pub fn ferrompi_op_set_closure(slot: int32_t, data: *mut c_void, vtbl: *mut c_void);

    /// Create an MPI_Op for the given slot.
    /// `commute = 1` → commutative; `commute = 0` → non-commutative.
    /// Writes the handle (same value as `slot`) to `*out_handle`.
    pub fn ferrompi_op_create_user(
        slot: int32_t,
        commute: int32_t,
        out_handle: *mut int32_t,
    ) -> c_int;

    /// Free the MPI_Op and release the slot.
    /// Drop ordering: MPI_Op_free → ferrompi_op_drop_closure → free_op_slot.
    pub fn ferrompi_op_free(handle: int32_t) -> c_int;

    /// Release the op slot WITHOUT calling MPI_Op_free.
    ///
    /// Use this in rollback paths where `MPI_Op_create` failed and the slot
    /// therefore holds `MPI_OP_NULL`.  Clears the closure pointers and marks
    /// the slot as unused.  Does not invoke `ferrompi_op_drop_closure` — the
    /// caller must have already dropped the closure before calling this.
    pub fn ferrompi_op_free_slot_only(handle: int32_t) -> c_int;

    /// MPI_Allreduce using a user-defined reduction op.
    pub fn ferrompi_allreduce_user_op(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op_handle: int32_t,
        comm: int32_t,
    ) -> c_int;

    // ============================================================
    // Error Class Constants
    // ============================================================

    pub fn ferrompi_err_file() -> int32_t;
    pub fn ferrompi_err_info() -> int32_t;
    pub fn ferrompi_err_win() -> int32_t;

}

#[cfg(feature = "rma")]
extern "C" {
    // ============================================================
    // RMA / Window
    // ============================================================

    pub fn ferrompi_win_allocate_shared(
        size: int64_t,
        disp_unit: int32_t,
        info: int32_t,
        comm: int32_t,
        baseptr: *mut *mut c_void,
        win: *mut int32_t,
    ) -> c_int;

    pub fn ferrompi_win_create(
        base: *mut c_void,
        size: int64_t,
        disp_unit: int32_t,
        info: int32_t,
        comm: int32_t,
        win: *mut int32_t,
    ) -> c_int;

    pub fn ferrompi_win_allocate(
        size: int64_t,
        disp_unit: int32_t,
        info: int32_t,
        comm: int32_t,
        baseptr: *mut *mut c_void,
        win: *mut int32_t,
    ) -> c_int;

    pub fn ferrompi_win_shared_query(
        win: int32_t,
        rank: int32_t,
        size: *mut int64_t,
        disp_unit: *mut int32_t,
        baseptr: *mut *mut c_void,
    ) -> c_int;

    pub fn ferrompi_win_free(win: int32_t) -> c_int;

    // ============================================================
    // RMA / Window Synchronization
    // ============================================================

    pub fn ferrompi_win_fence(assert_val: int32_t, win: int32_t) -> c_int;

    pub fn ferrompi_win_fence_mode_values(out: *mut int32_t);

    pub fn ferrompi_win_lock(
        lock_type: int32_t,
        rank: int32_t,
        assert_val: int32_t,
        win: int32_t,
    ) -> c_int;

    pub fn ferrompi_win_unlock(rank: int32_t, win: int32_t) -> c_int;

    pub fn ferrompi_win_lock_all(assert_val: int32_t, win: int32_t) -> c_int;

    pub fn ferrompi_win_unlock_all(win: int32_t) -> c_int;

    pub fn ferrompi_win_flush(rank: int32_t, win: int32_t) -> c_int;

    pub fn ferrompi_win_flush_all(win: int32_t) -> c_int;

    pub fn ferrompi_win_flush_local(rank: int32_t, win: int32_t) -> c_int;

    pub fn ferrompi_win_flush_local_all(win: int32_t) -> c_int;

    pub fn ferrompi_win_sync(win: int32_t) -> c_int;

    pub fn ferrompi_win_post(group: int32_t, assert_val: int32_t, win: int32_t) -> c_int;

    pub fn ferrompi_win_start(group: int32_t, assert_val: int32_t, win: int32_t) -> c_int;

    pub fn ferrompi_win_complete(win: int32_t) -> c_int;

    pub fn ferrompi_win_wait(win: int32_t) -> c_int;

    pub fn ferrompi_win_test(win: int32_t, flag: *mut int32_t) -> c_int;

    pub fn ferrompi_win_pscw_mode_values(out: *mut int32_t);

    pub fn ferrompi_put(
        origin: *const c_void,
        origin_count: int64_t,
        origin_dt_tag: int32_t,
        target_rank: int32_t,
        target_disp: int64_t,
        target_count: int64_t,
        target_dt_tag: int32_t,
        win_handle: int32_t,
    ) -> c_int;

    pub fn ferrompi_rput(
        origin: *const c_void,
        origin_count: int64_t,
        origin_dt_tag: int32_t,
        target_rank: int32_t,
        target_disp: int64_t,
        target_count: int64_t,
        target_dt_tag: int32_t,
        win_handle: int32_t,
        request_handle: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_get(
        origin: *mut c_void,
        origin_count: int64_t,
        origin_dt_tag: int32_t,
        target_rank: int32_t,
        target_disp: int64_t,
        target_count: int64_t,
        target_dt_tag: int32_t,
        win_handle: int32_t,
    ) -> c_int;

    pub fn ferrompi_rget(
        origin: *mut c_void,
        origin_count: int64_t,
        origin_dt_tag: int32_t,
        target_rank: int32_t,
        target_disp: int64_t,
        target_count: int64_t,
        target_dt_tag: int32_t,
        win_handle: int32_t,
        request_handle: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_accumulate(
        origin: *const c_void,
        origin_count: int64_t,
        origin_dt_tag: int32_t,
        target_rank: int32_t,
        target_disp: int64_t,
        target_count: int64_t,
        target_dt_tag: int32_t,
        op_tag: int32_t,
        win_handle: int32_t,
    ) -> c_int;

    pub fn ferrompi_raccumulate(
        origin: *const c_void,
        origin_count: int64_t,
        origin_dt_tag: int32_t,
        target_rank: int32_t,
        target_disp: int64_t,
        target_count: int64_t,
        target_dt_tag: int32_t,
        op_tag: int32_t,
        win_handle: int32_t,
        request_handle: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_get_accumulate(
        origin: *const c_void,
        origin_count: int64_t,
        origin_dt_tag: int32_t,
        result: *mut c_void,
        result_count: int64_t,
        result_dt_tag: int32_t,
        target_rank: int32_t,
        target_disp: int64_t,
        target_count: int64_t,
        target_dt_tag: int32_t,
        op_tag: int32_t,
        win_handle: int32_t,
    ) -> c_int;

    pub fn ferrompi_fetch_and_op(
        origin: *const c_void,
        result: *mut c_void,
        dt_tag: int32_t,
        target_rank: int32_t,
        target_disp: int64_t,
        op_tag: int32_t,
        win_handle: int32_t,
    ) -> c_int;

    pub fn ferrompi_compare_and_swap(
        origin: *const c_void,
        compare: *const c_void,
        result: *mut c_void,
        dt_tag: int32_t,
        target_rank: int32_t,
        target_disp: int64_t,
        win_handle: int32_t,
    ) -> c_int;
}
