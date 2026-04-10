//! Integration test for MPI Info object operations.
//!
//! Exercises Info::new, Info::null, Info::set, Info::get, Info::raw_handle,
//! and implicit drop for both created and null info objects.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_info

use ferrompi::{Info, Mpi};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    // ========================================================================
    // Test 1: Info::null() returns handle -1
    // ========================================================================
    {
        let null_info = Info::null();
        assert_eq!(
            null_info.raw_handle(),
            -1,
            "null info should have handle -1"
        );
        if rank == 0 {
            println!("PASS: Test 1 — Info::null() raw_handle = -1");
        }
        // Drop of null info is a no-op (no MPI_Info_free call)
    }

    // ========================================================================
    // Test 2: Info::new() creates a valid info object
    // ========================================================================
    {
        let info = Info::new().expect("Info::new failed");
        let handle = info.raw_handle();
        assert!(handle >= 0, "new info should have non-negative handle");
        if rank == 0 {
            println!("PASS: Test 2 — Info::new() raw_handle = {}", handle);
        }
        // Drop frees the MPI info object
    }

    // ========================================================================
    // Test 3: Info::set() and Info::get() for a single key-value pair
    // ========================================================================
    {
        let mut info = Info::new().expect("Info::new failed");
        info.set("test_key", "test_value")
            .expect("Info::set failed");
        let value = info
            .get("test_key")
            .expect("Info::get failed")
            .expect("Info::get returned None for existing key");
        assert_eq!(
            value, "test_value",
            "retrieved value should match set value"
        );
        if rank == 0 {
            println!("PASS: Test 3 — set/get single key-value pair");
        }
    }

    // ========================================================================
    // Test 4: Info::get() returns None for non-existent key
    // ========================================================================
    {
        let info = Info::new().expect("Info::new failed");
        let result = info
            .get("nonexistent_key")
            .expect("Info::get failed for non-existent key");
        assert!(
            result.is_none(),
            "get on non-existent key should return None"
        );
        if rank == 0 {
            println!("PASS: Test 4 — get non-existent key returns None");
        }
    }

    // ========================================================================
    // Test 5: Multiple key-value pairs
    // ========================================================================
    {
        let mut info = Info::new().expect("Info::new failed");
        info.set("key_a", "value_a").expect("set key_a failed");
        info.set("key_b", "value_b").expect("set key_b failed");
        info.set("key_c", "value_c").expect("set key_c failed");

        let val_a = info
            .get("key_a")
            .expect("get key_a failed")
            .expect("key_a not found");
        let val_b = info
            .get("key_b")
            .expect("get key_b failed")
            .expect("key_b not found");
        let val_c = info
            .get("key_c")
            .expect("get key_c failed")
            .expect("key_c not found");

        assert_eq!(val_a, "value_a", "key_a mismatch");
        assert_eq!(val_b, "value_b", "key_b mismatch");
        assert_eq!(val_c, "value_c", "key_c mismatch");

        if rank == 0 {
            println!("PASS: Test 5 — multiple key-value pairs");
        }
    }

    // ========================================================================
    // Test 6: Overwrite existing key
    // ========================================================================
    {
        let mut info = Info::new().expect("Info::new failed");
        info.set("overwrite_key", "original")
            .expect("set original failed");
        info.set("overwrite_key", "updated")
            .expect("set updated failed");

        let value = info
            .get("overwrite_key")
            .expect("get overwrite_key failed")
            .expect("overwrite_key not found after update");
        assert_eq!(value, "updated", "overwritten value should be 'updated'");

        if rank == 0 {
            println!("PASS: Test 6 — overwrite existing key");
        }
    }

    // ========================================================================
    // Test 7: Info::raw_handle() consistency
    // ========================================================================
    {
        let info = Info::new().expect("Info::new failed");
        let h1 = info.raw_handle();
        let h2 = info.raw_handle();
        assert_eq!(h1, h2, "raw_handle should be consistent across calls");
        assert!(h1 >= 0, "created info handle should be non-negative");

        if rank == 0 {
            println!("PASS: Test 7 — raw_handle consistency");
        }
    }

    // Synchronize before finishing
    world.barrier().expect("final barrier failed");

    drop(mpi);

    if rank == 0 {
        println!("\n========================================");
        println!("All info tests passed!");
        println!("========================================");
    }
}
