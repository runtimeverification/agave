pub use self::{
    mem_ops::SyscallMemcpy,
};
#[allow(deprecated)]
use {
    solana_program_runtime::{
        invoke_context::InvokeContext,
    },
    solana_sbpf::{
        declare_builtin_function,
        memory_region::{AccessType, MemoryMapping},
    },
    std::{
        mem::{align_of, size_of},
        slice::from_raw_parts_mut,
    },
    thiserror::Error as ThisError,
};

mod mem_ops;

/// Error definitions
#[derive(Debug, ThisError, PartialEq, Eq)]
pub enum SyscallError {
    #[error("Unaligned pointer")]
    UnalignedPointer,
    #[error("Overlapping copy")]
    CopyOverlapping,
    #[error("InvalidLength")]
    InvalidLength,
}

type Error = Box<dyn std::error::Error>;

// The VmSlice class is used for cases when you need a slice that is stored in the BPF
// interpreter's virtual address space. Because this source code can be compiled with
// addresses of different bit depths, we cannot assume that the 64-bit BPF interpreter's
// pointer sizes can be mapped to physical pointer sizes. In particular, if you need a
// slice-of-slices in the virtual space, the inner slices will be different sizes in a
// 32-bit app build than in the 64-bit virtual space. Therefore instead of a slice-of-slices,
// you should implement a slice-of-VmSlices, which can then use VmSlice::translate() to
// map to the physical address.
// This class must consist only of 16 bytes: a u64 ptr and a u64 len, to match the 64-bit
// implementation of a slice in Rust. The PhantomData entry takes up 0 bytes.


fn consume_compute_meter(invoke_context: &InvokeContext, amount: u64) -> Result<(), Error> {
    invoke_context.consume_checked(amount)?;
    Ok(())
}

fn address_is_aligned<T>(address: u64) -> bool {
    (address as *mut T as usize)
        .checked_rem(align_of::<T>())
        .map(|rem| rem == 0)
        .expect("T to be non-zero aligned")
}

// Do not use this directly
#[macro_export]
macro_rules! translate_inner {
    ($memory_mapping:expr, $map:ident, $access_type:expr, $vm_addr:expr, $len:expr $(,)?) => {
        Result::<u64, Error>::from(
            $memory_mapping
                .$map($access_type, $vm_addr, $len)
                .map_err(|err| err.into()),
        )
    };
}
// Do not use this directly
#[macro_export]
macro_rules! translate_type_inner {
    ($memory_mapping:expr, $access_type:expr, $vm_addr:expr, $T:ty, $check_aligned:expr $(,)?) => {{
        let host_addr = translate_inner!(
            $memory_mapping,
            map,
            $access_type,
            $vm_addr,
            size_of::<$T>() as u64
        )?;
        if !$check_aligned {
            Ok(unsafe { std::mem::transmute::<u64, &mut $T>(host_addr) })
        } else if !address_is_aligned::<$T>(host_addr) {
            Err(SyscallError::UnalignedPointer.into())
        } else {
            Ok(unsafe { &mut *(host_addr as *mut $T) })
        }
    }};
}
// Do not use this directly
#[macro_export]
macro_rules! translate_slice_inner {
    ($memory_mapping:expr, $access_type:expr, $vm_addr:expr, $len:expr, $T:ty, $check_aligned:expr $(,)?) => {{
        if $len == 0 {
            return Ok(&mut []);
        }
        let total_size = $len.saturating_mul(size_of::<$T>() as u64);
        if isize::try_from(total_size).is_err() {
            return Err(SyscallError::InvalidLength.into());
        }
        let host_addr = translate_inner!($memory_mapping, map, $access_type, $vm_addr, total_size)?;
        if $check_aligned && !address_is_aligned::<$T>(host_addr) {
            return Err(SyscallError::UnalignedPointer.into());
        }
        Ok(unsafe { from_raw_parts_mut(host_addr as *mut $T, $len as usize) })
    }};
}

fn translate_slice<'a, T>(
    memory_mapping: &'a MemoryMapping,
    vm_addr: u64,
    len: u64,
    check_aligned: bool,
) -> Result<&'a [T], Error> {
    translate_slice_inner!(
        memory_mapping,
        AccessType::Load,
        vm_addr,
        len,
        T,
        check_aligned,
    )
    .map(|value| &*value)
}

fn translate_slice_mut<'a, T>(
    memory_mapping: &'a MemoryMapping,
    vm_addr: u64,
    len: u64,
    check_aligned: bool,
) -> Result<&'a mut [T], Error> {
    translate_slice_inner!(
        memory_mapping,
        AccessType::Store,
        vm_addr,
        len,
        T,
        check_aligned,
    )
}

fn touch_slice_mut<T>(
    memory_mapping: &mut MemoryMapping,
    vm_addr: u64,
    element_count: u64,
) -> Result<(), Error> {
    if element_count == 0 {
        return Ok(());
    }
    translate_inner!(
        memory_mapping,
        map_with_access_violation_handler,
        AccessType::Store,
        vm_addr,
        element_count.saturating_mul(size_of::<T>() as u64),
    )
    .map(|_| ())
}

// No other translated references can be live when calling this.
// Meaning it should generally be at the beginning or end of a syscall and
// it should only be called once with all translations passed in one call.
#[macro_export]
macro_rules! translate_mut {
    (internal, $memory_mapping:expr, &mut [$T:ty], $vm_addr_and_element_count:expr) => {
        touch_slice_mut::<$T>(
            $memory_mapping,
            $vm_addr_and_element_count.0,
            $vm_addr_and_element_count.1,
        )?
    };
    (internal, $memory_mapping:expr, &mut $T:ty, $vm_addr:expr) => {
        touch_type_mut::<$T>(
            $memory_mapping,
            $vm_addr,
        )?
    };
    (internal, $memory_mapping:expr, $check_aligned:expr, &mut [$T:ty], $vm_addr_and_element_count:expr) => {{
        let slice = translate_slice_mut::<$T>(
            $memory_mapping,
            $vm_addr_and_element_count.0,
            $vm_addr_and_element_count.1,
            $check_aligned,
        )?;
        let host_addr = slice.as_ptr() as usize;
        (slice, host_addr, std::mem::size_of::<$T>().saturating_mul($vm_addr_and_element_count.1 as usize))
    }};
    (internal, $memory_mapping:expr, $check_aligned:expr, &mut $T:ty, $vm_addr:expr) => {{
        let reference = translate_type_mut::<$T>(
            $memory_mapping,
            $vm_addr,
            $check_aligned,
        )?;
        let host_addr = reference as *const _ as usize;
        (reference, host_addr, std::mem::size_of::<$T>())
    }};
    ($memory_mapping:expr, $check_aligned:expr, $(let $binding:ident : &mut $T:tt = map($vm_addr:expr $(, $element_count:expr)?) $try:tt;)+) => {
        // This ensures that all the parameters are collected first so that if they depend on previous translations
        $(let $binding = ($vm_addr $(, $element_count)?);)+
        // they are not invalidated by the following translations here:
        $(translate_mut!(internal, $memory_mapping, &mut $T, $binding);)+
        $(let $binding = translate_mut!(internal, $memory_mapping, $check_aligned, &mut $T, $binding);)+
        let host_ranges = [
            $(($binding.1, $binding.2),)+
        ];
        for (index, range_a) in host_ranges.get(..host_ranges.len().saturating_sub(1)).unwrap().iter().enumerate() {
            for range_b in host_ranges.get(index.saturating_add(1)..).unwrap().iter() {
                if !is_nonoverlapping(range_a.0, range_a.1, range_b.0, range_b.1) {
                    return Err(SyscallError::CopyOverlapping.into());
                }
            }
        }
        $(let $binding = $binding.0;)+
    };
}
