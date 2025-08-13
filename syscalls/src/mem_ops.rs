use super::{InvokeContext, declare_builtin_function, Error, MemoryMapping, SyscallError, consume_compute_meter, translate_mut, translate_slice, translate_slice_mut, touch_slice_mut};

fn mem_op_consume(invoke_context: &mut InvokeContext, n: u64) -> Result<(), Error> {
    let compute_cost = invoke_context.get_execution_cost();
    let cost = compute_cost.mem_op_base_cost.max(
        n.checked_div(compute_cost.cpi_bytes_per_unit)
            .unwrap_or(u64::MAX),
    );
    consume_compute_meter(invoke_context, cost)
}

/// Check that two regions do not overlap.
pub(crate) fn is_nonoverlapping<N>(src: N, src_len: N, dst: N, dst_len: N) -> bool
where
    N: Ord + num_traits::SaturatingSub,
{
    // If the absolute distance between the ptrs is at least as big as the size of the other,
    // they do not overlap.
    if src > dst {
        src.saturating_sub(&dst) >= dst_len
    } else {
        dst.saturating_sub(&src) >= src_len
    }
}

declare_builtin_function!(
    /// memcpy
    SyscallMemcpy,
    fn rust(
        invoke_context: &mut InvokeContext,
        dst_addr: u64,
        src_addr: u64,
        n: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Error> {
        mem_op_consume(invoke_context, n)?;

        if !is_nonoverlapping(src_addr, n, dst_addr, n) {
            return Err(SyscallError::CopyOverlapping.into());
        }

        // host addresses can overlap so we always invoke memmove
        memmove(invoke_context, dst_addr, src_addr, n, memory_mapping)
    }
);

fn memmove(
    invoke_context: &mut InvokeContext,
    dst_addr: u64,
    src_addr: u64,
    n: u64,
    memory_mapping: &mut MemoryMapping,
) -> Result<u64, Error> {
    translate_mut!(
        memory_mapping,
        invoke_context.get_check_aligned(),
        let dst_ref_mut: &mut [u8] = map(dst_addr, n)?;
    );
    let dst_ptr = dst_ref_mut.as_mut_ptr();
    let src_ptr = translate_slice::<u8>(
        memory_mapping,
        src_addr,
        n,
        invoke_context.get_check_aligned(),
    )?
    .as_ptr();

    unsafe { std::ptr::copy(src_ptr, dst_ptr, n as usize) };
    Ok(0)
}
