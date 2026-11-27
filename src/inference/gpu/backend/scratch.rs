// SPDX-License-Identifier: AGPL-3.0-only
//! Reusable WGPU scratch buffers and activation-copy metadata for linear dispatches.

use super::{storage_buffer, COOP_DIRECT_KERNEL, PARAM_STRIDE, Q8DOT_KERNEL};

/// Scratch device buffers, grown on demand and reused across calls.
pub(super) struct Scratch {
    pub(super) input: wgpu::Buffer,
    input_capacity: usize,
    pub(super) output: wgpu::Buffer,
    output_capacity: usize,
    pub(super) staging: wgpu::Buffer,
    staging_capacity: usize,
    pub(super) params: wgpu::Buffer,
    params_capacity: usize,
}

/// Shared memory the register-blocked kernel needs: two slices of 16 x 33 vec4.
pub(super) const GEMM_V4_SHARED_BYTES: u32 = 2 * 528 * 16;

/// Which activation copy a kernel reads at binding 1.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in super::super) enum ActivationCopy {
    F16,
    Int8,
}

impl ActivationCopy {
    /// The copy the named kernel needs, if any.
    pub(in super::super) fn for_kernel(name: &str) -> Option<Self> {
        match name {
            COOP_DIRECT_KERNEL => Some(Self::F16),
            Q8DOT_KERNEL => Some(Self::Int8),
            _ => None,
        }
    }
}

impl Scratch {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let (x, y) = (256 << 10, 1 << 20);
        let in_use = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        Self {
            input: storage_buffer(device, "scratch_input", x, in_use),
            input_capacity: x,
            output: storage_buffer(
                device,
                "scratch_output",
                y,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            output_capacity: y,
            staging: storage_buffer(
                device,
                "scratch_staging",
                y,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            staging_capacity: y,
            params: storage_buffer(device, "scratch_params", 16 * PARAM_STRIDE, in_use),
            params_capacity: 16 * PARAM_STRIDE,
        }
    }

    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        x_bytes: usize,
        y_bytes: usize,
        p_bytes: usize,
    ) {
        if x_bytes > self.input_capacity {
            self.input_capacity = x_bytes.next_power_of_two();
            self.input = storage_buffer(
                device,
                "scratch_input",
                self.input_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
        }
        if y_bytes > self.output_capacity {
            self.output_capacity = y_bytes.next_power_of_two();
            self.output = storage_buffer(
                device,
                "scratch_output",
                self.output_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
        }
        if y_bytes > self.staging_capacity {
            self.staging_capacity = y_bytes.next_power_of_two();
            self.staging = storage_buffer(
                device,
                "scratch_staging",
                self.staging_capacity,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
        }
        if p_bytes > self.params_capacity {
            self.params_capacity = p_bytes.next_power_of_two();
            self.params = storage_buffer(
                device,
                "scratch_params",
                self.params_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
        }
    }
}
