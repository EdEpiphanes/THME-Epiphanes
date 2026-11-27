// SPDX-License-Identifier: AGPL-3.0-only
//! GPU timestamp query probe and per-dispatch op timing samples.

use super::storage_buffer;

/// Device time of one dispatch from a profiled decoder pass.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct OpSample {
    /// Kernel name; linears carry their kernel and `rows x in x out` shape.
    pub label: String,
    pub ns: u64,
}

/// Two-slot timestamp query around a compute pass, with buffers to read it back.
pub(in super::super) struct TimestampProbe {
    pub(in super::super) queries: wgpu::QuerySet,
    pub(in super::super) resolve: wgpu::Buffer,
    pub(in super::super) staging: wgpu::Buffer,
    pub(super) period_ns: f32,
}

impl TimestampProbe {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("kernel_time"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: storage_buffer(
                device,
                "kernel_time_resolve",
                16,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: storage_buffer(
                device,
                "kernel_time_staging",
                16,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            period_ns: queue.get_timestamp_period(),
        }
    }
}
