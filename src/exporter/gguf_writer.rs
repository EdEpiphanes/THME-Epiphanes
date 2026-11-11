//! Pure-Rust GGUF v3 Binary Writer (Milestone v0.1.8.2)
//!
//! Direct serialization of tensors and model metadata into binary GGUF v3
//! containers adhering to the GGML specification with 32-byte alignment.

use anyhow::{bail, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::exporter::streaming_writer::StreamingChunkWriter;
use crate::ir::tensor::TensorSource;
use crate::parser::gguf::{GgmlType, GgufValue};

#[derive(Debug, Clone)]
enum TensorPayloadSource {
    Copy(TensorSource),
    Convert {
        source: TensorSource,
        source_dtype: String,
    },
}

/// Single tensor descriptor holding name, shape, GGML type, and raw byte payload
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: GgmlType,
    pub data: Vec<u8>,
}

impl GgufTensor {
    /// Creates a new tensor descriptor
    pub fn new(
        name: impl Into<String>,
        shape: Vec<usize>,
        ggml_type: GgmlType,
        data: Vec<u8>,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            ggml_type,
            data,
        }
    }
}

/// Pure-Rust GGUF v3 Binary Writer
#[derive(Debug, Clone)]
pub struct GgufWriter {
    alignment: usize,
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<GgufTensor>,
    tensor_sources: Vec<Option<TensorPayloadSource>>,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    /// Creates a new GGUF v3 writer with default 32-byte alignment
    pub fn new() -> Self {
        Self {
            alignment: 32,
            metadata: Vec::new(),
            tensors: Vec::new(),
            tensor_sources: Vec::new(),
        }
    }

    /// Sets the byte alignment for the tensor data section (default: 32)
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.set_alignment(alignment);
        self
    }

    /// Mutably updates alignment and synchronizes `general.alignment` metadata
    pub fn set_alignment(&mut self, alignment: usize) -> &mut Self {
        self.alignment = alignment.max(1);
        if self.alignment != 32 {
            self.add_metadata(
                "general.alignment",
                GgufValue::Uint32(self.alignment as u32),
            );
        }
        self
    }

    /// Configured alignment boundary
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Current metadata entries
    pub fn metadata(&self) -> &[(String, GgufValue)] {
        &self.metadata
    }

    /// Current registered tensors
    pub fn tensors(&self) -> &[GgufTensor] {
        &self.tensors
    }

    /// Appends or updates a metadata key-value pair
    pub fn add_metadata(&mut self, key: impl Into<String>, val: GgufValue) -> &mut Self {
        let key_str = key.into();
        if key_str == "general.alignment" {
            if let Some(align) = val.as_u64() {
                self.alignment = (align as usize).max(1);
            }
        }
        if let Some(entry) = self.metadata.iter_mut().find(|(k, _)| *k == key_str) {
            entry.1 = val;
        } else {
            self.metadata.push((key_str, val));
        }
        self
    }

    /// Convenience helper: adds string metadata
    pub fn add_string(&mut self, key: impl Into<String>, val: impl Into<String>) -> &mut Self {
        self.add_metadata(key, GgufValue::String(val.into()))
    }

    /// Convenience helper: adds u32 metadata
    pub fn add_u32(&mut self, key: impl Into<String>, val: u32) -> &mut Self {
        self.add_metadata(key, GgufValue::Uint32(val))
    }

    /// Convenience helper: adds u64 metadata
    pub fn add_u64(&mut self, key: impl Into<String>, val: u64) -> &mut Self {
        self.add_metadata(key, GgufValue::Uint64(val))
    }

    /// Convenience helper: adds i32 metadata
    pub fn add_i32(&mut self, key: impl Into<String>, val: i32) -> &mut Self {
        self.add_metadata(key, GgufValue::Int32(val))
    }

    /// Convenience helper: adds f32 metadata
    pub fn add_f32(&mut self, key: impl Into<String>, val: f32) -> &mut Self {
        self.add_metadata(key, GgufValue::Float32(val))
    }

    /// Convenience helper: adds bool metadata
    pub fn add_bool(&mut self, key: impl Into<String>, val: bool) -> &mut Self {
        self.add_metadata(key, GgufValue::Bool(val))
    }

    /// Appends a tensor to be written
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        ggml_type: GgmlType,
        data: Vec<u8>,
    ) -> &mut Self {
        self.tensors
            .push(GgufTensor::new(name, shape, ggml_type, data));
        self.tensor_sources.push(None);
        self
    }

    /// Registers a tensor whose unchanged payload will be copied from a source file in chunks.
    pub fn add_tensor_source(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        ggml_type: GgmlType,
        source: TensorSource,
    ) -> &mut Self {
        self.tensors
            .push(GgufTensor::new(name, shape, ggml_type, Vec::new()));
        self.tensor_sources
            .push(Some(TensorPayloadSource::Copy(source)));
        self
    }

    /// Registers a float tensor source to be converted to Q8_0 or TQ1_0 while streaming.
    pub fn add_converted_tensor_source(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        ggml_type: GgmlType,
        source: TensorSource,
        source_dtype: impl Into<String>,
    ) -> &mut Self {
        self.tensors
            .push(GgufTensor::new(name, shape, ggml_type, Vec::new()));
        self.tensor_sources.push(Some(TensorPayloadSource::Convert {
            source,
            source_dtype: source_dtype.into(),
        }));
        self
    }

    /// Appends an existing `GgufTensor` descriptor
    pub fn add_tensor_entry(&mut self, tensor: GgufTensor) -> &mut Self {
        self.tensors.push(tensor);
        self.tensor_sources.push(None);
        self
    }

    /// Serializes entire GGUF v3 stream to any `Write` destination
    pub fn write<W: Write>(&self, mut writer: W) -> Result<usize> {
        let mut payload_lengths = Vec::with_capacity(self.tensors.len());
        for (index, tensor) in self.tensors.iter().enumerate() {
            let source = self
                .tensor_sources
                .get(index)
                .ok_or_else(|| anyhow::anyhow!("GGUF tensor source table is inconsistent"))?;
            let payload_len = match source {
                Some(TensorPayloadSource::Copy(source)) => usize::try_from(source.byte_len)
                    .map_err(|_| anyhow::anyhow!("GGUF tensor payload exceeds addressable size"))?,
                Some(TensorPayloadSource::Convert { .. }) => {
                    let elements = tensor
                        .shape
                        .iter()
                        .try_fold(1usize, |n, d| n.checked_mul(*d))
                        .ok_or_else(|| anyhow::anyhow!("GGUF tensor dimensions overflow"))?;
                    let row_width = tensor.shape.last().copied().unwrap_or(elements);
                    if !row_width.is_multiple_of(tensor.ggml_type.block_size()) {
                        bail!("GGUF tensor '{}' row width {} must be divisible by block size {} for {}", tensor.name, row_width, tensor.ggml_type.block_size(), tensor.ggml_type.type_name());
                    }
                    if !elements.is_multiple_of(tensor.ggml_type.block_size()) {
                        bail!("GGUF tensor '{}' element count is not divisible by conversion block size", tensor.name);
                    }
                    (elements / tensor.ggml_type.block_size())
                        .checked_mul(tensor.ggml_type.type_size())
                        .ok_or_else(|| anyhow::anyhow!("GGUF converted tensor size overflows"))?
                }
                None => tensor.data.len(),
            };
            validate_tensor(tensor, payload_len)?;
            payload_lengths.push(payload_len);
        }
        for (k, _) in &self.metadata {
            if k.is_empty() {
                bail!("GGUF metadata key cannot be empty");
            }
        }

        let alignment = self.alignment.max(1);
        let mut tensor_offsets = Vec::with_capacity(self.tensors.len());
        let mut current_offset: u64 = 0;
        for &payload_len in &payload_lengths {
            tensor_offsets.push(current_offset);
            let payload_len = u64::try_from(payload_len)?;
            let alignment = u64::try_from(alignment)?;
            let pad = (alignment - (payload_len % alignment)) % alignment;
            current_offset = current_offset
                .checked_add(payload_len)
                .and_then(|offset| offset.checked_add(pad))
                .ok_or_else(|| anyhow::anyhow!("GGUF tensor payload offsets overflow"))?;
        }

        let mut total_written: usize = 0;

        // Magic "GGUF" (0x46554747 LE)
        writer.write_all(b"GGUF")?;
        total_written += 4;

        // Version 3
        writer.write_all(&3u32.to_le_bytes())?;
        total_written += 4;

        // Tensor count
        let tensor_count = self.tensors.len() as u64;
        writer.write_all(&tensor_count.to_le_bytes())?;
        total_written += 8;

        // Metadata KV count
        let metadata_kv_count = self.metadata.len() as u64;
        writer.write_all(&metadata_kv_count.to_le_bytes())?;
        total_written += 8;

        // Metadata key-value records
        for (key, val) in &self.metadata {
            total_written += write_metadata_entry(&mut writer, key, val)?;
        }

        // Tensor info headers
        for (i, tensor) in self.tensors.iter().enumerate() {
            total_written += write_string(&mut writer, &tensor.name)?;

            let n_dims = tensor.shape.len() as u32;
            writer.write_all(&n_dims.to_le_bytes())?;
            total_written += 4;

            // GGML column-major dimension ordering: ne[0] (fastest) to ne[n-1] (slowest)
            for &dim in tensor.shape.iter().rev() {
                writer.write_all(&(dim as u64).to_le_bytes())?;
                total_written += 8;
            }

            let type_id = ggml_type_to_u32(tensor.ggml_type);
            writer.write_all(&type_id.to_le_bytes())?;
            total_written += 4;

            let offset = tensor_offsets[i];
            writer.write_all(&offset.to_le_bytes())?;
            total_written += 8;
        }

        // Header alignment padding
        let header_end = total_written;
        let header_padding = (alignment - (header_end % alignment)) % alignment;
        if header_padding > 0 {
            write_zeros(&mut writer, header_padding)?;
            total_written += header_padding;
        }

        // Tensor payloads and per-tensor alignment padding
        for ((tensor, source), &payload_len) in self
            .tensors
            .iter()
            .zip(&self.tensor_sources)
            .zip(&payload_lengths)
        {
            if let Some(TensorPayloadSource::Copy(source)) = source {
                let mut file = File::open(&source.path)
                    .map_err(|_| anyhow::anyhow!("Failed to open GGUF tensor source payload"))?;
                file.seek(SeekFrom::Start(source.byte_offset))
                    .map_err(|_| anyhow::anyhow!("Failed to seek to GGUF tensor source payload"))?;
                let mut chunk_writer = StreamingChunkWriter::new(&mut writer);
                total_written +=
                    usize::try_from(chunk_writer.write_reader(&mut file, source.byte_len)?)?;
            } else if let Some(TensorPayloadSource::Convert {
                source,
                source_dtype,
            }) = source
            {
                let mut file = File::open(&source.path)
                    .map_err(|_| anyhow::anyhow!("Failed to open GGUF tensor source payload"))?;
                file.seek(SeekFrom::Start(source.byte_offset))
                    .map_err(|_| anyhow::anyhow!("Failed to seek to GGUF tensor source payload"))?;
                let mut limited = file.take(source.byte_len);
                let elements = tensor
                    .shape
                    .iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d))
                    .ok_or_else(|| anyhow::anyhow!("GGUF tensor dimensions overflow"))?;
                let written = crate::exporter::conversion::encode_float_source(
                    &mut limited,
                    &mut writer,
                    source_dtype,
                    tensor.ggml_type,
                    elements,
                )?;
                if written != payload_len as u64 {
                    bail!("Converted GGUF tensor payload length mismatch");
                }
                if limited.limit() != 0 {
                    bail!("GGUF tensor source payload ended before recorded length");
                }
                total_written += usize::try_from(written)?;
            } else {
                writer.write_all(&tensor.data)?;
                total_written += tensor.data.len();
            }

            let pad = (alignment - (payload_len % alignment)) % alignment;
            if pad > 0 {
                write_zeros(&mut writer, pad)?;
                total_written += pad;
            }
        }

        writer.flush()?;
        Ok(total_written)
    }

    /// Serializes entire GGUF v3 stream to an in-memory byte buffer
    pub fn write_to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.write(&mut buf)?;
        Ok(buf)
    }

    /// Serializes entire GGUF v3 file to disk using a buffered writer
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let file = std::fs::File::create(path)?;
        let mut buf_writer = std::io::BufWriter::new(file);
        let bytes = self.write(&mut buf_writer)?;
        buf_writer.flush()?;
        Ok(bytes)
    }
}

/// Converts `GgmlType` to its canonical GGUF u32 discriminant
pub fn ggml_type_to_u32(gtype: GgmlType) -> u32 {
    match gtype {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Q4_0 => 2,
        GgmlType::Q4_1 => 3,
        GgmlType::Q5_0 => 6,
        GgmlType::Q5_1 => 7,
        GgmlType::Q8_0 => 8,
        GgmlType::Q8_1 => 9,
        GgmlType::Q2_K => 10,
        GgmlType::Q3_K => 11,
        GgmlType::Q4_K => 12,
        GgmlType::Q5_K => 13,
        GgmlType::Q6_K => 14,
        GgmlType::Q8_K => 15,
        GgmlType::IQ2_XXS => 16,
        GgmlType::IQ2_XS => 17,
        GgmlType::IQ3_XXS => 18,
        GgmlType::IQ1_S => 19,
        GgmlType::IQ4_NL => 20,
        GgmlType::IQ3_S => 21,
        GgmlType::IQ2_S => 22,
        GgmlType::IQ4_XS => 23,
        GgmlType::I8 => 24,
        GgmlType::I16 => 25,
        GgmlType::I32 => 26,
        GgmlType::I64 => 27,
        GgmlType::F64 => 28,
        GgmlType::IQ1_M => 29,
        GgmlType::BF16 => 30,
        GgmlType::TQ1_0 => 34,
        GgmlType::Unknown(val) => val,
    }
}

/// Returns the GGUF metadata type ID discriminant (0..=12)
pub fn gguf_value_type_id(val: &GgufValue) -> u32 {
    match val {
        GgufValue::Uint8(_) => 0,
        GgufValue::Int8(_) => 1,
        GgufValue::Uint16(_) => 2,
        GgufValue::Int16(_) => 3,
        GgufValue::Uint32(_) => 4,
        GgufValue::Int32(_) => 5,
        GgufValue::Float32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::Uint64(_) => 10,
        GgufValue::Int64(_) => 11,
        GgufValue::Float64(_) => 12,
    }
}

/// Validates tensor name, dimensions, and expected payload byte size
fn validate_tensor(tensor: &GgufTensor, payload_len: usize) -> Result<()> {
    if tensor.name.is_empty() {
        bail!("GGUF tensor name cannot be empty");
    }
    if tensor.shape.len() > 8 {
        bail!(
            "GGUF tensor '{}' dimension count {} exceeds maximum supported (8)",
            tensor.name,
            tensor.shape.len()
        );
    }
    if let GgmlType::Unknown(_) = tensor.ggml_type {
        return Ok(());
    }

    let numel: usize = tensor
        .shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| anyhow::anyhow!("GGUF tensor '{}' dimensions overflow", tensor.name))?;
    let block_size = tensor.ggml_type.block_size();
    let type_size = tensor.ggml_type.type_size();

    if !numel.is_multiple_of(block_size) {
        bail!(
            "GGUF tensor '{}' element count {} is not divisible by block size {} for {}",
            tensor.name,
            numel,
            block_size,
            tensor.ggml_type.type_name()
        );
    }

    let expected_bytes = (numel / block_size)
        .checked_mul(type_size)
        .ok_or_else(|| anyhow::anyhow!("GGUF tensor '{}' payload size overflows", tensor.name))?;
    if payload_len != expected_bytes {
        bail!(
            "GGUF tensor '{}' payload size mismatch: shape {:?} with {} requires {} bytes, but {} bytes were provided",
            tensor.name,
            tensor.shape,
            tensor.ggml_type.type_name(),
            expected_bytes,
            payload_len
        );
    }

    Ok(())
}

fn write_string<W: Write>(w: &mut W, s: &str) -> Result<usize> {
    let bytes = s.as_bytes();
    let len = bytes.len() as u64;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(8 + bytes.len())
}

fn write_metadata_entry<W: Write>(w: &mut W, key: &str, val: &GgufValue) -> Result<usize> {
    let mut written = write_string(w, key)?;
    let type_id = gguf_value_type_id(val);
    w.write_all(&type_id.to_le_bytes())?;
    written += 4;
    written += write_value(w, val)?;
    Ok(written)
}

fn write_value<W: Write>(w: &mut W, val: &GgufValue) -> Result<usize> {
    match val {
        GgufValue::Uint8(v) => {
            w.write_all(&[*v])?;
            Ok(1)
        }
        GgufValue::Int8(v) => {
            w.write_all(&v.to_le_bytes())?;
            Ok(1)
        }
        GgufValue::Uint16(v) => {
            w.write_all(&v.to_le_bytes())?;
            Ok(2)
        }
        GgufValue::Int16(v) => {
            w.write_all(&v.to_le_bytes())?;
            Ok(2)
        }
        GgufValue::Uint32(v) => {
            w.write_all(&v.to_le_bytes())?;
            Ok(4)
        }
        GgufValue::Int32(v) => {
            w.write_all(&v.to_le_bytes())?;
            Ok(4)
        }
        GgufValue::Float32(v) => {
            w.write_all(&v.to_le_bytes())?;
            Ok(4)
        }
        GgufValue::Bool(v) => {
            let b = if *v { 1u8 } else { 0u8 };
            w.write_all(&[b])?;
            Ok(1)
        }
        GgufValue::String(s) => write_string(w, s),
        GgufValue::Array(arr) => {
            let count = arr.len() as u64;
            let elem_type = if let Some(first) = arr.first() {
                gguf_value_type_id(first)
            } else {
                8u32 // Default element type: STRING
            };
            w.write_all(&elem_type.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
            let mut written = 12;
            for elem in arr {
                let elem_id = gguf_value_type_id(elem);
                if elem_id != elem_type {
                    bail!(
                        "GGUF array elements must be homogeneous; expected type {}, found {}",
                        elem_type,
                        elem_id
                    );
                }
                written += write_value(w, elem)?;
            }
            Ok(written)
        }
        GgufValue::Uint64(v) => {
            w.write_all(&v.to_le_bytes())?;
            Ok(8)
        }
        GgufValue::Int64(v) => {
            w.write_all(&v.to_le_bytes())?;
            Ok(8)
        }
        GgufValue::Float64(v) => {
            w.write_all(&v.to_le_bytes())?;
            Ok(8)
        }
    }
}

fn write_zeros<W: Write>(w: &mut W, count: usize) -> Result<()> {
    const ZEROS: [u8; 128] = [0u8; 128];
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(ZEROS.len());
        w.write_all(&ZEROS[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::gguf::{GgufParser, GgufReader};

    fn assert_gguf_val_eq(a: &GgufValue, b: &GgufValue) {
        match (a, b) {
            (GgufValue::Uint8(x), GgufValue::Uint8(y)) => assert_eq!(x, y),
            (GgufValue::Int8(x), GgufValue::Int8(y)) => assert_eq!(x, y),
            (GgufValue::Uint16(x), GgufValue::Uint16(y)) => assert_eq!(x, y),
            (GgufValue::Int16(x), GgufValue::Int16(y)) => assert_eq!(x, y),
            (GgufValue::Uint32(x), GgufValue::Uint32(y)) => assert_eq!(x, y),
            (GgufValue::Int32(x), GgufValue::Int32(y)) => assert_eq!(x, y),
            (GgufValue::Float32(x), GgufValue::Float32(y)) => {
                assert!((x - y).abs() < 1e-6, "{x} vs {y}");
            }
            (GgufValue::Bool(x), GgufValue::Bool(y)) => assert_eq!(x, y),
            (GgufValue::String(x), GgufValue::String(y)) => assert_eq!(x, y),
            (GgufValue::Array(x), GgufValue::Array(y)) => {
                assert_eq!(x.len(), y.len());
                for (xi, yi) in x.iter().zip(y.iter()) {
                    assert_gguf_val_eq(xi, yi);
                }
            }
            (GgufValue::Uint64(x), GgufValue::Uint64(y)) => assert_eq!(x, y),
            (GgufValue::Int64(x), GgufValue::Int64(y)) => assert_eq!(x, y),
            (GgufValue::Float64(x), GgufValue::Float64(y)) => {
                assert!((x - y).abs() < 1e-12, "{x} vs {y}");
            }
            _ => panic!("Type mismatch: {:?} vs {:?}", a, b),
        }
    }

    #[test]
    fn test_gguf_writer_header_magic_and_version() {
        let writer = GgufWriter::new();
        let bytes = writer.write_to_bytes().expect("Serialization failed");

        // Header must be exactly 32 bytes (24 bytes header + 8 bytes padding)
        assert_eq!(bytes.len(), 32);
        assert_eq!(&bytes[0..4], b"GGUF");
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            3
        );
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0);
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0);
        assert_eq!(&bytes[24..32], &[0u8; 8]);

        // Validate GgufParser accepts minimal header
        let model = GgufParser::parse_from_bytes(&bytes, Path::new("minimal.gguf"))
            .expect("Parser should parse empty model");
        assert_eq!(model.tensors.len(), 0);
    }

    #[test]
    fn test_gguf_writer_metadata_roundtrip_all_types() {
        let mut writer = GgufWriter::new();
        writer
            .add_metadata("m.u8", GgufValue::Uint8(250))
            .add_metadata("m.i8", GgufValue::Int8(-120))
            .add_metadata("m.u16", GgufValue::Uint16(65000))
            .add_metadata("m.i16", GgufValue::Int16(-32000))
            .add_metadata("m.u32", GgufValue::Uint32(1_000_000))
            .add_metadata("m.i32", GgufValue::Int32(-500_000))
            .add_metadata("m.f32", GgufValue::Float32(std::f32::consts::PI))
            .add_metadata("m.bool_t", GgufValue::Bool(true))
            .add_metadata("m.bool_f", GgufValue::Bool(false))
            .add_metadata("m.str", GgufValue::String("pure_rust_gguf_v3".into()))
            .add_metadata(
                "m.arr_u32",
                GgufValue::Array(vec![
                    GgufValue::Uint32(10),
                    GgufValue::Uint32(20),
                    GgufValue::Uint32(30),
                ]),
            )
            .add_metadata(
                "m.arr_str",
                GgufValue::Array(vec![
                    GgufValue::String("alpha".into()),
                    GgufValue::String("beta".into()),
                ]),
            )
            .add_metadata("m.u64", GgufValue::Uint64(123456789012345))
            .add_metadata("m.i64", GgufValue::Int64(-98765432109876))
            .add_metadata("m.f64", GgufValue::Float64(std::f64::consts::E));

        let bytes = writer.write_to_bytes().expect("Failed to serialize");

        // Verify with GgufReader
        let mut reader = GgufReader::new(&bytes);
        assert_eq!(reader.read_bytes(4).unwrap(), b"GGUF");
        assert_eq!(reader.read_u32().unwrap(), 3);
        assert_eq!(reader.read_u64().unwrap(), 0);
        let kv_count = reader.read_u64().unwrap() as usize;
        assert_eq!(kv_count, writer.metadata.len());

        for (expected_key, expected_val) in &writer.metadata {
            let key = reader.read_string().unwrap();
            assert_eq!(&key, expected_key);
            let val_type = reader.read_u32().unwrap();
            assert_eq!(val_type, gguf_value_type_id(expected_val));
            let parsed_val = reader.read_value(val_type).unwrap();
            assert_gguf_val_eq(&parsed_val, expected_val);
        }
    }

    #[test]
    fn test_gguf_writer_shape_dimension_reversal() {
        let mut writer = GgufWriter::new();
        // Shape [128, 64] in row-major
        let f32_payload = vec![0u8; 128 * 64 * 4];
        writer.add_tensor("layer.weight", vec![128, 64], GgmlType::F32, f32_payload);

        let bytes = writer.write_to_bytes().expect("Write failed");

        // Verify raw bytes contain reversed dimensions: 64 then 128
        let mut reader = GgufReader::new(&bytes);
        reader.read_bytes(4).unwrap(); // magic
        reader.read_u32().unwrap(); // version
        assert_eq!(reader.read_u64().unwrap(), 1); // tensor_count
        assert_eq!(reader.read_u64().unwrap(), 0); // metadata_count

        let name = reader.read_string().unwrap();
        assert_eq!(name, "layer.weight");
        let n_dims = reader.read_u32().unwrap();
        assert_eq!(n_dims, 2);
        let dim0 = reader.read_u64().unwrap();
        let dim1 = reader.read_u64().unwrap();
        // On disk: fastest dim first (64), then slowest (128)
        assert_eq!(dim0, 64);
        assert_eq!(dim1, 128);

        // Parser inverts back to row-major [128, 64]
        let model = GgufParser::parse_from_bytes(&bytes, Path::new("rev.gguf")).unwrap();
        assert_eq!(model.tensors[0].shape, vec![128, 64]);
    }

    #[test]
    fn test_gguf_writer_32byte_alignment_padding() {
        let mut writer = GgufWriter::new();
        writer.add_metadata("general.architecture", GgufValue::String("test".into()));

        // Tensors with irregular payload sizes (17 bytes and 35 bytes) using I8
        let data1 = vec![1u8; 17];
        let data2 = vec![2u8; 35];
        writer.add_tensor("t1", vec![17], GgmlType::I8, data1);
        writer.add_tensor("t2", vec![35], GgmlType::I8, data2);

        let bytes = writer.write_to_bytes().expect("Write failed");

        let mut reader = GgufReader::new(&bytes);
        reader.read_bytes(24).unwrap(); // base header
        let _key = reader.read_string().unwrap();
        let val_type = reader.read_u32().unwrap();
        let _val = reader.read_value(val_type).unwrap();

        // Read t1 header
        let _ = reader.read_string().unwrap();
        let _ = reader.read_u32().unwrap();
        let _ = reader.read_u64().unwrap();
        let _ = reader.read_u32().unwrap();
        let offset1 = reader.read_u64().unwrap();

        // Read t2 header
        let _ = reader.read_string().unwrap();
        let _ = reader.read_u32().unwrap();
        let _ = reader.read_u64().unwrap();
        let _ = reader.read_u32().unwrap();
        let offset2 = reader.read_u64().unwrap();

        let header_end = reader.cursor;
        let pad = (32 - (header_end % 32)) % 32;
        let data_start = header_end + pad;

        assert_eq!(data_start % 32, 0);
        assert_eq!(offset1, 0);
        // t1 is 17 bytes, padded to 32 bytes boundary -> offset2 must be 32
        assert_eq!(offset2, 32);
        assert_eq!((data_start + offset1 as usize) % 32, 0);
        assert_eq!((data_start + offset2 as usize) % 32, 0);

        // Verify data contents at calculated offsets
        let t1_start = data_start + offset1 as usize;
        assert_eq!(&bytes[t1_start..t1_start + 17], &[1u8; 17]);
        let t2_start = data_start + offset2 as usize;
        assert_eq!(&bytes[t2_start..t2_start + 35], &[2u8; 35]);
    }

    #[test]
    fn test_gguf_writer_custom_alignment() {
        let mut writer = GgufWriter::new().with_alignment(64);
        assert_eq!(writer.alignment(), 64);
        writer.add_tensor("t_64", vec![64], GgmlType::I8, vec![7u8; 64]);

        let bytes = writer.write_to_bytes().expect("Write failed");

        let mut reader = GgufReader::new(&bytes);
        reader.read_bytes(24).unwrap();
        // Read general.alignment metadata
        let key = reader.read_string().unwrap();
        assert_eq!(key, "general.alignment");
        let vtype = reader.read_u32().unwrap();
        let val = reader.read_value(vtype).unwrap();
        assert_eq!(val.as_u64().unwrap(), 64);

        // Read tensor header
        let _ = reader.read_string().unwrap();
        let _ = reader.read_u32().unwrap();
        let _ = reader.read_u64().unwrap();
        let _ = reader.read_u32().unwrap();
        let offset = reader.read_u64().unwrap();
        assert_eq!(offset, 0);

        let header_end = reader.cursor;
        let pad = (64 - (header_end % 64)) % 64;
        let data_start = header_end + pad;
        assert_eq!(data_start % 64, 0);

        // Verify GgufParser respects custom alignment
        let model = GgufParser::parse_from_bytes(&bytes, Path::new("align64.gguf")).unwrap();
        assert_eq!(model.tensors.len(), 1);
    }

    #[test]
    fn test_gguf_writer_single_f32_tensor_roundtrip() {
        let mut writer = GgufWriter::new();
        writer.add_string("general.architecture", "llama");
        writer.add_u32("llama.block_count", 1);
        writer.add_u32("llama.embedding_length", 4);

        let raw_floats: [f32; 4] = [1.0, -2.5, 3.25, -0.125];
        let mut payload = Vec::new();
        for &f in &raw_floats {
            payload.extend_from_slice(&f.to_le_bytes());
        }

        writer.add_tensor("blk.0.attn_q.weight", vec![4], GgmlType::F32, payload);
        let bytes = writer.write_to_bytes().expect("Failed to write");

        let model = GgufParser::parse_from_bytes(&bytes, Path::new("test_f32.gguf")).unwrap();
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(model.tensors[0].shape, vec![4]);
        assert_eq!(model.tensors[0].original_dtype, "F32");
        assert_eq!(model.tensors[0].byte_size, 16);
    }

    #[test]
    fn test_gguf_writer_invalid_tensor_payload_validation() {
        let mut writer = GgufWriter::new();
        // Shape [2, 2] F32 requires 4 * 4 = 16 bytes, but only 8 bytes provided
        writer.add_tensor("bad_tensor", vec![2, 2], GgmlType::F32, vec![0u8; 8]);

        let result = writer.write_to_bytes();
        assert!(
            result.is_err(),
            "Expected validation error for payload size mismatch"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("payload size mismatch"),
            "Error message should mention mismatch: {err_msg}"
        );
    }
}
