// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::ir::model::ModelMetadataValue;
use crate::ir::tensor::TensorSource;
use crate::ir::{ModelArchitecture, ModelIR, TensorIR};
use crate::profiling::WeightAnalyzer;

/// Comprehensive GGML quantization types and super-block metadata (GGUF v2/v3 specification)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(val: u32) -> Self {
        match val {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            other => Self::Unknown(other),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::TQ1_0 => "TQ1_0",
            Self::Unknown(_) => "QUANT_OTHER",
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64
            | Self::Unknown(_) => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL => 32,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::TQ1_0 => 256,
        }
    }

    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 | Self::Unknown(_) => 1,
            Self::I64 | Self::F64 => 8,
            Self::Q4_0 | Self::IQ4_NL => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2_K => 84,
            Self::Q3_K | Self::IQ3_S => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 48,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 136,
            Self::IQ1_M => 56,
            Self::TQ1_0 => 54,
        }
    }

    pub fn bytes_per_weight(&self) -> f64 {
        self.type_size() as f64 / self.block_size() as f64
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::F64
                | Self::Unknown(_)
        )
    }
}

pub fn ggml_type_str(t: u32) -> &'static str {
    GgmlType::from_u32(t).type_name()
}

pub fn ggml_type_bytes_per_elem(t: u32) -> f64 {
    GgmlType::from_u32(t).bytes_per_weight()
}

/// Zero-allocation block decoders for GGUF quantized formats
pub struct GgufBlockDecoder;

impl GgufBlockDecoder {
    /// Unpacks one standard 256-value TQ1_0 block (54 bytes).
    pub fn unpack_tq1_0(block: &[u8]) -> Result<[f32; 256]> {
        if block.len() < 54 {
            bail!("TQ1_0 block requires 54 bytes, got {}", block.len());
        }
        let scale = crate::profiling::f16_to_f32(u16::from_le_bytes([block[52], block[53]]));
        let mut out = [0.0; 256];
        let powers = [1u16, 3, 9, 27, 81];
        for (j, byte) in block[..48].iter().copied().enumerate() {
            let (base, stride, offset) = if j < 32 {
                (j, 32, 0)
            } else {
                (j - 32, 16, 160)
            };
            for n in 0..5 {
                let wrapped = byte.wrapping_mul(powers[n] as u8);
                let trit = ((u16::from(wrapped) * 3) >> 8) as i16 - 1;
                out[offset + base + n * stride] = trit as f32 * scale;
            }
        }
        for (j, byte) in block[48..52].iter().copied().enumerate() {
            for n in 0..4 {
                let wrapped = byte.wrapping_mul(powers[n] as u8);
                let trit = ((u16::from(wrapped) * 3) >> 8) as i16 - 1;
                out[240 + n * 4 + j] = trit as f32 * scale;
            }
        }
        Ok(out)
    }

    /// Unpacks 32 weights from a Q8_0 block (34 bytes: 2 bytes f16 delta + 32 signed int8s)
    pub fn unpack_q8_0(block: &[u8]) -> Result<[f32; 32]> {
        if block.len() < 34 {
            bail!("Q8_0 block requires 34 bytes, got {}", block.len());
        }
        let scale_u16 = u16::from_le_bytes([block[0], block[1]]);
        let d = crate::profiling::f16_to_f32(scale_u16);
        let mut out = [0.0_f32; 32];
        for i in 0..32 {
            out[i] = d * (block[2 + i] as i8 as f32);
        }
        Ok(out)
    }

    /// Unpacks 32 weights from a Q4_0 block (18 bytes: 2 bytes f16 delta + 16 bytes nibbles with -8 offset)
    pub fn unpack_q4_0(block: &[u8]) -> Result<[f32; 32]> {
        if block.len() < 18 {
            bail!("Q4_0 block requires 18 bytes, got {}", block.len());
        }
        let scale_u16 = u16::from_le_bytes([block[0], block[1]]);
        let d = crate::profiling::f16_to_f32(scale_u16);
        let mut out = [0.0_f32; 32];
        for i in 0..16 {
            let byte = block[2 + i];
            let x0 = (byte & 0x0F) as i8 - 8;
            let x1 = ((byte >> 4) & 0x0F) as i8 - 8;
            out[i] = d * (x0 as f32);
            out[i + 16] = d * (x1 as f32);
        }
        Ok(out)
    }

    /// Unpacks 32 weights from a Q4_1 block (20 bytes: 2 bytes f16 delta + 2 bytes f16 min + 16 bytes nibbles)
    pub fn unpack_q4_1(block: &[u8]) -> Result<[f32; 32]> {
        if block.len() < 20 {
            bail!("Q4_1 block requires 20 bytes, got {}", block.len());
        }
        let d = crate::profiling::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = crate::profiling::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let mut out = [0.0_f32; 32];
        for i in 0..16 {
            let byte = block[4 + i];
            let x0 = (byte & 0x0F) as f32;
            let x1 = ((byte >> 4) & 0x0F) as f32;
            out[i] = d * x0 + m;
            out[i + 16] = d * x1 + m;
        }
        Ok(out)
    }

    /// Unpacks 32 weights from a Q5_0 block (22 bytes: 2 bytes f16 delta + 4 bytes high bits + 16 bytes low bits)
    pub fn unpack_q5_0(block: &[u8]) -> Result<[f32; 32]> {
        if block.len() < 22 {
            bail!("Q5_0 block requires 22 bytes, got {}", block.len());
        }
        let d = crate::profiling::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let mut out = [0.0_f32; 32];
        for i in 0..16 {
            let byte = block[6 + i];
            let h0 = ((qh >> i) & 1) as u8;
            let h1 = ((qh >> (i + 16)) & 1) as u8;
            let x0 = ((byte & 0x0F) | (h0 << 4)) as i8 - 16;
            let x1 = (((byte >> 4) & 0x0F) | (h1 << 4)) as i8 - 16;
            out[i] = d * (x0 as f32);
            out[i + 16] = d * (x1 as f32);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl From<GgufValue> for ModelMetadataValue {
    fn from(value: GgufValue) -> Self {
        match value {
            GgufValue::Uint8(value) => Self::Uint8(value),
            GgufValue::Int8(value) => Self::Int8(value),
            GgufValue::Uint16(value) => Self::Uint16(value),
            GgufValue::Int16(value) => Self::Int16(value),
            GgufValue::Uint32(value) => Self::Uint32(value),
            GgufValue::Int32(value) => Self::Int32(value),
            GgufValue::Float32(value) => Self::Float32(value),
            GgufValue::Bool(value) => Self::Bool(value),
            GgufValue::String(value) => Self::String(value),
            GgufValue::Array(values) => Self::Array(values.into_iter().map(Self::from).collect()),
            GgufValue::Uint64(value) => Self::Uint64(value),
            GgufValue::Int64(value) => Self::Int64(value),
            GgufValue::Float64(value) => Self::Float64(value),
        }
    }
}

impl From<ModelMetadataValue> for GgufValue {
    fn from(value: ModelMetadataValue) -> Self {
        match value {
            ModelMetadataValue::Uint8(value) => Self::Uint8(value),
            ModelMetadataValue::Int8(value) => Self::Int8(value),
            ModelMetadataValue::Uint16(value) => Self::Uint16(value),
            ModelMetadataValue::Int16(value) => Self::Int16(value),
            ModelMetadataValue::Uint32(value) => Self::Uint32(value),
            ModelMetadataValue::Int32(value) => Self::Int32(value),
            ModelMetadataValue::Float32(value) => Self::Float32(value),
            ModelMetadataValue::Bool(value) => Self::Bool(value),
            ModelMetadataValue::String(value) => Self::String(value),
            ModelMetadataValue::Array(values) => {
                Self::Array(values.into_iter().map(Self::from).collect())
            }
            ModelMetadataValue::Uint64(value) => Self::Uint64(value),
            ModelMetadataValue::Int64(value) => Self::Int64(value),
            ModelMetadataValue::Float64(value) => Self::Float64(value),
        }
    }
}

impl GgufValue {
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            Self::Int8(v) if *v >= 0 => Some(*v as u64),
            Self::Int16(v) if *v >= 0 => Some(*v as u64),
            Self::Int32(v) if *v >= 0 => Some(*v as u64),
            Self::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

pub struct GgufReader<'a> {
    data: &'a [u8],
    pub cursor: usize,
}

impl<'a> GgufReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, cursor: 0 }
    }

    pub fn remaining_bytes(&self) -> usize {
        self.data.len().saturating_sub(self.cursor)
    }

    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.cursor.saturating_add(n) > self.data.len() {
            bail!(
                "Unexpected EOF in GGUF stream: requested {} bytes at offset {} (total buffer {})",
                n,
                self.cursor,
                self.data.len()
            );
        }
        let slice = &self.data[self.cursor..self.cursor + n];
        self.cursor += n;
        Ok(slice)
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    pub fn read_i8(&mut self) -> Result<i8> {
        let b = self.read_bytes(1)?;
        Ok(b[0] as i8)
    }

    pub fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn read_i16(&mut self) -> Result<i16> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_i64(&mut self) -> Result<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_f64(&mut self) -> Result<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        // Strict boundary: protect against allocation overflow if string size is corrupt
        if len > self.remaining_bytes() {
            bail!(
                "Malformed string length {} exceeds remaining GGUF buffer size {}",
                len,
                self.remaining_bytes()
            );
        }
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    pub fn skip_value(&mut self, val_type: u32) -> Result<()> {
        match val_type {
            0 | 1 | 7 => {
                self.read_bytes(1)?;
                Ok(())
            }
            2 | 3 => {
                self.read_bytes(2)?;
                Ok(())
            }
            4..=6 => {
                self.read_bytes(4)?;
                Ok(())
            }
            10..=12 => {
                self.read_bytes(8)?;
                Ok(())
            }
            8 => {
                let len = self.read_u64()? as usize;
                if len > self.remaining_bytes() {
                    bail!("Malformed string length in skip_value");
                }
                self.read_bytes(len)?;
                Ok(())
            }
            _ => {
                let _ = self.read_value(val_type)?;
                Ok(())
            }
        }
    }

    pub fn read_value(&mut self, val_type: u32) -> Result<GgufValue> {
        self.read_value_internal(val_type, 1024, false)
    }

    pub fn read_value_with_array_limit(
        &mut self,
        val_type: u32,
        max_array_items: usize,
    ) -> Result<GgufValue> {
        self.read_value_internal(val_type, max_array_items, true)
    }

    fn read_value_internal(
        &mut self,
        val_type: u32,
        max_array_items: usize,
        reject_oversized_arrays: bool,
    ) -> Result<GgufValue> {
        match val_type {
            0 => Ok(GgufValue::Uint8(self.read_u8()?)),
            1 => Ok(GgufValue::Int8(self.read_i8()?)),
            2 => Ok(GgufValue::Uint16(self.read_u16()?)),
            3 => Ok(GgufValue::Int16(self.read_i16()?)),
            4 => Ok(GgufValue::Uint32(self.read_u32()?)),
            5 => Ok(GgufValue::Int32(self.read_i32()?)),
            6 => Ok(GgufValue::Float32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let elem_type = self.read_u32()?;
                let count = usize::try_from(self.read_u64()?)
                    .context("GGUF array count exceeds platform limits")?;
                // Safety bound on array size to prevent OOM on corrupt streams
                if reject_oversized_arrays && count > max_array_items {
                    bail!(
                        "GGUF array has {} elements, exceeding configured limit {}",
                        count,
                        max_array_items
                    );
                }
                let max_collect = count.min(max_array_items);
                let mut arr = Vec::new();
                arr.try_reserve_exact(max_collect)
                    .context("Unable to allocate GGUF metadata array")?;
                for i in 0..count {
                    if i < max_collect {
                        let v = self.read_value(elem_type)?;
                        arr.push(v);
                    } else {
                        self.skip_value(elem_type)?;
                    }
                }
                Ok(GgufValue::Array(arr))
            }
            10 => Ok(GgufValue::Uint64(self.read_u64()?)),
            11 => Ok(GgufValue::Int64(self.read_i64()?)),
            12 => Ok(GgufValue::Float64(self.read_f64()?)),
            _ => bail!("Unsupported GGUF value type ID: {}", val_type),
        }
    }
}

pub struct RawTensorHeader {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub struct GgufParser;

impl GgufParser {
    pub fn parse_from_file<P: AsRef<Path>>(path: P) -> Result<ModelIR> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open GGUF file: {:?}", path.as_ref()))?;

        // SAFETY: File descriptor is open and unmodified during the lifetime of this read-only memory map.
        let mmap = unsafe { Mmap::map(&file)? };
        let source_path = path
            .as_ref()
            .canonicalize()
            .with_context(|| format!("Failed to resolve GGUF source path: {:?}", path.as_ref()))?;

        Self::parse_from_bytes_with_source(&mmap, path.as_ref(), Some(&source_path))
    }

    pub fn parse_from_bytes(data: &[u8], path: &Path) -> Result<ModelIR> {
        Self::parse_from_bytes_with_source(data, path, None)
    }

    fn parse_from_bytes_with_source(
        data: &[u8],
        path: &Path,
        source_path: Option<&Path>,
    ) -> Result<ModelIR> {
        if data.len() < 24 {
            bail!(
                "GGUF buffer truncated: {} bytes provided (minimum header requires 24 bytes)",
                data.len()
            );
        }

        let mut reader = GgufReader::new(data);

        // Header check: 4 bytes "GGUF"
        let magic = reader.read_bytes(4)?;
        if magic != b"GGUF" {
            bail!(
                "Invalid GGUF magic bytes: expected b'GGUF', got {:?}",
                magic
            );
        }

        let version = reader.read_u32()?;
        if version != 2 && version != 3 {
            bail!(
                "Unsupported GGUF version {}: only v2 and v3 are supported",
                version
            );
        }

        let tensor_count = reader.read_u64()? as usize;
        let metadata_kv_count = reader.read_u64()? as usize;

        // Prevent DoS on malformed kv_count
        let max_possible_kvs = data.len() / 8;
        if metadata_kv_count > max_possible_kvs {
            bail!(
                "Corrupt GGUF metadata count {} exceeds physical buffer bound {}",
                metadata_kv_count,
                max_possible_kvs
            );
        }

        // Metadata KV pairs
        let mut metadata: HashMap<String, GgufValue> =
            HashMap::with_capacity(metadata_kv_count.min(2048));
        for _ in 0..metadata_kv_count {
            let key = reader.read_string()?;
            let val_type = reader.read_u32()?;
            const MAX_METADATA_ARRAY_ITEMS: usize = 1_000_000;
            let val = reader.read_value_with_array_limit(val_type, MAX_METADATA_ARRAY_ITEMS)?;
            metadata.insert(key, val);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u64)
            .unwrap_or(32)
            .max(1) as usize;

        // Resilient Tensor Header Reading (recovers intact tensors if stream is truncated)
        let mut raw_tensors = Vec::with_capacity(tensor_count.min(2048));
        for i in 0..tensor_count {
            match Self::read_single_tensor_header(&mut reader) {
                Ok(raw) => raw_tensors.push(raw),
                Err(e) => {
                    // Log truncation diagnostic and stop header reading safely
                    eprintln!(
                        "GGUF stream truncated at tensor index {}/{}: {}. Preserving {} intact tensors.",
                        i, tensor_count, e, raw_tensors.len()
                    );
                    break;
                }
            }
        }

        // Tensor data alignment boundary
        let header_end = reader.cursor;
        let padding = (alignment - (header_end % alignment)) % alignment;
        let data_start = header_end + padding;

        let mut tensors = Vec::with_capacity(raw_tensors.len());
        for raw in raw_tensors {
            let gtype = GgmlType::from_u32(raw.ggml_type);
            let dtype = gtype.type_name().to_string();
            let numel = raw
                .shape
                .iter()
                .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                .with_context(|| format!("GGUF tensor '{}' dimensions overflow", raw.name))?;
            let byte_size = numel
                .div_ceil(gtype.block_size())
                .checked_mul(gtype.type_size())
                .with_context(|| format!("GGUF tensor '{}' byte size overflows", raw.name))?;

            let mut tensor_ir = TensorIR::new(raw.name, raw.shape, dtype.clone(), byte_size);

            // Extract weight profile if data is within bounds
            let tensor_byte_offset = usize::try_from(raw.offset)
                .ok()
                .and_then(|offset| data_start.checked_add(offset));
            let tensor_range = tensor_byte_offset.and_then(|start| {
                start
                    .checked_add(byte_size)
                    .filter(|&end| end <= data.len())
                    .map(|end| start..end)
            });
            if let Some(range) = tensor_range {
                let slice = &data[range.clone()];
                tensor_ir.weight_profile = Some(WeightAnalyzer::profile_tensor(slice, &dtype));
                if let Some(source_path) = source_path {
                    tensor_ir.source = Some(TensorSource {
                        path: source_path.to_path_buf(),
                        byte_offset: u64::try_from(range.start)
                            .context("GGUF tensor source offset exceeds u64")?,
                        byte_len: u64::try_from(range.len())
                            .context("GGUF tensor source length exceeds u64")?,
                    });
                }
            } else {
                tensor_ir.weight_profile = Some(crate::ir::WeightProfile::default());
            }

            tensors.push(tensor_ir);
        }

        // Extract model attributes from metadata
        let arch_str = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");
        let architecture = ModelArchitecture::from_str_name(arch_str);

        let num_layers = metadata
            .get(&format!("{}.block_count", arch_str))
            .or_else(|| metadata.get(&format!("{}.layer_count", arch_str)))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                let mut max_l = 0;
                for t in &tensors {
                    if let Some(l) = t.layer_index {
                        if l >= max_l {
                            max_l = l + 1;
                        }
                    }
                }
                max_l.max(1)
            });

        let hidden_size = metadata
            .get(&format!("{}.embedding_length", arch_str))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or(4096);

        let num_attention_heads = metadata
            .get(&format!("{}.attention.head_count", arch_str))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or(32);

        let num_key_value_heads = metadata
            .get(&format!("{}.attention.head_count_kv", arch_str))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let max_position_embeddings = metadata
            .get(&format!("{}.context_length", arch_str))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or(4096);

        let vocab_size = metadata
            .get(&format!("{}.vocab_size", arch_str))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or(32000);

        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("gguf_model")
            .to_string();

        let source_gguf_metadata = metadata
            .iter()
            .map(|(key, value)| (key.clone(), ModelMetadataValue::from(value.clone())))
            .collect();
        let mut model_ir = ModelIR::new(
            model_name,
            architecture,
            num_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            vocab_size,
            max_position_embeddings,
            tensors,
        );
        model_ir.source_gguf_metadata = source_gguf_metadata;
        let float = |key: String| metadata.get(&key).and_then(GgufValue::as_f32);
        model_ir.norm_eps = float(format!("{arch_str}.attention.layer_norm_rms_epsilon"))
            .or_else(|| float(format!("{arch_str}.attention.layer_norm_epsilon")));
        model_ir.rope_theta = float(format!("{arch_str}.rope.freq_base"));

        Ok(model_ir)
    }

    fn read_single_tensor_header(reader: &mut GgufReader) -> Result<RawTensorHeader> {
        let name = reader.read_string()?;
        let n_dims = reader.read_u32()? as usize;
        if n_dims > 8 {
            bail!("Suspicious GGUF tensor dimensions {} > 8", n_dims);
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(reader.read_u64()? as usize);
        }
        dims.reverse();
        let ggml_type = reader.read_u32()?;
        let offset = reader.read_u64()?;

        Ok(RawTensorHeader {
            name,
            shape: dims,
            ggml_type,
            offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_synthetic_parse() {
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(b"GGUF");
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata KV count = 1
        buf.extend_from_slice(&1u64.to_le_bytes());

        // Metadata: key="general.architecture", type=8 (String), val="llama"
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // type String
        let val = "llama";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());

        // Tensor 0: name="blk.0.attn_q.weight", n_dims=2, dims=[128, 64], type=0 (F32), offset=0
        let t_name = "blk.0.attn_q.weight";
        buf.extend_from_slice(&(t_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(t_name.as_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
        buf.extend_from_slice(&64u64.to_le_bytes());
        buf.extend_from_slice(&128u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        let res = GgufParser::parse_from_bytes(&buf, Path::new("dummy.gguf"));
        assert!(res.is_ok());
        let ir = res.unwrap();
        assert_eq!(ir.architecture, ModelArchitecture::Llama);
        assert_eq!(ir.tensors.len(), 1);
        assert_eq!(ir.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(ir.tensors[0].shape, vec![128, 64]);
        assert!(ir.tensors[0].source.is_none());
    }

    #[test]
    fn test_gguf_file_parse_retains_tensor_payload_range() {
        use crate::exporter::gguf_writer::GgufWriter;
        use std::io::{Read, Seek, SeekFrom};

        let payload = [1.0f32, -2.0, 3.5, 0.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let q8_payload = vec![0u8; 34];
        let mut writer = GgufWriter::new();
        writer.add_tensor("weights", vec![4], GgmlType::F32, payload.clone());
        writer.add_tensor("quantized", vec![32], GgmlType::Q8_0, q8_payload.clone());

        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("source.gguf");
        writer.write_to_file(&path).expect("write GGUF");

        let model = GgufParser::parse_from_file(&path).expect("parse GGUF file");
        let source = model.tensors[0].source.as_ref().expect("source range");
        assert_eq!(source.path, path.canonicalize().expect("canonical path"));
        assert_eq!(source.byte_len, payload.len() as u64);
        let quantized_source = model.tensors[1]
            .source
            .as_ref()
            .expect("quantized source range");
        assert_eq!(quantized_source.byte_len, q8_payload.len() as u64);

        let mut file = File::open(&source.path).expect("open source");
        file.seek(SeekFrom::Start(source.byte_offset))
            .expect("seek to payload");
        let mut actual = vec![0; payload.len()];
        file.read_exact(&mut actual).expect("read payload");
        assert_eq!(actual, payload);

        file.seek(SeekFrom::Start(quantized_source.byte_offset))
            .expect("seek to quantized payload");
        let mut actual_quantized = vec![0; q8_payload.len()];
        file.read_exact(&mut actual_quantized)
            .expect("read quantized payload");
        assert_eq!(actual_quantized, q8_payload);
    }

    #[test]
    fn test_gguf_quant_types_block_decoders() {
        // Test Q8_0 block unpacking (scale = 0.5, values = -16..15)
        let mut q8_block = [0u8; 34];
        // f16 for 0.5: sign=0, exp=14 (14-15+127=126 -> exp=14), mant=0 -> 0x3800
        let scale_f16 = 0x3800_u16;
        q8_block[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        for i in 0..32 {
            q8_block[2 + i] = (i as i8 - 16) as u8;
        }
        let weights = GgufBlockDecoder::unpack_q8_0(&q8_block).unwrap();
        assert_eq!(weights.len(), 32);
        assert!((weights[0] - (-8.0)).abs() < 1e-4);
        assert!((weights[16] - 0.0).abs() < 1e-4);

        // Test Q4_0 block unpacking
        let mut q4_block = [0u8; 18];
        q4_block[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        // Nibbles 8 (value 0) and 10 (value 2) -> byte = 0xA8
        for i in 0..16 {
            q4_block[2 + i] = 0xA8;
        }
        let q4_weights = GgufBlockDecoder::unpack_q4_0(&q4_block).unwrap();
        assert_eq!(q4_weights.len(), 32);
        assert!((q4_weights[0] - 0.0).abs() < 1e-4); // (8 - 8) * 0.5 = 0.0
        assert!((q4_weights[16] - 1.0).abs() < 1e-4); // (10 - 8) * 0.5 = 1.0
    }
}
