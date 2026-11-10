use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::ir::{ModelArchitecture, ModelIR, TensorIR};
use crate::profiling::WeightAnalyzer;

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

impl GgufValue {
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

pub fn ggml_type_str(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        28 => "BF16",
        _ => "QUANT_OTHER",
    }
}

pub fn ggml_type_bytes_per_elem(t: u32) -> f64 {
    match t {
        0 => 4.0,
        1 | 28 => 2.0,
        8 => 34.0 / 32.0, // 1.0625
        2 => 18.0 / 32.0, // 0.5625
        10 => 84.0 / 256.0,
        12 => 144.0 / 256.0,
        13 => 176.0 / 256.0,
        14 => 210.0 / 256.0,
        _ => 1.0,
    }
}

pub struct GgufReader<'a> {
    data: &'a [u8],
    cursor: usize,
}

impl<'a> GgufReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, cursor: 0 }
    }

    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.cursor + n > self.data.len() {
            bail!(
                "Unexpected EOF in GGUF file: needed {} bytes at {}",
                n,
                self.cursor
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
        let bytes = self.read_bytes(len)?;
        let s = String::from_utf8_lossy(bytes).into_owned();
        Ok(s)
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
                let count = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    if arr.len() < 1024 {
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
            _ => bail!("Unsupported GGUF value type: {}", val_type),
        }
    }
}

pub struct GgufParser;

impl GgufParser {
    pub fn parse_from_file<P: AsRef<Path>>(path: P) -> Result<ModelIR> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open GGUF file: {:?}", path.as_ref()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        Self::parse_from_bytes(&mmap, path.as_ref())
    }

    pub fn parse_from_bytes(data: &[u8], path: &Path) -> Result<ModelIR> {
        let mut reader = GgufReader::new(data);

        // Header check: 4 bytes "GGUF"
        let magic = reader.read_bytes(4)?;
        if magic != b"GGUF" {
            bail!("Invalid GGUF magic bytes: {:?}", magic);
        }

        let version = reader.read_u32()?;
        if version != 2 && version != 3 {
            bail!("Unsupported GGUF version: {}", version);
        }

        let tensor_count = reader.read_u64()? as usize;
        let metadata_kv_count = reader.read_u64()? as usize;

        // Metadata KV pairs
        let mut metadata: HashMap<String, GgufValue> = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = reader.read_string()?;
            let val_type = reader.read_u32()?;
            let val = reader.read_value(val_type)?;
            metadata.insert(key, val);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32)
            .max(1) as usize;

        // Tensor Info Table
        struct RawTensorInfo {
            name: String,
            shape: Vec<usize>,
            ggml_type: u32,
            offset: u64,
        }

        let mut raw_tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dims = reader.read_u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(reader.read_u64()? as usize);
            }
            // GGUF dims are stored in column-major reverse order
            dims.reverse();

            let ggml_type = reader.read_u32()?;
            let offset = reader.read_u64()?;

            raw_tensors.push(RawTensorInfo {
                name,
                shape: dims,
                ggml_type,
                offset,
            });
        }

        // Tensor data alignment boundary
        let header_end = reader.cursor;
        let padding = (alignment - (header_end % alignment)) % alignment;
        let data_start = header_end + padding;

        let mut tensors = Vec::with_capacity(tensor_count);
        for raw in raw_tensors {
            let dtype = ggml_type_str(raw.ggml_type).to_string();
            let numel: usize = raw.shape.iter().product();
            let byte_size =
                (numel as f64 * ggml_type_bytes_per_elem(raw.ggml_type)).ceil() as usize;

            let mut tensor_ir = TensorIR::new(raw.name, raw.shape, dtype.clone(), byte_size);

            // Extract weight profile if data is within bounds
            let tensor_byte_offset = data_start.saturating_add(raw.offset as usize);
            if tensor_byte_offset.saturating_add(byte_size) <= data.len() {
                let slice = &data[tensor_byte_offset..tensor_byte_offset + byte_size];
                tensor_ir.weight_profile = Some(WeightAnalyzer::profile_tensor(slice, &dtype));
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
            .and_then(|v| v.as_u64())
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
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4096);

        let num_attention_heads = metadata
            .get(&format!("{}.attention.head_count", arch_str))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(32);

        let num_key_value_heads = metadata
            .get(&format!("{}.attention.head_count_kv", arch_str))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let max_position_embeddings = metadata
            .get(&format!("{}.context_length", arch_str))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4096);

        let vocab_size = metadata
            .get(&format!("{}.vocab_size", arch_str))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(32000);

        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("gguf_model")
            .to_string();

        let model_ir = ModelIR::new(
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

        Ok(model_ir)
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
    }
}
