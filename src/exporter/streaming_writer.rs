//! Bounded-memory streaming helpers for exporter payloads.

use anyhow::{Context, Result};
use std::io::{Read, Write};

/// Maximum temporary buffer used while copying a tensor payload.
pub const STREAM_CHUNK_SIZE: usize = 64 * 1024;

/// Writes payloads from readers without retaining the full payload in memory.
#[derive(Debug)]
pub struct StreamingChunkWriter<W> {
    writer: W,
}

impl<W: Write> StreamingChunkWriter<W> {
    /// Wraps a destination writer.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Copies exactly `byte_len` bytes using a fixed-size buffer.
    ///
    /// Returns the number of bytes written. A short input returns an error.
    pub fn write_reader<R: Read>(&mut self, reader: &mut R, byte_len: u64) -> Result<u64> {
        let mut buffer = [0u8; STREAM_CHUNK_SIZE];
        let mut remaining = byte_len;

        while remaining > 0 {
            let chunk_len = usize::try_from(remaining.min(STREAM_CHUNK_SIZE as u64))
                .context("Streaming chunk length does not fit in memory")?;
            reader
                .read_exact(&mut buffer[..chunk_len])
                .with_context(|| {
                    format!("Tensor payload ended with {remaining} bytes remaining")
                })?;
            self.writer
                .write_all(&buffer[..chunk_len])
                .context("Failed writing tensor payload chunk")?;
            remaining -= chunk_len as u64;
        }

        Ok(byte_len)
    }

    /// Flushes buffered destination state.
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .context("Failed flushing streamed output")
    }

    /// Returns the wrapped destination.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

/// Copies a reader into a writer using a fixed-size buffer until EOF.
pub fn copy_reader<W: Write, R: Read>(writer: &mut W, reader: &mut R) -> Result<u64> {
    let mut buffer = [0u8; STREAM_CHUNK_SIZE];
    let mut total = 0u64;

    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .context("Failed reading payload chunk")?;
        if bytes_read == 0 {
            break;
        }
        writer
            .write_all(&buffer[..bytes_read])
            .context("Failed writing payload chunk")?;
        total = total
            .checked_add(bytes_read as u64)
            .ok_or_else(|| anyhow::anyhow!("Stream length exceeds u64"))?;
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_streaming_chunk_writer_exact() {
        let input_data = vec![42u8; 128 * 1024]; // 128 KiB
        let mut reader = Cursor::new(input_data.clone());
        let mut output = Vec::new();

        let mut chunk_writer = StreamingChunkWriter::new(&mut output);
        let written = chunk_writer
            .write_reader(&mut reader, 128 * 1024)
            .expect("write reader");
        chunk_writer.flush().expect("flush");

        assert_eq!(written, 128 * 1024);
        assert_eq!(output, input_data);
    }

    #[test]
    fn test_copy_reader_eof() {
        let input_data = b"Streaming payload test buffer across multiple boundaries";
        let mut reader = Cursor::new(input_data);
        let mut output = Vec::new();

        let copied = copy_reader(&mut output, &mut reader).expect("copy reader");
        assert_eq!(copied, input_data.len() as u64);
        assert_eq!(&output[..], input_data);
    }
}
