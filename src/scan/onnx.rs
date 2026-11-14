// SPDX-License-Identifier: AGPL-3.0-only
//! Static ONNX inspection: walks the protobuf `ModelProto` without loading anything. Reading the
//! file executes nothing; the risk is what a loader does with it. A tensor may keep its bytes in
//! another file named by `external_data` `location`, and a loader opens `<model folder>/<location>`,
//! so an absolute path or a `..` makes a hostile model read any file (CVE-2022-25882).
//!
//! The walker follows only the fields that lead to tensors, nodes and operator domains, and skips
//! everything else (weights included) without copying it.

use std::ops::Range;

/// What the walk found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OnnxScan {
    /// Every `external_data` `location` value, in file order.
    pub external_locations: Vec<String>,
    /// Operator domains of `opset_import` entries and nodes, without duplicates.
    pub domains: Vec<String>,
    /// The protobuf ended early or broke at this byte offset.
    pub malformed_at: Option<usize>,
}

/// Deepest message nesting followed; deeper content is reported as malformed.
const MAX_DEPTH: usize = 16;

#[derive(Clone, Copy)]
enum Msg {
    Model,
    OpSet,
    Graph,
    Node,
    Attribute,
    Tensor,
    Entry,
}

fn varint(data: &[u8], at: usize) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for i in 0..10 {
        let b = *data.get(at.checked_add(i)?)?;
        value |= u64::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// Fields of one message in `data[at..end]`: `(number, wire type, value range)`. An `Err` carries
/// the offset where the encoding breaks and ends the iteration.
struct Fields<'a> {
    data: &'a [u8],
    at: usize,
    end: usize,
}

impl Iterator for Fields<'_> {
    type Item = Result<(u64, u8, Range<usize>), usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.at >= self.end {
            return None;
        }
        let start = self.at;
        let item = (|| {
            let (tag, n) = varint(self.data, start)?;
            let at = start + n;
            let (value, next) = match tag & 7 {
                0 => {
                    let (_, n) = varint(self.data, at)?;
                    (at..at + n, at + n)
                }
                1 => (at..at.checked_add(8)?, at + 8),
                5 => (at..at.checked_add(4)?, at + 4),
                2 => {
                    let (len, n) = varint(self.data, at)?;
                    let from = at + n;
                    let to = from.checked_add(usize::try_from(len).ok()?)?;
                    (from..to, to)
                }
                _ => return None,
            };
            (next <= self.end).then_some((tag >> 3, (tag & 7) as u8, value))
        })();
        match item {
            Some((number, wire, value)) => {
                self.at = value.end;
                Some(Ok((number, wire, value)))
            }
            None => {
                self.at = self.end;
                Some(Err(start))
            }
        }
    }
}

fn walk(data: &[u8], range: Range<usize>, msg: Msg, depth: usize, out: &mut OnnxScan) {
    if depth > MAX_DEPTH {
        out.malformed_at.get_or_insert(range.start);
        return;
    }
    let (mut key, mut value) = (None, None);
    let fields = Fields {
        data,
        at: range.start,
        end: range.end,
    };
    for field in fields {
        let (number, wire, r) = match field {
            Ok(f) => f,
            Err(at) => {
                out.malformed_at.get_or_insert(at);
                return;
            }
        };
        if wire != 2 {
            continue;
        }
        let child = match (msg, number) {
            (Msg::Model, 7) => Some(Msg::Graph),
            (Msg::Model, 8) => Some(Msg::OpSet),
            (Msg::Graph, 1) => Some(Msg::Node),
            (Msg::Graph, 5) | (Msg::Attribute, 5 | 10) => Some(Msg::Tensor),
            (Msg::Attribute, 6 | 11) => Some(Msg::Graph),
            (Msg::Node, 5) => Some(Msg::Attribute),
            (Msg::Tensor, 13) => Some(Msg::Entry),
            _ => None,
        };
        let text = |r: &Range<usize>| {
            data.get(r.clone())
                .map(|b| String::from_utf8_lossy(b).into_owned())
        };
        match (msg, number, child) {
            (_, _, Some(child)) => walk(data, r, child, depth + 1, out),
            (Msg::OpSet, 1, _) | (Msg::Node, 7, _) => {
                out.domains.extend(text(&r));
            }
            (Msg::Entry, 1, _) => key = text(&r),
            (Msg::Entry, 2, _) => value = text(&r),
            _ => {}
        }
    }
    if let (Msg::Entry, Some(key), Some(value)) = (msg, key, value) {
        if key == "location" {
            out.external_locations.push(value);
        }
    }
}

/// Walks `data` as an ONNX `ModelProto`.
pub fn scan_onnx(data: &[u8]) -> OnnxScan {
    let mut out = OnnxScan::default();
    walk(data, 0..data.len(), Msg::Model, 0, &mut out);
    let mut seen = std::collections::HashSet::new();
    out.domains.retain(|d| seen.insert(d.clone()));
    out
}

/// True when a loader would leave the model folder: absolute path, drive prefix, NUL or a `..`
/// component.
pub fn is_unsafe_location(location: &str) -> bool {
    location.starts_with(['/', '\\'])
        || location.chars().nth(1) == Some(':')
        || location.contains('\0')
        || location.split(['/', '\\']).any(|c| c == "..")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn pb_varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return out;
            }
            out.push(b | 0x80);
        }
    }

    pub(crate) fn pb_field(num: u32, wire: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = pb_varint(u64::from(num << 3 | u32::from(wire)));
        if wire == 2 {
            out.extend(pb_varint(payload.len() as u64));
        }
        out.extend_from_slice(payload);
        out
    }

    pub(crate) fn tensor(location: &str) -> Vec<u8> {
        let mut entry = pb_field(1, 2, b"location");
        entry.extend(pb_field(2, 2, location.as_bytes()));
        let mut t = pb_field(9, 2, b"raw bytes skipped");
        t.extend(pb_field(13, 2, &entry));
        t.extend(pb_field(14, 0, &pb_varint(1)));
        t
    }

    pub(crate) fn model(graph: &[u8]) -> Vec<u8> {
        pb_field(7, 2, graph)
    }

    #[test]
    fn finds_locations_in_initializers_and_domains() {
        let mut graph = pb_field(5, 2, &tensor("w.bin"));
        graph.extend(pb_field(1, 2, &pb_field(7, 2, b"com.example")));
        let mut m = model(&graph);
        m.extend(pb_field(8, 2, &pb_field(1, 2, b"com.example")));
        let scan = scan_onnx(&m);
        assert_eq!(scan.external_locations, ["w.bin"]);
        assert_eq!(scan.domains, ["com.example"]);
        assert_eq!(scan.malformed_at, None);
    }

    #[test]
    fn unsafe_locations() {
        for bad in [
            "../../etc/passwd",
            "/etc/passwd",
            "C:\\Windows\\x",
            "a\\..\\b",
            "a\0b",
        ] {
            assert!(is_unsafe_location(bad), "{bad}");
        }
        for ok in ["sub/dir/w.bin", "..hidden", "w.bin"] {
            assert!(!is_unsafe_location(ok), "{ok}");
        }
    }

    #[test]
    fn every_prefix_is_safe_and_a_mid_field_cut_is_malformed() {
        let m = model(&pb_field(5, 2, &tensor("w.bin")));
        for cut in 0..=m.len() {
            let _ = scan_onnx(&m[..cut]);
        }
        assert!(scan_onnx(&m[..m.len() - 1]).malformed_at.is_some());
    }

    #[test]
    fn junk_never_panics() {
        let mut state = 7u64;
        for len in 0..300 {
            let junk: Vec<u8> = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 56) as u8
                })
                .collect();
            let _ = scan_onnx(&junk);
        }
    }

    #[test]
    fn repeated_attribute_lists_are_walked() {
        let node = |attr: &[u8]| pb_field(1, 2, &pb_field(5, 2, attr));
        let graph = pb_field(5, 2, &tensor("../g"));
        for attr in [pb_field(10, 2, &tensor("../t")), pb_field(11, 2, &graph)] {
            let scan = scan_onnx(&model(&node(&attr)));
            assert_eq!(scan.external_locations.len(), 1);
        }
    }

    #[test]
    fn deep_nesting_is_reported_not_overflowed() {
        let mut graph = pb_field(5, 2, &tensor("../x"));
        for _ in 0..100 {
            let attr = pb_field(6, 2, &graph);
            graph = pb_field(1, 2, &pb_field(5, 2, &attr));
        }
        assert!(scan_onnx(&model(&graph)).malformed_at.is_some());
    }
}
