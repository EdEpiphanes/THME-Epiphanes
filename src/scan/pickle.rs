//! Static pickle inspection: walks the opcode stream without executing anything and reports
//! every import (`GLOBAL`, `INST`, `STACK_GLOBAL`) a load would perform. Loading a pickle can
//! call any importable function, so the imports are the risk.
//!
//! `STACK_GLOBAL` takes its module and name from the stack. The walker tracks string pushes
//! and memo slots, which covers how pickles are written in practice; an import it cannot
//! resolve is reported as unresolved rather than ignored.

/// One import a pickle would perform when loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickleImport {
    pub module: String,
    pub name: String,
}

/// What the walk found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PickleScan {
    pub imports: Vec<PickleImport>,
    /// `STACK_GLOBAL` whose operands were not plain strings.
    pub unresolved_globals: usize,
    /// The stream ended early or held an unknown opcode at this byte offset.
    pub malformed_at: Option<usize>,
}

fn line(data: &[u8], at: usize) -> Option<(String, usize)> {
    let rest = data.get(at..)?;
    let end = rest.iter().position(|&b| b == b'\n')?;
    Some((
        String::from_utf8_lossy(&rest[..end]).into_owned(),
        at + end + 1,
    ))
}

fn le(data: &[u8], at: usize, n: usize) -> Option<u64> {
    let bytes = data.get(at..at.checked_add(n)?)?;
    Some(
        bytes
            .iter()
            .rev()
            .fold(0u64, |acc, &b| (acc << 8) | u64::from(b)),
    )
}

/// Walks `data` as a pickle stream (protocols 0 to 5) until `STOP` or the end.
pub fn scan_pickle(data: &[u8]) -> PickleScan {
    let mut out = PickleScan::default();
    let mut recent: Vec<Option<String>> = Vec::new();
    let mut memo: std::collections::HashMap<u64, Option<String>> = Default::default();
    let mut next_memo = 0u64;
    let mut at = 0usize;
    let push = |recent: &mut Vec<Option<String>>, v: Option<String>| {
        recent.push(v);
        if recent.len() > 8 {
            recent.remove(0);
        }
    };
    loop {
        let Some(&op) = data.get(at) else {
            out.malformed_at = Some(at);
            return out;
        };
        at += 1;
        // Length-prefixed and fixed-size arguments.
        let fixed = |n: usize| -> Option<usize> { at.checked_add(n).filter(|&e| e <= data.len()) };
        let sized = |len_bytes: usize| -> Option<(usize, usize)> {
            let len = usize::try_from(le(data, at, len_bytes)?).ok()?;
            let start = at + len_bytes;
            let end = start.checked_add(len)?;
            (end <= data.len()).then_some((start, end))
        };
        let next = match op {
            b'.' => return out,
            // No argument.
            b'(' | b'0' | b'1' | b'2' | b'N' | b'R' | b'a' | b'b' | b'd' | b'}' | b'e' | b'l'
            | b']' | b'o' | b's' | b't' | b')' | b'u' | b'Q' | 0x81 | 0x85 | 0x86 | 0x87 | 0x88
            | 0x89 | 0x8f | 0x90 | 0x91 | 0x92 | 0x97 | 0x98 => {
                if !matches!(op, b'0' | b'1' | b'2' | b'N') {
                    push(&mut recent, None);
                }
                Some(at)
            }
            0x94 => {
                // MEMOIZE the top of the stack.
                memo.insert(next_memo, recent.last().cloned().flatten());
                next_memo += 1;
                Some(at)
            }
            0x93 => {
                // STACK_GLOBAL: module and name are the two most recent pushes.
                let n = recent.len();
                match (n.checked_sub(2).and_then(|i| recent.get(i)), recent.last()) {
                    (Some(Some(module)), Some(Some(name))) => out.imports.push(PickleImport {
                        module: module.clone(),
                        name: name.clone(),
                    }),
                    _ => out.unresolved_globals += 1,
                }
                push(&mut recent, None);
                Some(at)
            }
            // Text line arguments.
            b'I' | b'L' | b'F' | b'P' => line(data, at).map(|(_, e)| {
                push(&mut recent, None);
                e
            }),
            b'S' | b'V' => line(data, at).map(|(s, e)| {
                push(&mut recent, Some(s.trim_matches(['\'', '"']).to_string()));
                e
            }),
            b'g' | b'p' => line(data, at).map(|(s, e)| {
                let idx = s.trim().parse::<u64>().unwrap_or(u64::MAX);
                if op == b'g' {
                    push(&mut recent, memo.get(&idx).cloned().flatten());
                } else {
                    memo.insert(idx, recent.last().cloned().flatten());
                }
                e
            }),
            b'c' | b'i' => line(data, at).and_then(|(module, e)| {
                line(data, e).map(|(name, e2)| {
                    out.imports.push(PickleImport { module, name });
                    push(&mut recent, None);
                    e2
                })
            }),
            // Fixed-size arguments.
            0x8a | 0x8b => {
                // LONG1 / LONG4: little-endian length, then the integer bytes.
                let n = if op == 0x8a { 1 } else { 4 };
                sized(n).map(|(_, e)| {
                    push(&mut recent, None);
                    e
                })
            }
            b'J' => fixed(4).inspect(|_| push(&mut recent, None)),
            b'K' => fixed(1).inspect(|_| push(&mut recent, None)),
            b'M' => fixed(2).inspect(|_| push(&mut recent, None)),
            b'G' => fixed(8).inspect(|_| push(&mut recent, None)),
            0x82 => fixed(1).inspect(|_| push(&mut recent, None)),
            0x83 => fixed(2).inspect(|_| push(&mut recent, None)),
            0x84 => fixed(4).inspect(|_| push(&mut recent, None)),
            0x80 => fixed(1),
            0x95 => fixed(8),
            b'h' | b'j' => {
                let n = if op == b'h' { 1 } else { 4 };
                le(data, at, n).map(|idx| {
                    push(&mut recent, memo.get(&idx).cloned().flatten());
                    at + n
                })
            }
            b'q' | b'r' => {
                let n = if op == b'q' { 1 } else { 4 };
                le(data, at, n).map(|idx| {
                    memo.insert(idx, recent.last().cloned().flatten());
                    at + n
                })
            }
            // Length-prefixed strings and bytes.
            b'X' | 0x8c | 0x8d | b'T' | b'U' => {
                let n = match op {
                    0x8c | b'U' => 1,
                    0x8d => 8,
                    _ => 4,
                };
                sized(n).map(|(s, e)| {
                    let text = data
                        .get(s..e)
                        .map(|b| String::from_utf8_lossy(b).into_owned());
                    push(&mut recent, text);
                    e
                })
            }
            b'B' | b'C' | 0x8e | 0x96 => {
                let n = match op {
                    b'C' => 1,
                    b'B' => 4,
                    _ => 8,
                };
                sized(n).map(|(_, e)| {
                    push(&mut recent, None);
                    e
                })
            }
            _ => None,
        };
        match next {
            Some(n) => at = n,
            None => {
                out.malformed_at = Some(at - 1);
                return out;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Protocol 0 `GLOBAL` + `REDUCE`, the classic code-execution pattern. Test fixture only:
    /// bytes are inspected, never loaded.
    #[test]
    fn finds_protocol_0_global() {
        let data = b"cos\nsystem\n(S'echo fixture'\ntR.";
        let scan = scan_pickle(data);
        assert_eq!(
            scan.imports,
            vec![PickleImport {
                module: "os".into(),
                name: "system".into()
            }]
        );
        assert_eq!(scan.malformed_at, None);
    }

    /// Protocol 4 `STACK_GLOBAL` with memoized strings, as modern pickles write it.
    #[test]
    fn resolves_stack_global_through_memo() {
        let mut data = vec![0x80, 4, 0x95];
        data.extend_from_slice(&0u64.to_le_bytes());
        for s in ["builtins", "eval"] {
            data.push(0x8c);
            data.push(s.len() as u8);
            data.extend_from_slice(s.as_bytes());
            data.push(0x94);
        }
        data.push(0x93);
        data.extend_from_slice(b"\x8c\x011\x85R.");
        let scan = scan_pickle(&data);
        assert_eq!(scan.imports[0].module, "builtins");
        assert_eq!(scan.imports[0].name, "eval");
    }

    #[test]
    fn plain_data_has_no_imports_and_garbage_never_panics() {
        // [1, 2, 'x'] in protocol 2.
        let data = b"\x80\x02]q\x00(K\x01K\x02X\x01\x00\x00\x00xq\x01e.";
        let scan = scan_pickle(data);
        assert!(scan.imports.is_empty());
        assert_eq!(scan.malformed_at, None);
        let mut state = 5u64;
        for len in 0..300 {
            let junk: Vec<u8> = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 56) as u8
                })
                .collect();
            let _ = scan_pickle(&junk);
        }
        assert!(scan_pickle(b"\x8c\xff").malformed_at.is_some());
    }
}
