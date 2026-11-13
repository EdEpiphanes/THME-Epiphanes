//! Signed `.epiph` files (specification section 7).
//!
//! The signature is a Minisign signature over the manifest bytes. The manifest holds the
//! SHA-256 of every payload, so one signature binds the whole file: change a weight and its
//! digest no longer matches; change a digest and the signature no longer verifies.
//!
//! It is stored in a trailer after the last payload:
//! `[minisign signature text][u64 length, little-endian][b"EPIPHSIG"]`. Readers that do not know
//! the trailer ignore it; signing again replaces it.

use anyhow::{bail, Context, Result};
use minisign::{PublicKey, SecretKey, SignatureBox};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::{align, open, Opened};
use crate::exporter::compute_sha256;

/// Last eight bytes of a signed file.
pub const TRAILER_MAGIC: [u8; 8] = *b"EPIPHSIG";
const MAX_SIGNATURE: u64 = 64 * 1024;

/// The public key Epiphanes signs official releases and `.epiph` files with.
pub const EPIPHANES_PUBLIC_KEY: &str = include_str!("../../minisign.pub");

/// What a signature check found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignatureStatus {
    Unsigned,
    /// Signed and verified with the given key.
    Valid {
        key_id: String,
        trusted_comment: String,
        by_epiphanes: bool,
    },
    /// Signed with a key the caller did not supply; not verified.
    UnknownKey {
        key_id: String,
    },
}

/// The Epiphanes public key from `minisign.pub`.
pub fn epiphanes_key() -> Result<PublicKey> {
    public_key_from_text(EPIPHANES_PUBLIC_KEY)
}

/// A Minisign public key from the text of a `.pub` file (comment line, then base64).
pub fn public_key_from_text(text: &str) -> Result<PublicKey> {
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("untrusted comment:"))
        .context("no key line in the public key file")?;
    PublicKey::from_base64(line).map_err(|e| anyhow::anyhow!("invalid Minisign public key: {e}"))
}

fn key_id(keynum: &[u8]) -> String {
    keynum.iter().rev().map(|b| format!("{b:02X}")).collect()
}

/// The signature stored in `path`, if any.
pub fn read_signature(path: &Path) -> Result<Option<SignatureBox>> {
    let Opened { payload_end, .. } = open(path)?;
    let mut file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    let start = align(payload_end);
    if len < start + 16 {
        return Ok(None);
    }
    let mut tail = [0u8; 16];
    file.seek(SeekFrom::Start(len - 16))?;
    file.read_exact(&mut tail)?;
    if tail[8..] != TRAILER_MAGIC {
        return Ok(None);
    }
    let mut size = [0u8; 8];
    size.copy_from_slice(&tail[..8]);
    let sig_len = u64::from_le_bytes(size);
    if sig_len > MAX_SIGNATURE || start + sig_len + 16 != len {
        bail!("{path:?} has a malformed signature trailer");
    }
    let mut text = vec![0u8; usize::try_from(sig_len)?];
    file.seek(SeekFrom::Start(start))?;
    file.read_exact(&mut text)?;
    let text = String::from_utf8(text).context("signature trailer is not UTF-8")?;
    SignatureBox::from_string(&text)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("{path:?} has an unreadable signature: {e}"))
}

/// Signs `path` in place with `secret_key`, replacing any earlier signature. `public_key`, when
/// given, must belong to the same pair; the new signature is checked against it.
pub fn sign(path: &Path, secret_key: &SecretKey, public_key: Option<&PublicKey>) -> Result<()> {
    let Opened {
        manifest_bytes,
        payload_end,
        ..
    } = open(path)?;
    let trusted = format!("epiph-manifest-sha256:{}", compute_sha256(&manifest_bytes));
    let signature = minisign::sign(
        public_key,
        secret_key,
        std::io::Cursor::new(&manifest_bytes),
        Some(&trusted),
        Some("signature of an Epiphanes model file (.epiph)"),
    )
    .map_err(|e| anyhow::anyhow!("signing failed: {e}"))?;
    let text = signature.into_string();
    let start = align(payload_end);
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("Failed to open {path:?} for signing"))?;
    file.set_len(start)?;
    file.seek(SeekFrom::Start(start))?;
    file.write_all(text.as_bytes())?;
    file.write_all(&(text.len() as u64).to_le_bytes())?;
    file.write_all(&TRAILER_MAGIC)?;
    file.flush()?;
    Ok(())
}

/// Checks the signature of `path` against `key`, or against the Epiphanes key when `key` is
/// `None`. A signature that does not verify is an error.
pub fn check(path: &Path, key: Option<&PublicKey>) -> Result<SignatureStatus> {
    let Some(signature) = read_signature(path)? else {
        return Ok(SignatureStatus::Unsigned);
    };
    let Opened { manifest_bytes, .. } = open(path)?;
    let epiphanes = epiphanes_key()?;
    let id = key_id(signature.keynum());
    let (pk, by_epiphanes) = match key {
        Some(k) => (k.clone(), k.keynum() == epiphanes.keynum()),
        None if signature.keynum() == epiphanes.keynum() => (epiphanes, true),
        None => return Ok(SignatureStatus::UnknownKey { key_id: id }),
    };
    minisign::verify(
        &pk,
        &signature,
        std::io::Cursor::new(&manifest_bytes),
        true,
        false,
        false,
    )
    .map_err(|e| anyhow::anyhow!("{path:?}: signature does not verify: {e}"))?;
    let trusted_comment = signature
        .trusted_comment()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let expected = format!("epiph-manifest-sha256:{}", compute_sha256(&manifest_bytes));
    if trusted_comment != expected {
        bail!("{path:?}: signature comment does not match the manifest");
    }
    Ok(SignatureStatus::Valid {
        key_id: id,
        trusted_comment,
        by_epiphanes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::evaluate::tests::synthetic_checkpoint;
    use minisign::KeyPair;

    fn packed(dir: &Path) -> std::path::PathBuf {
        let (model, _) = synthetic_checkpoint(dir);
        let ir = crate::parser::ModelImporter::import(&model).unwrap();
        let out = dir.join("m.epiph");
        let options = crate::epiph::PackOptions {
            created_unix: Some(0),
            ..Default::default()
        };
        crate::epiph::write(&ir, &options, &out).unwrap();
        out
    }

    #[test]
    fn sign_verify_and_resign() {
        let dir = tempfile::tempdir().unwrap();
        let path = packed(dir.path());
        let pair = KeyPair::generate_unencrypted_keypair().unwrap();
        assert_eq!(
            check(&path, Some(&pair.pk)).unwrap(),
            SignatureStatus::Unsigned
        );

        sign(&path, &pair.sk, Some(&pair.pk)).unwrap();
        let status = check(&path, Some(&pair.pk)).unwrap();
        assert!(matches!(
            status,
            SignatureStatus::Valid {
                by_epiphanes: false,
                ..
            }
        ));
        // Without a key, a non-Epiphanes signature is reported, not trusted.
        assert!(matches!(
            check(&path, None).unwrap(),
            SignatureStatus::UnknownKey { .. }
        ));
        // A signed file is still a valid .epiph file for every other reader.
        assert!(crate::epiph::verify(&path).is_ok());
        assert!(crate::epiph::read(&path).is_ok());

        // Signing again replaces the trailer instead of stacking a second one.
        let before = std::fs::metadata(&path).unwrap().len();
        sign(&path, &pair.sk, None).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), before);
        assert!(check(&path, Some(&pair.pk)).is_ok());
    }

    #[test]
    fn tampering_and_wrong_keys_fail() {
        let dir = tempfile::tempdir().unwrap();
        let path = packed(dir.path());
        let pair = KeyPair::generate_unencrypted_keypair().unwrap();
        let other = KeyPair::generate_unencrypted_keypair().unwrap();
        sign(&path, &pair.sk, Some(&pair.pk)).unwrap();
        assert!(check(&path, Some(&other.pk)).is_err(), "wrong key");

        // Change one manifest byte (inside a JSON string, so the file still parses).
        let mut bytes = std::fs::read(&path).unwrap();
        let needle = b"\"tool\":\"thme\"";
        let at = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .unwrap()
            + 9;
        bytes[at] = b'T';
        std::fs::write(&path, &bytes).unwrap();
        assert!(crate::epiph::read(&path).is_ok());
        assert!(check(&path, Some(&pair.pk)).is_err(), "edited manifest");

        // A broken trailer is an error, not a panic.
        let path2 = packed(&dir.path().join("b"));
        sign(&path2, &pair.sk, None).unwrap();
        let mut b2 = std::fs::read(&path2).unwrap();
        let n = b2.len();
        b2[n - 12] ^= 0x40; // corrupt the length field
        std::fs::write(&path2, b2).unwrap();
        assert!(check(&path2, Some(&pair.pk)).is_err());
    }

    #[test]
    fn epiphanes_key_parses_from_the_committed_public_key() {
        let key = epiphanes_key().unwrap();
        assert_eq!(key_id(key.keynum()).len(), 16);
        assert!(public_key_from_text("untrusted comment: x\n").is_err());
    }
}
