# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |

## Binary Verification & Cryptographic Provenance

All official release binaries, archives, and manifests for THME (Epiphanes) are signed cryptographically:

1. **SHA-256 Digest**:
   ```bash
   sha256sum -c SHA256SUMS.txt
   ```

2. **Minisign Verification**:
   ```bash
   minisign -Vm <archive>.tar.gz -P RWTJ3yD... (Epiphanes public key)
   ```

3. **GPG Signature Verification**:
   ```bash
   gpg --verify <archive>.tar.gz.asc <archive>.tar.gz
   ```

## Reporting a Vulnerability

We take the security of **THME** by **Epiphanes** seriously. If you discover a vulnerability or security flaw, please do not disclose it publicly in an issue or forum.

Instead, please report vulnerabilities by contacting the maintainers directly through GitHub private vulnerability reporting or contacting the Epiphanes team at `https://github.com/EdEpiphanes/THME-Epiphanes/security/advisories`.

Please provide:
- A description of the vulnerability and its potential impact.
- Steps or a minimal reproducible example demonstrating the issue.
- Potential mitigations or fixes if you have identified any.

We will acknowledge receipt within 48 hours and work with you to address and publish a fix.
