# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |

## Binary Verification & Cryptographic Provenance

Every release archive on the GitHub Releases page ships with a SHA-256 checksum and a Minisign
signature. The combined `SHA256SUMS.txt` manifest is signed too.

**Release signing key** (also committed as [minisign.pub](minisign.pub)):

```text
untrusted comment: minisign public key 35DF5AFBFC0331CD
RWTNMQP8+1rfNWQsVzNDUlnVRotV3ZdJrNsbpoP8+FedvedBGyRUzB6O
```

1. **Verify the signature** (install Minisign with `winget install jedisct1.minisign`,
   `brew install minisign` or `apt install minisign`):
   ```bash
   minisign -Vm SHA256SUMS.txt -P RWTNMQP8+1rfNWQsVzNDUlnVRotV3ZdJrNsbpoP8+FedvedBGyRUzB6O
   minisign -Vm <archive>.tar.gz -P RWTNMQP8+1rfNWQsVzNDUlnVRotV3ZdJrNsbpoP8+FedvedBGyRUzB6O
   ```

2. **Verify the checksums**:
   ```bash
   sha256sum -c SHA256SUMS.txt --ignore-missing
   ```

If the key ever changes, the new key is announced in a release signed with the old one.

## Reporting a Vulnerability

We take the security of **THME** by **Epiphanes** seriously. If you discover a vulnerability or security flaw, please do not disclose it publicly in an issue or forum.

Instead, please report vulnerabilities by contacting the maintainers directly through GitHub private vulnerability reporting or contacting the Epiphanes team at `https://github.com/EdEpiphanes/THME-Epiphanes/security/advisories`.

Please provide:
- A description of the vulnerability and its potential impact.
- Steps or a minimal reproducible example demonstrating the issue.
- Potential mitigations or fixes if you have identified any.

We will acknowledge receipt within 48 hours and work with you to address and publish a fix.
