// SPDX-License-Identifier: AGPL-3.0-only
//! The real network fetcher behind [`super::Fetch`]: ureq over rustls (ring) with the OS roots.

use super::Fetch;
use anyhow::{Context, Result};
use std::io::Write;

/// Largest API reply accepted. A model listing with blob hashes is far below this.
const MAX_TEXT_BYTES: u64 = 64 << 20;

pub struct HttpFetch {
    agent: ureq::Agent,
}

impl HttpFetch {
    pub fn new() -> Self {
        // The OS certificate store, so corporate roots already trusted by the machine work.
        let tls = ureq::tls::TlsConfig::builder()
            .root_certs(ureq::tls::RootCerts::PlatformVerifier)
            .build();
        Self {
            agent: ureq::Agent::config_builder()
                .tls_config(tls)
                .build()
                .new_agent(),
        }
    }
}

impl Default for HttpFetch {
    fn default() -> Self {
        Self::new()
    }
}

impl Fetch for HttpFetch {
    fn text(&self, url: &str, token: Option<&str>) -> Result<String> {
        let mut req = self.agent.get(url);
        if let Some(t) = token {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        // Errors name the URL only; the header value is never part of an error or a log line.
        let mut resp = req.call().with_context(|| format!("GET {url}"))?;
        resp.body_mut()
            .with_config()
            .limit(MAX_TEXT_BYTES)
            .read_to_string()
            .with_context(|| format!("read {url}"))
    }

    // shortcut: no resume and no byte cap on file bodies; the size and sha256 checks in
    // mod.rs reject a bad body, but a stalled transfer waits on the OS. Upgrade with a
    // timeout and Range resume if the owner asks.
    fn file(&self, url: &str, token: Option<&str>, sink: &mut dyn Write) -> Result<u64> {
        let mut req = self.agent.get(url);
        if let Some(t) = token {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        let resp = req.call().with_context(|| format!("GET {url}"))?;
        let mut reader = resp.into_body().into_reader();
        std::io::copy(&mut reader, sink).with_context(|| format!("read {url}"))
    }
}
