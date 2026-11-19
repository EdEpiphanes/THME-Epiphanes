// SPDX-License-Identifier: AGPL-3.0-only
//! Text in and out (THME-3037). Encodes and decodes with a model folder's own `tokenizer.json`,
//! and renders the model's own `chat_template` from `tokenizer_config.json`. Only small text
//! files are read here: no weights are loaded and nothing is downloaded.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

/// A model's tokenizer, loaded from `tokenizer.json`.
#[derive(Debug)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        if !path.is_file() {
            bail!("missing tokenizer.json in {}", dir.display());
        }
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("cannot load {}: {e}", path.display()))?;
        Ok(Self { inner })
    }

    /// `add_special` adds the model's own special tokens (such as BOS) as its template does.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special)
            .map_err(|e| anyhow!("cannot encode text: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Special tokens are left out of the text, so a round trip gives back the words only.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow!("cannot decode ids: {e}"))
    }

    /// The id of a token string in this vocabulary, if it has one.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

/// The `eos_token` string named by a model folder's `tokenizer_config.json`, if any.
pub fn eos_token(dir: &Path) -> Option<String> {
    read_config(dir)
        .ok()
        .map(|config| special_token(&config, "eos_token"))
        .filter(|token| !token.is_empty())
}

/// True when `tokenizer_config.json` names a chat template (a string, or a list with `default`).
pub fn has_chat_template(dir: &Path) -> bool {
    read_config(dir)
        .and_then(|config| template_of(&config))
        .is_ok()
}

/// One chat turn, as the messages file and the chat template see it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Renders the model's `default` chat template for `messages`, as `transformers` would.
pub fn render_chat_template(
    dir: &Path,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    let config = read_config(dir)?;
    let source = template_of(&config)?;
    let mut env = minijinja::Environment::new();
    // The same whitespace rules as Hugging Face templates are written for.
    let mut syntax = minijinja::syntax::SyntaxConfig::builder();
    syntax.trim_blocks(true).lstrip_blocks(true);
    env.set_syntax(
        syntax
            .build()
            .map_err(|e| anyhow!("cannot set template syntax: {e}"))?,
    );
    // Python string methods such as `.strip()` and `.startswith()`.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    // shortcut: strftime_now uses UTC, not the local time zone; upgrade when a template needs it.
    env.add_function(
        "strftime_now",
        |fmt: String| -> std::result::Result<String, minijinja::Error> {
            Ok(format_date(now_secs(), &fmt))
        },
    );
    env.add_template_owned("chat_template", source)
        .map_err(|e| anyhow!("chat_template in tokenizer_config.json does not parse: {e}"))?;
    let template = env
        .get_template("chat_template")
        .map_err(|e| anyhow!("chat_template: {e}"))?;
    let context = minijinja::context! {
        messages => minijinja::Value::from(minijinja::value::Serde(messages)),
        add_generation_prompt => add_generation_prompt,
        bos_token => special_token(&config, "bos_token"),
        eos_token => special_token(&config, "eos_token"),
    };
    template
        .render(context)
        .map_err(|e| anyhow!("chat template failed: {e}"))
}

fn read_config(dir: &Path) -> Result<Value> {
    let path = dir.join("tokenizer_config.json");
    if !path.is_file() {
        bail!("missing tokenizer_config.json in {}", dir.display());
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("{} is not valid JSON", path.display()))
}

/// The chat template text, from a string or from the list of named templates (`default`).
fn template_of(config: &Value) -> Result<String> {
    match config.get("chat_template") {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(entries)) => entries
            .iter()
            .find(|e| e.get("name").and_then(Value::as_str) == Some("default"))
            .and_then(|e| e.get("template"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                anyhow!("chat_template in tokenizer_config.json has no entry named \"default\"")
            }),
        Some(_) => bail!("chat_template in tokenizer_config.json must be a string or a list"),
        None => bail!("tokenizer_config.json has no chat_template"),
    }
}

/// A special token is a string, or an object with its `content`; absent means empty.
fn special_token(config: &Value, key: &str) -> String {
    match config.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(obj @ Value::Object(_)) => obj
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_secs()).ok())
        .unwrap_or(0)
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

const WEEKDAYS: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];

/// A UTC `strftime` for the common specifiers (`%Y %m %d %e %H %M %S %b %B %a %A %%`); any other
/// specifier is copied through unchanged.
fn format_date(secs: i64, fmt: &str) -> String {
    let (y, mo, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    let (hh, mi, ss) = (rem / 3600, rem % 3600 / 60, rem % 60);
    let weekday = (secs.div_euclid(86_400) + 4).rem_euclid(7) as usize;
    let month = (mo - 1) as usize;
    let month_name = MONTHS.get(month).copied().unwrap_or("");
    let weekday_name = WEEKDAYS.get(weekday).copied().unwrap_or("");
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&format!("{y:04}")),
            Some('m') => out.push_str(&format!("{mo:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:>2}")),
            Some('H') => out.push_str(&format!("{hh:02}")),
            Some('M') => out.push_str(&format!("{mi:02}")),
            Some('S') => out.push_str(&format!("{ss:02}")),
            Some('b') => out.push_str(month_name.get(..3).unwrap_or(month_name)),
            Some('B') => out.push_str(month_name),
            Some('a') => out.push_str(weekday_name.get(..3).unwrap_or(weekday_name)),
            Some('A') => out.push_str(weekday_name),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// Days since 1970-01-01 to (year, month 1-12, day 1-31), in the proleptic Gregorian calendar.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

#[cfg(test)]
mod tests;
