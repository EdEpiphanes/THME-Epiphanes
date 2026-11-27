// SPDX-License-Identifier: AGPL-3.0-only
//! Studio tab: TAB 11: Chat with a loaded model (THME-3038). Replies are decoded on a background
//! thread with the reference CPU forward pass, so the window keeps drawing while they run.

use super::*;
use crate::chat::{self as core, Finish, Folder, Reply, Request};
use crate::inference::{CausalLm, DeviceArg};
use crate::text::{ChatMessage, Tokenizer};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

/// Which device the chat's forward pass runs on. CPU is the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ChatDevice {
    #[default]
    Cpu,
    Gpu,
    CpuGpu,
}

impl ChatDevice {
    const ALL: [ChatDevice; 3] = [ChatDevice::Cpu, ChatDevice::Gpu, ChatDevice::CpuGpu];

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU (experimental)",
            Self::CpuGpu => "CPU + GPU (experimental)",
        }
    }

    fn arg(self) -> DeviceArg {
        match self {
            Self::Cpu => DeviceArg::Cpu,
            Self::Gpu => DeviceArg::Gpu,
            Self::CpuGpu => DeviceArg::Auto,
        }
    }

    /// Tooltip for the choice. The GPU timing is the only measurement; it is not a speedup claim.
    fn tooltip(self) -> &'static str {
        match self {
            Self::Cpu => "Reference CPU forward pass (default).",
            Self::Gpu => {
                "Experimental. Measured on SmolLM-135M, 128-token pass, RTX 4060 Ti, 20 November 2026: \
                 4.3 s on the GPU path vs 1.1 s on CPU. GPU is currently slower; no speedup is claimed."
            }
            Self::CpuGpu => {
                "Experimental. GPU takes the tensor formats it supports, CPU does the rest, and CPU \
                 alone if no adapter is found. Not measured yet; the GPU path alone is slower than CPU \
                 (4.3 s vs 1.1 s on SmolLM-135M, 128 tokens, RTX 4060 Ti, 20 November 2026)."
            }
        }
    }
}

/// A model loaded for chat: the weights and the folder's tokenizer, kept between replies.
/// The device is part of the cache key, so a device change reloads the model.
struct Loaded {
    source: PathBuf,
    device: ChatDevice,
    model: Box<dyn CausalLm>,
    tokenizer: Tokenizer,
}

enum Message {
    Progress(usize),
    Done(Arc<Loaded>, Reply),
    Failed(String),
}

struct Session {
    rx: Receiver<Message>,
    cancel: Arc<AtomicBool>,
    tokens: usize,
}

/// Chat state: the turns, the picked model and the reply in progress.
pub(super) struct ChatTab {
    model: Option<usize>,
    device: ChatDevice,
    max_new_tokens: usize,
    plain_text: bool,
    input: String,
    turns: Vec<ChatMessage>,
    folder: Option<(PathBuf, Result<Folder, String>)>,
    loaded: Option<Arc<Loaded>>,
    session: Option<Session>,
    status: String,
}

impl Default for ChatTab {
    fn default() -> Self {
        Self {
            model: None,
            device: ChatDevice::default(),
            max_new_tokens: 64,
            plain_text: false,
            input: String::new(),
            turns: Vec::new(),
            folder: None,
            loaded: None,
            session: None,
            status: String::new(),
        }
    }
}

/// The folder that holds a model's tokenizer: the folder itself, or the folder of a model file.
fn folder_of(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().map(Path::to_path_buf).unwrap_or_default()
    }
}

impl ChatTab {
    fn folder_state(&mut self, dir: &Path) -> Result<Folder, String> {
        if self.folder.as_ref().is_none_or(|(d, _)| d != dir) {
            self.folder = Some((dir.to_path_buf(), core::check_folder(dir)));
        }
        self.folder.as_ref().map_or_else(
            || Err("no model folder".to_string()),
            |(_, state)| state.clone(),
        )
    }

    fn submit(&mut self, ctx: &egui::Context, source: PathBuf) {
        self.turns.push(ChatMessage {
            role: "user".to_string(),
            content: self.input.trim().to_string(),
        });
        self.input.clear();
        let device = self.device;
        let job = Job {
            cached: self
                .loaded
                .take()
                .filter(|l| l.source == source && l.device == device),
            source,
            device,
            messages: self.turns.clone(),
            plain_text: self.plain_text,
            max_new_tokens: self.max_new_tokens,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let worker_cancel = cancel.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let message = match run_job(job, &worker_cancel, &tx, &ctx) {
                Ok((loaded, reply)) => Message::Done(loaded, reply),
                Err(error) => Message::Failed(format!("{error:#}")),
            };
            let _ = tx.send(message);
            ctx.request_repaint();
        });
        self.session = Some(Session {
            rx,
            cancel,
            tokens: 0,
        });
        self.status = "Loading the model and generating...".to_string();
    }

    fn cancel(&self) {
        if let Some(session) = &self.session {
            session.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn poll(&mut self) {
        let Some(mut session) = self.session.take() else {
            return;
        };
        let mut finished = false;
        while let Ok(message) = session.rx.try_recv() {
            match message {
                Message::Progress(tokens) => session.tokens = tokens,
                Message::Done(loaded, reply) => {
                    let device = loaded.device;
                    self.loaded = Some(loaded);
                    if !reply.token_ids.is_empty() {
                        self.turns.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: reply.text.clone(),
                        });
                    }
                    self.status = run_status(&reply, device);
                    finished = true;
                }
                Message::Failed(error) => {
                    self.status = format!("Error: {error}");
                    finished = true;
                }
            }
        }
        if !finished {
            self.session = Some(session);
        }
    }
}

/// Everything the worker needs, moved to its thread.
struct Job {
    source: PathBuf,
    device: ChatDevice,
    cached: Option<Arc<Loaded>>,
    messages: Vec<ChatMessage>,
    plain_text: bool,
    max_new_tokens: usize,
}

fn run_job(
    job: Job,
    cancel: &AtomicBool,
    tx: &Sender<Message>,
    ctx: &egui::Context,
) -> anyhow::Result<(Arc<Loaded>, Reply)> {
    let loaded = match job.cached {
        Some(loaded) => loaded,
        None => {
            let (_, _, model) =
                crate::inference::load_reference_model(&job.source, job.device.arg())?;
            let dir = folder_of(&job.source);
            let tokenizer = Tokenizer::from_model_dir(&dir)?;
            Arc::new(Loaded {
                source: job.source.clone(),
                device: job.device,
                model,
                tokenizer,
            })
        }
    };
    let dir = folder_of(&job.source);
    let request = Request {
        dir: &dir,
        messages: &job.messages,
        plain_text: job.plain_text,
        max_new_tokens: job.max_new_tokens,
        kv_precision: crate::inference::KvPrecision::F32,
        kv_key_precision: None,
        kv_value_precision: None,
    };
    let reply = core::generate(
        loaded.model.as_ref(),
        &loaded.tokenizer,
        &request,
        cancel,
        &mut |tokens| {
            let _ = tx.send(Message::Progress(tokens));
            ctx.request_repaint();
        },
    )?;
    Ok((loaded, reply))
}

fn run_status(reply: &Reply, device: ChatDevice) -> String {
    let stop = match reply.finish {
        Finish::Eos => "end of sequence",
        Finish::MaxTokens => "token limit",
        Finish::LoopBroken => "4-gram loop broken",
        Finish::Cancelled => "cancelled",
    };
    // provenance: chat-speed
    format!(
        "{} tokens in {:.1} s ({:.2} tok/s, {}, {} reference forward pass); stopped: {stop}",
        reply.token_ids.len(),
        reply.wall_secs,
        reply.tokens_per_sec,
        Evidence::Measured.tag(),
        device.label()
    )
}

impl THMEApp {
    /// TAB 11: Chat with a loaded model (THME-3038)
    pub(super) fn render_tab_chat(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();
        let ctx = ui.ctx().clone();
        self.chat.poll();

        ui.heading(
            egui::RichText::new("CHAT WITH THE LOADED MODEL")
                .size(13.0)
                .color(p.text_main),
        );
        manga::glass_frame(&p, p.warning)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Replies come from THME's reference forward pass, which is slow (CPU by default). \
                         Speed is measured on this machine. A mock model has random weights, so its \
                         text is meaningless. No reply here is a measure of quality.",
                    )
                    .size(9.5)
                    .color(p.text_main),
                );
            });
        ui.add_space(6.0);

        let choices: Vec<(usize, String, PathBuf)> = self
            .models
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.source_path.clone().map(|path| (i, m.name.clone(), path)))
            .collect();
        if choices.is_empty() {
            ui.label(
                egui::RichText::new(
                    "No file-backed model is loaded. Use Load Model File... in the side panel, then come back here.",
                )
                .size(10.0)
                .color(p.text_muted),
            );
            return;
        }
        if !choices.iter().any(|c| Some(c.0) == self.chat.model) {
            self.chat.model = choices.first().map(|c| c.0);
        }
        let picked = choices.iter().find(|c| Some(c.0) == self.chat.model);
        let picked_name = picked.map_or("", |c| c.1.as_str()).to_string();
        let picked_path = picked.map(|c| c.2.clone());
        let busy = self.chat.session.is_some();

        ui.horizontal(|ui| {
            ui.label("Model:");
            egui::ComboBox::from_id_salt("chat_model")
                .selected_text(picked_name)
                .show_ui(ui, |ui| {
                    for (i, name, _) in &choices {
                        ui.add_enabled_ui(!busy, |ui| {
                            ui.selectable_value(&mut self.chat.model, Some(*i), name.as_str());
                        });
                    }
                });
            ui.label("Max new tokens:");
            ui.add(egui::Slider::new(&mut self.chat.max_new_tokens, 1..=256));
            ui.label("Device:");
            egui::ComboBox::from_id_salt("chat_device")
                .selected_text(self.chat.device.label())
                .show_ui(ui, |ui| {
                    for choice in ChatDevice::ALL {
                        ui.add_enabled_ui(!busy, |ui| {
                            ui.selectable_value(&mut self.chat.device, choice, choice.label())
                                .on_hover_text(choice.tooltip())
                        });
                    }
                });
        });

        let Some(path) = picked_path else {
            return;
        };
        let dir = folder_of(&path);
        let folder = self.chat.folder_state(&dir);
        let ready = match &folder {
            Err(missing) => {
                ui.label(
                    egui::RichText::new(missing.as_str())
                        .size(10.0)
                        .color(p.danger),
                );
                false
            }
            Ok(Folder::Template) => true,
            Ok(Folder::PlainTextOnly(reason)) => {
                ui.label(
                    egui::RichText::new(format!(
                        "{reason}. Without a chat template, only plain text mode can talk to this model."
                    ))
                    .size(10.0)
                    .color(p.warning),
                );
                ui.checkbox(
                    &mut self.chat.plain_text,
                    "Plain text mode (turns are sent as plain text)",
                );
                self.chat.plain_text
            }
        };

        ui.add_space(6.0);
        egui::ScrollArea::vertical()
            .max_height(320.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if self.chat.turns.is_empty() {
                    ui.label(
                        egui::RichText::new("No messages yet.")
                            .size(10.0)
                            .color(p.text_muted),
                    );
                }
                for turn in &self.chat.turns {
                    let (who, color) = if turn.role == "user" {
                        ("You", p.text_main)
                    } else {
                        ("Model", p.primary)
                    };
                    ui.label(egui::RichText::new(who).strong().size(9.0).color(color));
                    ui.label(
                        egui::RichText::new(&turn.content)
                            .size(10.0)
                            .color(p.text_main),
                    );
                    ui.add_space(4.0);
                }
            });

        ui.add(
            egui::TextEdit::multiline(&mut self.chat.input)
                .desired_rows(2)
                .hint_text("Type a message"),
        );
        ui.horizontal(|ui| {
            let can_send = ready && !busy && !self.chat.input.trim().is_empty();
            if ui
                .add_enabled(can_send, egui::Button::new("Send"))
                .clicked()
            {
                self.chat.submit(&ctx, path.clone());
            }
            if ui.add_enabled(busy, egui::Button::new("Cancel")).clicked() {
                self.chat.cancel();
            }
            if let Some(session) = &self.chat.session {
                ui.label(
                    egui::RichText::new(format!("Generating... {} tokens so far", session.tokens))
                        .size(10.0)
                        .color(p.text_muted),
                );
            }
        });
        if !self.chat.status.is_empty() {
            ui.label(
                egui::RichText::new(&self.chat.status)
                    .size(9.5)
                    .monospace()
                    .color(p.text_main),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_choice_maps_to_device_arg() {
        assert_eq!(ChatDevice::default(), ChatDevice::Cpu);
        assert_eq!(ChatDevice::Cpu.arg(), DeviceArg::Cpu);
        assert_eq!(ChatDevice::Gpu.arg(), DeviceArg::Gpu);
        assert_eq!(ChatDevice::CpuGpu.arg(), DeviceArg::Auto);
    }
}
