// AGENTS.md rule 4: no panicking shortcuts outside tests; return errors with context instead.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unimplemented,
        clippy::todo,
        clippy::unreachable
    )
)]

fn main() -> anyhow::Result<()> {
    thme::cli::run()
}
