# Paper v5.1 notes: where the text and the code differ

Working notes for the next revision of the research paper. The paper itself is not in this
repository; each entry gives the code reference and a replacement sentence.

## Loop breaker (paper section 7.9, THME-3099)

- **Paper says:** a detected loop raises the temperature and resets DoLa.
- **Code does:** generation stops. `NgramLoopBreaker::push_and_check_loop`
  (`src/inference/grounded_decode.rs`) returns `true` when the trailing 4-gram has now
  appeared three times within the last `max_history` 4-grams (256 in `src/chat.rs`; the
  constructor never goes below 16). `chat.rs` then ends the reply with `Finish::LoopBroken`.
  Temperature is not raised and DoLa is not reset.
- **Pinned by tests:** a repeating `1,2,3,4,...` stream fires at the 12th token and not before; a
  stream with no repeated 4-gram never fires
  (`loop_breaker_fires_on_the_third_repeat_of_a_four_gram_and_not_before`,
  `loop_breaker_ignores_a_sequence_with_no_repeated_four_gram`).
- **Suggested sentence:** "When the trailing 4-gram of the generated text has appeared three times in
  the last 256 positions, generation stops and the reply is marked as loop-broken."
