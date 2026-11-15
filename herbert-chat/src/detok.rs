//! Incremental detokenization of a stream of tokens.
//!
//! Decoding one token at a time loses information: a SentencePiece decoder
//! strips the space that opens a decoded sequence, so `decode(["▁est"])`
//! renders `"est"` and every space between words disappears. The remedy: keep
//! the tokens of the block, decode the whole sequence at each step and emit
//! only the difference. A still-incomplete multi-byte character (half an
//! emoji) decodes to a trailing `U+FFFD`: it is held back until the next
//! token completes it, or the block closes.
//!
//! One flow per block (text, thinking); tokens routed elsewhere (a tool call)
//! do not pass through here. The cost is quadratic in the block length and
//! measured in microseconds, small against the time of a decoded token.

use tokenizers::Tokenizer;

/// Emittable length of `s`: everything except the trailing `U+FFFD`
/// characters, which may still resolve into a real character at the next
/// token.
fn safe_length(s: &str) -> usize {
    s.trim_end_matches('\u{FFFD}').len()
}

/// The detokenization state of one content block.
pub struct DetokenizationFlow {
    tokens_of: Vec<u32>,
    /// Bytes of `decode(&tokens_of)` already emitted.
    emitted: usize,
}

impl DetokenizationFlow {
    pub fn fresh() -> Self {
        Self {
            tokens_of: Vec::new(),
            emitted: 0,
        }
    }

    /// Pushes a token and returns the text it adds, not yet emitted: empty
    /// while a multi-byte character is incomplete.
    ///
    /// The already emitted prefix is assumed to be stable from one step to the
    /// next: true for the usual decoders (`Replace/ByteFallback/Fuse/Strip`,
    /// `ByteLevel`) as long as trailing `U+FFFD` are held back. If a decoder
    /// broke this, `get` returns `None` and nothing is emitted: the
    /// remainder comes out of [`Self::balance`] when the block closes.
    pub fn push(&mut self, tokenizer: &Tokenizer, token: u32) -> String {
        self.tokens_of.push(token);
        let full = tokenizer.decode(&self.tokens_of, true).unwrap_or_default();
        let end = safe_length(&full);
        match full.get(self.emitted..end) {
            Some(fresh) if !fresh.is_empty() => {
                let fresh = fresh.to_string();
                self.emitted = end;
                fresh
            }
            _ => String::new(),
        }
    }

    /// Closes the block: returns the remainder (including a truly
    /// final `U+FFFD`), and resets the stream to zero for the next
    /// block.
    pub fn balance(&mut self, tokenizer: &Tokenizer) -> String {
        if self.tokens_of.is_empty() {
            return String::new();
        }
        let full = tokenizer.decode(&self.tokens_of, true).unwrap_or_default();
        let rest = full.get(self.emitted..).unwrap_or_default().to_string();
        self.tokens_of.clear();
        self.emitted = 0;
        rest
    }
}
