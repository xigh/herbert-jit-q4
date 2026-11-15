//! The model's own chat template (`chat_template.jinja`, else `chat_template`
//! of `tokenizer_config.json`), rendered by minijinja with Python's string and
//! dict methods (`.get`, `.split`, `.startswith`, `.rstrip`…): the prompt the
//! model's authors wrote, tools, thinking and tool responses included, as
//! llama-server renders it with `--jinja`. Tool call arguments a client sends
//! as a JSON string are parsed first, as llama-server does.
//!
//! What each family's template needs besides:
//!
//! - Gemma 4: `bos_token` and `eos_token`;
//! - Qwen3.5: `tojson` as Python's `json.dumps` writes it (what transformers' templates call): `",
//!   "` and `": "` separators, the keys in their order, non-ASCII kept, nothing escaped for HTML
//!   (minijinja's own escapes `<`, `>`, `&`, `'`); the template walks the arguments with `|items`.

use crate::error::{Error, Result};
use crate::Family;
use minijinja::{Environment, Value as JValue};
use serde_json::Value;
use std::path::Path;

pub struct Template {
    env: Environment<'static>,
    family: Family,
    pub source: String,
}

/// `json.dumps(v, ensure_ascii=False)`: Python's default separators.
pub fn py_json(v: &Value) -> String {
    let mut out = String::new();
    write_py(v, &mut out);
    out
}

/// A value as Python's `str()` writes it (what jinja2's `string` filter gives a mapping or a list):
/// strings in single quotes (double when they hold a single quote and no double one), `True`,
/// `False`, `None`, `{'k': v}`, `[a, b]`.
pub fn py_repr(v: &Value) -> String {
    let mut out = String::new();
    write_repr(v, &mut out);
    out
}

fn write_repr(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("None"),
        Value::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_repr_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_repr(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_repr_str(k, out);
                out.push_str(": ");
                write_repr(x, out);
            }
            out.push('}');
        }
    }
}

fn write_repr_str(s: &str, out: &mut String) {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(q);
}

/// [`py_json`] as Python's `json.dumps(v, indent=n, ensure_ascii=False)` writes it: one item a
/// line, `n` spaces a level, `,` at the end of a line and `": "` after a key; an empty list or
/// object stays `[]` or `{}`.
pub fn py_json_indent(v: &Value, n: usize) -> String {
    let mut out = String::new();
    write_py_indent(v, n, 0, &mut out);
    out
}

fn write_py_indent(v: &Value, n: usize, level: usize, out: &mut String) {
    let pad = |out: &mut String, l: usize| {
        out.push('\n');
        out.push_str(&" ".repeat(n * l));
    };
    match v {
        Value::Array(a) if !a.is_empty() => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, level + 1);
                write_py_indent(x, n, level + 1, out);
            }
            pad(out, level);
            out.push(']');
        }
        Value::Object(m) if !m.is_empty() => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, level + 1);
                write_py_str(k, out);
                out.push_str(": ");
                write_py_indent(x, n, level + 1, out);
            }
            pad(out, level);
            out.push('}');
        }
        other => write_py(other, out),
    }
}

fn write_py(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_py_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_str(k, out);
                out.push_str(": ");
                write_py(x, out);
            }
            out.push('}');
        }
    }
}

fn write_py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Today's date (UTC) as Python's `strftime` writes it, for the directives `%Y`, `%m`, `%d`, `%B`,
/// `%b` and `%%`; any other directive is kept as it is.
fn strftime_now(format: &str) -> String {
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
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = secs.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&year.to_string()),
            Some('m') => out.push_str(&format!("{month:02}")),
            Some('d') => out.push_str(&format!("{day:02}")),
            Some('B') => out.push_str(MONTHS[(month - 1) as usize]),
            Some('b') => out.push_str(&MONTHS[(month - 1) as usize][..3]),
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

/// The template without its `{% generation %}` and `{% endgeneration %}` tags (any whitespace
/// control).
fn strip_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(at) = rest.find("{%") {
        let Some(end) = rest[at..].find("%}") else {
            break;
        };
        let tag = &rest[at + 2..at + end];
        let word = tag.trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        out.push_str(&rest[..at]);
        if word != "generation" && word != "endgeneration" {
            out.push_str(&rest[at..at + end + 2]);
        }
        rest = &rest[at + end + 2..];
    }
    out.push_str(rest);
    out
}

impl Template {
    pub fn load(model_dir: &Path, family: Family) -> Result<Template> {
        let source = match std::fs::read_to_string(model_dir.join("chat_template.jinja")) {
            Ok(s) => s,
            Err(_) => {
                let cfg: Value = serde_json::from_slice(&std::fs::read(
                    model_dir.join("tokenizer_config.json"),
                )?)?;
                cfg["chat_template"]
                    .as_str()
                    .ok_or_else(|| {
                        Error("Model loading error: no chat template in the model directory".into())
                    })?
                    .to_string()
            }
        };
        Self::from_source(source, family)
    }

    pub fn from_source(source: String, family: Family) -> Result<Template> {
        let mut env = Environment::new();
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
        if matches!(
            family,
            Family::Qwen
                | Family::Qwen2
                | Family::Granite
                | Family::Mistral
                | Family::Lfm2
                | Family::Lfm3
                | Family::Smol
        ) {
            env.add_filter(
                "tojson",
                |v: JValue,
                 kwargs: minijinja::value::Kwargs|
                 -> std::result::Result<JValue, minijinja::Error> {
                    let j = serde_json::to_value(&v).map_err(|e| {
                        minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            format!("tojson: {e}"),
                        )
                    })?;
                    // Granite 3's template writes the tools with `tojson(indent=4)`.
                    let indent: Option<usize> = kwargs.get("indent")?;
                    kwargs.assert_all_used()?;
                    Ok(JValue::from_safe_string(match indent {
                        Some(n) => py_json_indent(&j, n),
                        None => py_json(&j),
                    }))
                },
            );
        }
        // SmolLM3 writes its tools with `tool | string`: Python's `str()` of a mapping.
        if family == Family::Smol {
            env.add_filter(
                "string",
                |v: JValue| -> std::result::Result<JValue, minijinja::Error> {
                    if v.as_str().is_some() {
                        return Ok(v);
                    }
                    let j = serde_json::to_value(&v).map_err(|e| {
                        minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            format!("string: {e}"),
                        )
                    })?;
                    Ok(JValue::from(py_repr(&j)))
                },
            );
        }
        // Granite's template writes today's date (`strftime_now('%B %d, %Y')`), as transformers
        // renders it.
        env.add_function("strftime_now", |format: String| -> String {
            strftime_now(&format)
        });
        // LFM's and SmolLM3's templates mark the assistant's text with `{% generation %}`
        // (transformers' extension for training masks), which writes nothing and which
        // minijinja does not know: the tags are taken out.
        let source = if matches!(family, Family::Lfm2 | Family::Lfm3 | Family::Smol) {
            strip_generation_tags(&source)
        } else {
            source
        };
        env.add_template_owned("chat", source.clone())
            .map_err(|e| Error(format!("Model loading error: chat template: {e:#}")))?;
        Ok(Template {
            env,
            family,
            source,
        })
    }

    /// The prompt of `messages` (OpenAI format) with `tools`.
    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        enable_thinking: bool,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut messages = messages.clone();
        if let Some(list) = messages.as_array_mut() {
            for m in list {
                // LFM3-VL's template writes its image placeholder for a part `{"type": "image"}`
                // only: OpenAI's `image_url` parts are given that type.
                if self.family == Family::Lfm3 {
                    for part in m
                        .get_mut("content")
                        .and_then(Value::as_array_mut)
                        .into_iter()
                        .flatten()
                    {
                        if part.get("type").and_then(Value::as_str) == Some("image_url") {
                            *part = serde_json::json!({ "type": "image" });
                        }
                    }
                }
                if let Some(calls) = m.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    for c in calls {
                        if let Some(args) = c.pointer_mut("/function/arguments") {
                            if let Some(s) = args.as_str() {
                                if let Ok(v) = serde_json::from_str::<Value>(s) {
                                    *args = v;
                                }
                            }
                        }
                    }
                }
            }
        }
        let tools = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
        let ctx = match self.family {
            Family::Gemma => minijinja::context! {
                messages => JValue::from_serialize(&messages),
                tools => tools.map(JValue::from_serialize),
                bos_token => "<bos>",
                eos_token => "<eos>",
                add_generation_prompt => add_generation_prompt,
                enable_thinking => enable_thinking,
            },
            Family::Granite => minijinja::context! {
                messages => JValue::from_serialize(&messages),
                tools => tools.map(JValue::from_serialize),
                add_generation_prompt => add_generation_prompt,
                controls => JValue::from_serialize(serde_json::json!({})),
            },
            Family::Mistral => minijinja::context! {
                messages => JValue::from_serialize(&messages),
                tools => tools.map(JValue::from_serialize),
                bos_token => "<s>",
                eos_token => "</s>",
                add_generation_prompt => add_generation_prompt,
            },
            Family::Smol => minijinja::context! {
                messages => JValue::from_serialize(&messages),
                tools => tools.map(JValue::from_serialize),
                add_generation_prompt => add_generation_prompt,
                enable_thinking => enable_thinking,
            },
            Family::Lfm2 | Family::Lfm3 => minijinja::context! {
                messages => JValue::from_serialize(&messages),
                tools => tools.map(JValue::from_serialize),
                bos_token => "<|startoftext|>",
                eos_token => "<|im_end|>",
                add_generation_prompt => add_generation_prompt,
                enable_thinking => enable_thinking,
            },
            Family::Qwen | Family::Qwen2 => minijinja::context! {
                messages => JValue::from_serialize(&messages),
                tools => tools.map(JValue::from_serialize),
                add_generation_prompt => add_generation_prompt,
                enable_thinking => enable_thinking,
            },
        };
        let t = self
            .env
            .get_template("chat")
            .map_err(|e| Error(format!("Backend error: chat template: {e:#}")))?;
        t.render(ctx)
            .map_err(|e| Error(format!("Backend error: chat template: {e:#}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_as_python_writes_it() {
        let v: Value = serde_json::from_str(r#"{"type": "function", "function": {"name": "f", "description": "a <b> & 'c' — é", "n": [1, 2.5, true, null]}}"#).unwrap();
        assert_eq!(
            py_json(&v),
            r#"{"type": "function", "function": {"name": "f", "description": "a <b> & 'c' — é", "n": [1, 2.5, true, null]}}"#
        );
        assert_eq!(py_json(&json!("l1\nl2\t\"q\"")), r#""l1\nl2\t\"q\"""#);
    }

    /// Needs Qwen3.5's template ([`crate::test_model_file`]).
    #[test]
    #[ignore = "needs Qwen3.5's chat_template.jinja (HERBERT_MODELS)"]
    fn qwen_template_renders_tools_calls_and_thinking() {
        let template = crate::test_model_file(
            &[
                "Qwen3.5-9B",
                "Qwen3.5-0.8B",
                "Qwen3.5-2B",
                "Qwen3.5-4B",
                "qwen/Qwen3.5-9B",
                "qwen/Qwen3.5-0.8B",
                "qwen/Qwen3.5-2B",
                "qwen/Qwen3.5-4B",
            ],
            "chat_template.jinja",
        );
        let dir = template.parent().unwrap();
        let t = Template::load(dir, Family::Qwen).unwrap();
        let tools = json!([{"type": "function", "function": {"name": "get_datetime", "description": "Date et heure", "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}]);
        let messages = json!([
            {"role": "system", "content": "Tu es utile."},
            {"role": "user", "content": "Quelle heure ?"},
            {"role": "assistant", "content": null, "reasoning_content": "Il faut l'heure.", "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "get_datetime", "arguments": "{\"tz\": \"Europe/Paris\", \"n\": 3}"}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "12:00"},
        ]);
        let p = t.render(&messages, Some(&tools), true, true).unwrap();
        assert!(p.starts_with("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_datetime\""), "{p}");
        assert!(p.contains("</IMPORTANT>\n\nTu es utile.<|im_end|>\n<|im_start|>user\nQuelle heure ?<|im_end|>\n"), "{p}");
        assert!(p.contains("<|im_start|>assistant\n<think>\nIl faut l'heure.\n</think>\n\n<tool_call>\n<function=get_datetime>\n<parameter=tz>\nEurope/Paris\n</parameter>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>\n"), "{p}");
        assert!(p.ends_with("<|im_start|>user\n<tool_response>\n12:00\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"), "{p}");
        let q = t
            .render(
                &json!([{"role": "user", "content": "Bonjour"}]),
                None,
                false,
                true,
            )
            .unwrap();
        assert_eq!(
            q,
            "<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
