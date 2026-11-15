//! The conformance bench's render by herbert-chat (`conformance/`): every template of `templates/`
//! over every conversation of `cases.json`, thinking on then off, written as JSON for `compare.py`
//! to set against the reference (`reference.py`, Python's jinja2 through transformers).
//!
//! usage: cargo run --release --example conformance -- CASES_JSON TEMPLATES_DIR OUT_JSON

use herbert_chat::{Family, Template};
use serde_json::{json, Map, Value};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [cases, dir, out] = &args[..] else {
        eprintln!("usage: conformance CASES_JSON TEMPLATES_DIR OUT_JSON");
        std::process::exit(2);
    };
    let spec: Value =
        serde_json::from_slice(&std::fs::read(cases).expect("cases")).expect("cases.json");
    let mut results = Map::new();
    for (name, t) in spec["templates"].as_object().expect("templates") {
        let family = match t["family"].as_str() {
            Some("qwen") => Family::Qwen,
            Some("qwen2") => Family::Qwen2,
            Some("granite") => Family::Granite,
            Some("mistral") => Family::Mistral,
            Some("lfm2") => Family::Lfm2,
            Some("lfm3") => Family::Lfm3,
            Some("smol") => Family::Smol,
            _ => Family::Gemma,
        };
        let source = std::fs::read_to_string(format!("{dir}/{name}.jinja")).expect("template");
        let template = match Template::from_source(source, family) {
            Ok(t) => t,
            Err(e) => {
                results.insert(format!("{name}|*"), json!({ "error": e.to_string() }));
                continue;
            }
        };
        for case in spec["cases"].as_array().expect("cases") {
            let tools = case["tools"].as_str().map(|k| &spec["tools"][k]);
            for thinking in spec["thinking"].as_array().expect("thinking") {
                let thinking = thinking.as_bool().unwrap_or(false);
                let key = format!(
                    "{name}|{}|{}",
                    case["id"].as_str().unwrap_or("?"),
                    if thinking { "think" } else { "nothink" }
                );
                let r = template.render(
                    &case["messages"],
                    tools,
                    thinking,
                    case["generation_prompt"].as_bool().unwrap_or(true),
                );
                results.insert(
                    key,
                    match r {
                        Ok(s) => json!({ "ok": s }),
                        Err(e) => json!({ "error": e.to_string() }),
                    },
                );
            }
        }
    }
    std::fs::write(
        out,
        serde_json::to_string_pretty(&Value::Object(results)).expect("json"),
    )
    .expect("write");
    eprintln!("{out}: written");
}
