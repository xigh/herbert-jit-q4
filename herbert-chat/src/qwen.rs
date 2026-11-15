//! What Qwen3.5 writes, split as llama-server splits it: the thinking block
//! (`<think> … </think>`) into `reasoning_content`, a tool call in Qwen3.5's
//! XML form (`<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>
//! …</function>\n</tool_call>`, the JSON form `{"name": …, "arguments": …}` of
//! Qwen3 accepted as well) into an OpenAI `tool_calls` entry with JSON
//! arguments, the rest into `content`, trimmed at both ends. The four markers
//! are single special tokens: the split is made on the ids, the text of each
//! part decoded as it streams.
//!
//! A parameter's value is a string when the tool's schema says `string` (the
//! template writes strings as they are); otherwise it is read as JSON (numbers,
//! booleans, objects and arrays, which the template wrote with `tojson`), and
//! kept as a string when it is not JSON.

use crate::detok::DetokenizationFlow;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

/// The ids of the markers (`chat::Tokens` reads them from the tokenizer).
#[derive(Clone, Copy, Debug)]
pub struct Markers {
    pub think_open: u32,
    pub think_close: u32,
    pub call_open: u32,
    pub call_close: u32,
}

pub use crate::Piece;

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Content,
    Reasoning,
    ToolCall,
}

pub struct OutputParser {
    m: Markers,
    tools: Option<Value>,
    state: State,
    content: DetokenizationFlow,
    reasoning: DetokenizationFlow,
    call: Vec<u32>,
    /// Content not sent yet: leading whitespace is dropped, trailing whitespace
    /// is held until something else follows it (and dropped at a tool call or the end).
    started: bool,
    held: String,
    /// Reasoning: the leading newlines dropped, trailing whitespace held the same way.
    r_started: bool,
    r_held: String,
}

impl OutputParser {
    /// A parser for the answer that follows `prompt`: in the thinking block when
    /// the prompt ends inside one (Qwen3.5's generation prompt `<think>\n`).
    /// `tools` (the request's) gives the parameters' types.
    pub fn after_prompt(m: Markers, tools: Option<&Value>, prompt: &[u32]) -> Self {
        let tail = &prompt[prompt.len().saturating_sub(8)..];
        let open = tail.iter().rposition(|&id| id == m.think_open);
        let inside = open.is_some_and(|o| !tail[o..].contains(&m.think_close));
        OutputParser {
            m,
            tools: tools.cloned(),
            state: if inside {
                State::Reasoning
            } else {
                State::Content
            },
            content: DetokenizationFlow::fresh(),
            reasoning: DetokenizationFlow::fresh(),
            call: Vec::new(),
            started: false,
            held: String::new(),
            r_started: false,
            r_held: String::new(),
        }
    }

    fn content_text(&mut self, s: String, out: &mut Vec<Piece>) {
        if s.is_empty() {
            return;
        }
        let s = if self.started {
            s
        } else {
            s.trim_start().to_string()
        };
        if s.is_empty() {
            return;
        }
        self.started = true;
        let body = s.trim_end();
        if body.is_empty() {
            self.held.push_str(&s);
            return;
        }
        let tail = &s[body.len()..];
        out.push(Piece::Content(format!(
            "{}{body}",
            std::mem::take(&mut self.held)
        )));
        self.held.push_str(tail);
    }

    fn reasoning_text(&mut self, s: String, out: &mut Vec<Piece>) {
        if s.is_empty() {
            return;
        }
        let s = if self.r_started {
            s
        } else {
            s.trim_start_matches('\n').to_string()
        };
        if s.is_empty() {
            return;
        }
        self.r_started = true;
        let body = s.trim_end();
        if body.is_empty() {
            self.r_held.push_str(&s);
            return;
        }
        let tail = &s[body.len()..];
        out.push(Piece::Reasoning(format!(
            "{}{body}",
            std::mem::take(&mut self.r_held)
        )));
        self.r_held.push_str(tail);
    }

    fn flush(&mut self, tok: &Tokenizer, out: &mut Vec<Piece>) {
        let c = self.content.balance(tok);
        self.content_text(c, out);
        let r = self.reasoning.balance(tok);
        self.reasoning_text(r, out);
    }

    /// One generated token; what it adds to the answer.
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> Vec<Piece> {
        let mut out = Vec::new();
        let m = self.m;
        match self.state {
            State::Content | State::Reasoning if id == m.think_open => {
                self.flush(tok, &mut out);
                self.state = State::Reasoning;
            }
            State::Reasoning if id == m.think_close => {
                self.flush(tok, &mut out);
                self.r_held.clear();
                self.state = State::Content;
            }
            State::Content if id == m.think_close => {}
            State::Content | State::Reasoning if id == m.call_open => {
                self.flush(tok, &mut out);
                self.held.clear();
                self.r_held.clear();
                self.call.clear();
                self.state = State::ToolCall;
            }
            State::ToolCall
                if id == m.call_close
                    && tok
                        .decode(&self.call, false)
                        .unwrap_or_default()
                        .trim_start()
                        .starts_with('[') =>
            {
                // LFM2: a Python list of calls, `[f(a='x'), g()]`.
                let text = tok.decode(&self.call, false).unwrap_or_default();
                match parse_python_calls(&text) {
                    Some(calls) => {
                        out.extend(calls.into_iter().map(|(name, args)| Piece::ToolCall {
                            name,
                            arguments: args.to_string(),
                        }))
                    }
                    None => self.content_text(
                        format!("<|tool_call_start|>{text}<|tool_call_end|>"),
                        &mut out,
                    ),
                }
                self.state = State::Content;
            }
            State::ToolCall if id == m.call_close => {
                let text = tok.decode(&self.call, false).unwrap_or_default();
                match parse_call(&text, self.tools.as_ref()) {
                    Some((name, args)) => out.push(Piece::ToolCall {
                        name,
                        arguments: args.to_string(),
                    }),
                    // Not a call we can read: handed back as text, as written.
                    None => {
                        let (open, close) = (marker(tok, m.call_open), marker(tok, m.call_close));
                        self.content_text(format!("{open}{text}{close}"), &mut out)
                    }
                }
                self.state = State::Content;
            }
            // Ministral 3: a call is closed by the next one.
            State::ToolCall if id == m.call_open => {
                let text = tok.decode(&self.call, false).unwrap_or_default();
                match parse_call(&text, self.tools.as_ref()) {
                    Some((name, args)) => out.push(Piece::ToolCall {
                        name,
                        arguments: args.to_string(),
                    }),
                    None => {
                        let open = marker(tok, m.call_open);
                        self.content_text(format!("{open}{text}"), &mut out)
                    }
                }
                self.call.clear();
            }
            State::ToolCall => self.call.push(id),
            State::Content => {
                let s = self.content.push(tok, id);
                self.content_text(s, &mut out);
            }
            State::Reasoning => {
                let s = self.reasoning.push(tok, id);
                self.reasoning_text(s, &mut out);
            }
        }
        out
    }

    /// The end of the answer: what is still held (trailing whitespace dropped),
    /// an unfinished call as text.
    pub fn finish(&mut self, tok: &Tokenizer) -> Vec<Piece> {
        let mut out = Vec::new();
        self.flush(tok, &mut out);
        if self.state == State::ToolCall && !self.call.is_empty() {
            let text = tok.decode(&self.call, false).unwrap_or_default();
            // A call cut by the budget: readable as it stands, it is a call.
            match parse_call(&text, self.tools.as_ref()) {
                Some((name, args)) => out.push(Piece::ToolCall {
                    name,
                    arguments: args.to_string(),
                }),
                None => {
                    let open = marker(tok, self.m.call_open);
                    self.content_text(format!("{open}{text}"), &mut out)
                }
            }
        }
        out
    }
}

/// The text of a marker token (`<tool_call>`, Ministral's `[TOOL_CALLS]`…): an unreadable call is
/// handed back with the family's own marker.
fn marker(tok: &Tokenizer, id: u32) -> String {
    tok.id_to_token(id).unwrap_or_default()
}

/// The JSON type the request's `tools` declare for parameter `key` of `name`.
fn declared_type(tools: Option<&Value>, name: &str, key: &str) -> Option<String> {
    let list = tools?.as_array()?;
    let f = list
        .iter()
        .map(|t| t.get("function").unwrap_or(t))
        .find(|f| f["name"].as_str() == Some(name))?;
    let ty = &f["parameters"]["properties"][key]["type"];
    match ty {
        Value::String(s) => Some(s.clone()),
        Value::Array(a) => {
            // ["string", "null"]: a string as soon as one of the types is.
            if a.iter().any(|t| t.as_str() == Some("string")) {
                Some("string".into())
            } else {
                a.iter().find_map(|t| t.as_str().map(str::to_string))
            }
        }
        _ => None,
    }
}

fn value_of(raw: &str, ty: Option<&str>) -> Value {
    if ty == Some("string") {
        return Value::String(raw.to_string());
    }
    let t = raw.trim();
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        // Without a schema a bare word that parses (a number) stays a number; a JSON string literal
        // is unwrapped.
        return v;
    }
    match t {
        "True" => Value::Bool(true),
        "False" => Value::Bool(false),
        "None" => Value::Null,
        _ => Value::String(raw.to_string()),
    }
}

/// The inside of `<tool_call>…</tool_call>` → the function's name and its
/// arguments as a JSON object.
pub fn parse_call(text: &str, tools: Option<&Value>) -> Option<(String, Value)> {
    let t = text.trim();
    // Ministral 3: `NAME[ARGS]{…}`.
    if let Some((name, args)) = t.split_once("[ARGS]") {
        let name = name.trim();
        let args: Value = serde_json::from_str(args.trim()).ok()?;
        return (!name.is_empty() && args.is_object()).then(|| (name.to_string(), args));
    }
    if t.starts_with('{') {
        let v: Value = serde_json::from_str(t).ok()?;
        let name = v["name"].as_str()?.to_string();
        let args = match &v["arguments"] {
            Value::String(s) => serde_json::from_str(s).ok()?,
            Value::Null => Value::Object(Map::new()),
            a => a.clone(),
        };
        return args.is_object().then_some((name, args));
    }
    let rest = t.strip_prefix("<function=")?;
    let close = rest.find('>')?;
    let name = rest[..close].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut body = &rest[close + 1..];
    let mut args = Map::new();
    while let Some(at) = body.find("<parameter=") {
        let after = &body[at + "<parameter=".len()..];
        let close = after.find('>')?;
        let key = after[..close].trim().to_string();
        let value_start = &after[close + 1..];
        // The value runs to `</parameter>`, or to the next parameter or the end of the function
        // when the model left it open.
        let end = value_start
            .find("</parameter>")
            .or_else(|| value_start.find("<parameter="))
            .or_else(|| value_start.find("</function>"))
            .unwrap_or(value_start.len());
        let mut raw = &value_start[..end];
        raw = raw.strip_prefix('\n').unwrap_or(raw);
        raw = raw.strip_suffix('\n').unwrap_or(raw);
        let ty = declared_type(tools, &name, &key);
        args.insert(key, value_of(raw, ty.as_deref()));
        body = &value_start[end..];
        body = body.strip_prefix("</parameter>").unwrap_or(body);
    }
    Some((name, Value::Object(args)))
}

/// LFM2's calls, a Python list of calls with keyword arguments: `[get_weather(city='Paris',
/// days=3), now()]`. The values are Python literals (strings in single or double quotes with
/// backslash escapes, numbers, `True`, `False`, `None`) or JSON (lists and objects, as the template
/// writes them with `tojson`).
pub fn parse_python_calls(text: &str) -> Option<Vec<(String, Value)>> {
    let s: Vec<char> = text.trim().chars().collect();
    let mut i = 0usize;
    let skip = |i: &mut usize| {
        while *i < s.len() && s[*i].is_whitespace() {
            *i += 1;
        }
    };
    if s.first() != Some(&'[') {
        return None;
    }
    i += 1;
    let mut calls = Vec::new();
    loop {
        skip(&mut i);
        if s.get(i) == Some(&']') {
            break;
        }
        let start = i;
        while i < s.len() && (s[i].is_alphanumeric() || s[i] == '_' || s[i] == '.' || s[i] == '-') {
            i += 1;
        }
        let name: String = s[start..i].iter().collect();
        skip(&mut i);
        if name.is_empty() || s.get(i) != Some(&'(') {
            return None;
        }
        i += 1;
        let mut args = Map::new();
        loop {
            skip(&mut i);
            if s.get(i) == Some(&')') {
                i += 1;
                break;
            }
            let start = i;
            while i < s.len() && (s[i].is_alphanumeric() || s[i] == '_') {
                i += 1;
            }
            let key: String = s[start..i].iter().collect();
            skip(&mut i);
            if key.is_empty() || s.get(i) != Some(&'=') {
                return None;
            }
            i += 1;
            skip(&mut i);
            let value = python_value(&s, &mut i)?;
            args.insert(key, value);
            skip(&mut i);
            match s.get(i) {
                Some(',') => i += 1,
                Some(')') => {}
                _ => return None,
            }
        }
        calls.push((name, Value::Object(args)));
        skip(&mut i);
        match s.get(i) {
            Some(',') => i += 1,
            Some(']') => break,
            _ => return None,
        }
    }
    (!calls.is_empty()).then_some(calls)
}

/// One Python literal or JSON value at `s[*i..]`.
fn python_value(s: &[char], i: &mut usize) -> Option<Value> {
    match *s.get(*i)? {
        q @ ('\'' | '"') => {
            *i += 1;
            let mut out = String::new();
            loop {
                let c = *s.get(*i)?;
                *i += 1;
                if c == q {
                    return Some(Value::String(out));
                }
                if c == '\\' {
                    let e = *s.get(*i)?;
                    *i += 1;
                    out.push(match e {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        other => other,
                    });
                } else {
                    out.push(c);
                }
            }
        }
        '[' | '{' => {
            // JSON up to the matching bracket, strings respected.
            let (open, close) = if s[*i] == '[' { ('[', ']') } else { ('{', '}') };
            let start = *i;
            let (mut depth, mut quote) = (0i32, None::<char>);
            while *i < s.len() {
                let c = s[*i];
                match quote {
                    Some(_) if c == '\\' => *i += 1,
                    Some(q) if c == q => quote = None,
                    Some(_) => {}
                    None if c == '"' || c == '\'' => quote = Some(c),
                    None if c == open => depth += 1,
                    None if c == close => {
                        depth -= 1;
                        if depth == 0 {
                            *i += 1;
                            let raw: String = s[start..*i].iter().collect();
                            return serde_json::from_str(&raw)
                                .ok()
                                .or_else(|| serde_json::from_str(&raw.replace('\'', "\"")).ok());
                        }
                    }
                    None => {}
                }
                *i += 1;
            }
            None
        }
        _ => {
            let start = *i;
            while *i < s.len() && !matches!(s[*i], ',' | ')') {
                *i += 1;
            }
            let raw: String = s[start..*i].iter().collect::<String>().trim().to_string();
            Some(match raw.as_str() {
                "True" | "true" => Value::Bool(true),
                "False" | "false" => Value::Bool(false),
                "None" | "null" => Value::Null,
                _ => serde_json::from_str(&raw).ok()?,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn calls_as_python_lists() {
        assert_eq!(
            parse_python_calls("[get_weather(city='Paris', days=3, units=\"metric\"), now()]"),
            Some(vec![
                (
                    "get_weather".into(),
                    json!({"city": "Paris", "days": 3, "units": "metric"})
                ),
                ("now".into(), json!({}))
            ])
        );
        assert_eq!(
            parse_python_calls(
                "[f(text='l\\'eau\\nfroide', ok=True, x=None, l=[1, 2], o={\"a\": 1})]"
            ),
            Some(vec![(
                "f".into(),
                json!({"text": "l'eau\nfroide", "ok": true, "x": null, "l": [1, 2], "o": {"a": 1}})
            )])
        );
        assert_eq!(parse_python_calls("not a list"), None);
    }

    #[test]
    fn calls_in_qwen_xml() {
        let tools = json!([{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "after": {"type": ["string", "null"]}}}}}]);
        assert_eq!(
            parse_call("\n<function=web_search>\n<parameter=query>\nNoël 2026\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n", Some(&tools)),
            Some(("web_search".into(), json!({"query": "Noël 2026", "limit": 5})))
        );
        // A string the schema declares stays a string even when it reads as a number.
        assert_eq!(
            parse_call(
                "<function=web_search>\n<parameter=query>\n2026\n</parameter>\n</function>",
                Some(&tools)
            ),
            Some(("web_search".into(), json!({"query": "2026"})))
        );
        assert_eq!(
            parse_call("<function=get_datetime>\n</function>", None),
            Some(("get_datetime".into(), json!({})))
        );
        assert_eq!(
            parse_call("<function=edit>\n<parameter=edits>\n[{\"old\": \"a\", \"new\": \"b\"}]\n</parameter>\n<parameter=dry>\nfalse\n</parameter>\n<parameter=text>\nligne 1\nligne 2\n</parameter>\n</function>", None),
            Some(("edit".into(), json!({"edits": [{"old": "a", "new": "b"}], "dry": false, "text": "ligne 1\nligne 2"})))
        );
        assert_eq!(
            parse_call("{\"name\": \"f\", \"arguments\": {\"x\": 1}}", None),
            Some(("f".into(), json!({"x": 1})))
        );
        assert_eq!(parse_call("not a call", None), None);
        assert_eq!(
            parse_call("get_weather[ARGS]{\"city\": \"Paris\"}", None),
            Some(("get_weather".into(), json!({"city": "Paris"})))
        );
    }

    /// Needs Qwen3.5's tokenizer ([`crate::test_model_file`]).
    #[test]
    #[ignore = "needs Qwen3.5's tokenizer.json (HERBERT_MODELS)"]
    fn thinking_content_and_a_call_split_on_the_markers() {
        let path = crate::test_model_file(
            &[
                "Qwen3.5-0.8B",
                "Qwen3.5-2B",
                "Qwen3.5-4B",
                "Qwen3.5-9B",
                "qwen/Qwen3.5-0.8B",
                "qwen/Qwen3.5-2B",
                "qwen/Qwen3.5-4B",
                "qwen/Qwen3.5-9B",
            ],
            "tokenizer.json",
        );
        let tok = Tokenizer::from_file(&path).unwrap();
        let id = |s: &str| tok.token_to_id(s).unwrap();
        let m = Markers {
            think_open: id("<think>"),
            think_close: id("</think>"),
            call_open: id("<tool_call>"),
            call_close: id("</tool_call>"),
        };
        let ids = |text: &str| tok.encode(text, false).unwrap().get_ids().to_vec();
        let prompt = ids("<|im_start|>user\nHeure ?<|im_end|>\n<|im_start|>assistant\n<think>\n");
        let mut p = OutputParser::after_prompt(m, None, &prompt);
        let mut pieces = Vec::new();
        for t in ids("Il faut l'heure.\n</think>\n\nJe regarde.\n\n<tool_call>\n<function=get_datetime>\n</function>\n</tool_call>") {
            pieces.extend(p.push(&tok, t));
        }
        pieces.extend(p.finish(&tok));
        let reasoning: String = pieces
            .iter()
            .filter_map(|x| {
                if let Piece::Reasoning(s) = x {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        let content: String = pieces
            .iter()
            .filter_map(|x| {
                if let Piece::Content(s) = x {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(reasoning, "Il faut l'heure.");
        assert_eq!(content, "Je regarde.");
        assert!(
            pieces.contains(&Piece::ToolCall {
                name: "get_datetime".into(),
                arguments: "{}".into()
            }),
            "{pieces:?}"
        );
        let closed = ids("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_eq!(
            OutputParser::after_prompt(m, None, &closed).state,
            State::Content
        );
    }
}
