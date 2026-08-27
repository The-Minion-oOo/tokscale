//! LM Studio local-server usage parser.
//!
//! The OpenAI-compatible server writes pretty-printed final responses beneath
//! `~/.lmstudio/server-logs/`. This parser extracts only the response identity,
//! model, local timestamp, and balanced `usage` object. Prompt and response
//! bodies are neither deserialized nor retained.

use super::utils::file_modified_timestamp_ms;
use super::UnifiedMessage;
use crate::TokenBreakdown;
use chrono::{Local, LocalResult, NaiveDateTime, TimeZone};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::Path;

const USAGE_MARKER: &[u8] = b"\"usage\"";

#[derive(Debug, Default, Deserialize)]
struct PromptTokenDetails {
    #[serde(default, alias = "cache_read_tokens")]
    cached_tokens: i64,
    #[serde(default, alias = "cache_write_tokens")]
    cache_creation_input_tokens: i64,
}

#[derive(Debug, Default, Deserialize)]
struct UsagePayload {
    #[serde(default, alias = "promptTokens")]
    prompt_tokens: i64,
    #[serde(default, alias = "completionTokens")]
    completion_tokens: i64,
    #[serde(default, alias = "totalTokens")]
    total_tokens: i64,
    #[serde(default)]
    prompt_tokens_details: PromptTokenDetails,
    #[serde(default)]
    cached_tokens: i64,
    #[serde(default)]
    cache_creation_input_tokens: i64,
}

fn find_bytes(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from > haystack.len().saturating_sub(needle.len()) {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| from + offset)
}

fn skip_ascii_whitespace(bytes: &[u8], mut index: usize) -> usize {
    while bytes
        .get(index)
        .is_some_and(|byte| byte.is_ascii_whitespace())
    {
        index += 1;
    }
    index
}

fn usage_object_start(bytes: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut cursor = from;
    while let Some(marker) = find_bytes(bytes, USAGE_MARKER, cursor) {
        cursor = marker + USAGE_MARKER.len();
        if marker > 0 && bytes[marker - 1] == b'\\' {
            continue;
        }
        let mut value = skip_ascii_whitespace(bytes, cursor);
        if bytes.get(value) != Some(&b':') {
            continue;
        }
        value = skip_ascii_whitespace(bytes, value + 1);
        if bytes.get(value) == Some(&b'{') {
            return Some((marker, value));
        }
    }
    None
}

fn balanced_object_end(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start) != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in bytes.iter().copied().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn json_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut escaped = false;
    for (index, byte) in bytes.iter().copied().enumerate().skip(start + 1) {
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            return Some(index + 1);
        }
    }
    None
}

fn last_json_string_field(bytes: &[u8], field: &[u8]) -> Option<String> {
    let mut marker = Vec::with_capacity(field.len() + 2);
    marker.push(b'"');
    marker.extend_from_slice(field);
    marker.push(b'"');
    let mut cursor = 0usize;
    let mut found = None;
    while let Some(index) = find_bytes(bytes, &marker, cursor) {
        cursor = index + marker.len();
        if index > 0 && bytes[index - 1] == b'\\' {
            continue;
        }
        let mut value = skip_ascii_whitespace(bytes, cursor);
        if bytes.get(value) != Some(&b':') {
            continue;
        }
        value = skip_ascii_whitespace(bytes, value + 1);
        let Some(end) = json_string_end(bytes, value) else {
            continue;
        };
        if let Ok(parsed) = serde_json::from_slice::<String>(&bytes[value..end]) {
            found = Some(parsed);
        }
    }
    found
}

fn last_log_timestamp(bytes: &[u8]) -> Option<i64> {
    let text = String::from_utf8_lossy(bytes);
    let mut remainder = text.as_ref();
    let mut parsed = None;
    while let Some(start) = remainder.find('[') {
        let after = &remainder[start + 1..];
        let Some(end) = after.find(']') else {
            break;
        };
        let candidate = &after[..end];
        if let Ok(naive) = NaiveDateTime::parse_from_str(candidate, "%Y-%m-%d %H:%M:%S") {
            parsed = match Local.from_local_datetime(&naive) {
                LocalResult::Single(value) => Some(value.timestamp_millis()),
                LocalResult::Ambiguous(first, _) => Some(first.timestamp_millis()),
                LocalResult::None => parsed,
            };
        }
        remainder = &after[end + 1..];
    }
    parsed
}

fn non_negative(value: i64) -> i64 {
    value.max(0)
}

fn normalized_tokens(usage: &UsagePayload) -> Option<TokenBreakdown> {
    let prompt = non_negative(usage.prompt_tokens);
    let output = non_negative(usage.completion_tokens);
    let cache_read = non_negative(
        usage
            .prompt_tokens_details
            .cached_tokens
            .max(usage.cached_tokens),
    )
    .min(prompt);
    let cache_write = non_negative(
        usage
            .prompt_tokens_details
            .cache_creation_input_tokens
            .max(usage.cache_creation_input_tokens),
    )
    .min(prompt.saturating_sub(cache_read));
    let total = non_negative(usage.total_tokens).max(prompt.saturating_add(output));
    if total == 0 {
        return None;
    }
    let input = total
        .saturating_sub(output)
        .saturating_sub(cache_read)
        .saturating_sub(cache_write);
    Some(TokenBreakdown {
        input,
        output,
        cache_read,
        cache_write,
        reasoning: 0,
    })
}

fn source_hash(path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    format!("{:x}", hasher.finalize())[..12].to_string()
}

fn fallback_dedup_key(path: &Path, marker: usize, model: &str, tokens: &TokenBreakdown) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(marker.to_le_bytes());
    hasher.update(model.as_bytes());
    for value in [
        tokens.input,
        tokens.output,
        tokens.cache_read,
        tokens.cache_write,
    ] {
        hasher.update(value.to_le_bytes());
    }
    format!("lmstudio:{:x}", hasher.finalize())
}

pub fn parse_lmstudio_file(path: &Path) -> Vec<UnifiedMessage> {
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    let fallback_timestamp = file_modified_timestamp_ms(path);
    let session_id = format!("lmstudio:{}", source_hash(path));
    let mut messages = Vec::new();
    let mut cursor = 0usize;
    let mut metadata_start = 0usize;

    while let Some((marker, object_start)) = usage_object_start(&bytes, cursor) {
        let Some(object_end) = balanced_object_end(&bytes, object_start) else {
            break;
        };
        cursor = object_end;
        let Ok(usage) = serde_json::from_slice::<UsagePayload>(&bytes[object_start..object_end])
        else {
            metadata_start = object_end;
            continue;
        };
        let Some(tokens) = normalized_tokens(&usage) else {
            metadata_start = object_end;
            continue;
        };
        let metadata = &bytes[metadata_start..marker];
        let response_id =
            last_json_string_field(metadata, b"id").filter(|value| value.starts_with("chatcmpl-"));
        let model = last_json_string_field(metadata, b"model")
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "unknown".to_string());
        let timestamp = last_log_timestamp(metadata).unwrap_or(fallback_timestamp);
        if timestamp <= 0 {
            metadata_start = object_end;
            continue;
        }
        let dedup_key = response_id
            .map(|id| format!("lmstudio:{id}"))
            .unwrap_or_else(|| fallback_dedup_key(path, marker, &model, &tokens));
        let mut message = UnifiedMessage::new_with_dedup(
            "lmstudio",
            model,
            "lmstudio",
            session_id.clone(),
            timestamp,
            tokens,
            0.0,
            Some(dedup_key),
        );
        message.mark_provider_reported_cost();
        message.is_turn_start = true;
        messages.push(message);
        metadata_start = object_end;
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn parses_exact_components_and_marks_local_cost_authoritative() {
        let mut file = NamedTempFile::new().unwrap();
        write!(
            file,
            r#"[2026-07-09 10:00:00][INFO][fixture-model]
Final response: {{
  "id": "chatcmpl-fixture",
  "model": "fixture-model",
  "choices": [{{"message": {{"content": "synthetic {{ braces }}"}}}}],
  "usage": {{
    "prompt_tokens": 100,
    "completion_tokens": 12,
    "total_tokens": 112,
    "prompt_tokens_details": {{"cached_tokens": 40, "cache_creation_input_tokens": 10}}
  }}
}}
"#
        )
        .unwrap();
        file.flush().unwrap();

        let messages = parse_lmstudio_file(file.path());
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].client, "lmstudio");
        assert_eq!(messages[0].model_id, "fixture-model");
        assert_eq!(messages[0].tokens.input, 50);
        assert_eq!(messages[0].tokens.output, 12);
        assert_eq!(messages[0].tokens.cache_read, 40);
        assert_eq!(messages[0].tokens.cache_write, 10);
        assert_eq!(messages[0].tokens.total(), 112);
        assert_eq!(messages[0].cost, 0.0);
        assert!(messages[0].has_authoritative_cost());
        assert_eq!(
            messages[0].dedup_key.as_deref(),
            Some("lmstudio:chatcmpl-fixture")
        );
    }

    #[test]
    fn keeps_distinct_identical_usage_and_skips_partial_or_zero_records() {
        let mut file = NamedTempFile::new().unwrap();
        for id in ["chatcmpl-a", "chatcmpl-b"] {
            writeln!(
                file,
                "[2026-07-09 11:00:00][INFO][m]\n{}",
                serde_json::json!({
                    "id": id,
                    "model": "m",
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
                })
            )
            .unwrap();
        }
        writeln!(file, "{{\"usage\":{{\"prompt_tokens\":0}}}}").unwrap();
        write!(file, "{{\"usage\":{{\"prompt_tokens\":9").unwrap();
        file.flush().unwrap();

        let messages = parse_lmstudio_file(file.path());
        assert_eq!(messages.len(), 2);
        assert_ne!(messages[0].dedup_key, messages[1].dedup_key);
        assert_eq!(
            messages
                .iter()
                .map(|message| message.tokens.total())
                .sum::<i64>(),
            20
        );
    }

    #[test]
    fn ignores_response_content_that_looks_like_metadata() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[2026-07-09 11:00:00][INFO][real-model]\n{}",
            serde_json::json!({
                "id": "chatcmpl-real",
                "model": "real-model",
                "choices": [{
                    "message": {
                        "content": r#"{"id":"chatcmpl-fake","model":"fake-model"}"#
                    }
                }],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10
                }
            })
        )
        .unwrap();
        file.flush().unwrap();

        let messages = parse_lmstudio_file(file.path());
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].model_id, "real-model");
        assert_eq!(
            messages[0].dedup_key.as_deref(),
            Some("lmstudio:chatcmpl-real")
        );
    }

    #[test]
    fn uses_reported_total_without_losing_component_closure() {
        let usage = UsagePayload {
            prompt_tokens: 20,
            completion_tokens: 5,
            total_tokens: 30,
            prompt_tokens_details: PromptTokenDetails {
                cached_tokens: 8,
                cache_creation_input_tokens: 2,
            },
            ..UsagePayload::default()
        };
        let tokens = normalized_tokens(&usage).unwrap();
        assert_eq!(tokens.input, 15);
        assert_eq!(tokens.total(), 30);
    }
}
