//! Unsloth Studio inference usage parser.
//!
//! Unsloth Studio stores durable chat and API usage in `studio.db` beneath
//! `$UNSLOTH_STUDIO_HOME` (normally `~/.unsloth/studio`). Internal Studio
//! responses keep usage in assistant-message metadata; authenticated external
//! API requests are copied into the content-free `api_usage_events` table.
//! Neither query selects message content, prompts, or response previews.

use super::utils::{open_readonly_sqlite_opt, sqlite_for_each_row_on, timestamp_secs_to_ms};
use super::UnifiedMessage;
use crate::TokenBreakdown;
use rusqlite::Connection;
use serde::Deserialize;
use std::path::Path;

const CLIENT_ID: &str = "unsloth";
const STUDIO_AGENT: &str = "Unsloth";
const API_AGENT: &str = "Unsloth API";

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContextUsage {
    #[serde(default)]
    prompt_tokens: i64,
    #[serde(default)]
    completion_tokens: i64,
    #[serde(default)]
    total_tokens: i64,
    #[serde(default)]
    cached_tokens: i64,
    #[serde(default)]
    cache_write_tokens: i64,
    #[serde(default)]
    reasoning_tokens: i64,
    #[serde(default)]
    model_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MessageMetadata {
    #[serde(default)]
    context_usage: Option<ContextUsage>,
}

fn non_blank(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn normalized_tokens(
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
    cached_tokens: i64,
    cache_write_tokens: i64,
    reasoning_tokens: i64,
) -> Option<TokenBreakdown> {
    let prompt = prompt_tokens.max(0);
    let completion = completion_tokens.max(0);
    let cache_read = cached_tokens.max(0).min(prompt);
    let cache_write = cache_write_tokens
        .max(0)
        .min(prompt.saturating_sub(cache_read));
    let reasoning = reasoning_tokens.max(0).min(completion);
    let total = total_tokens.max(0).max(prompt.saturating_add(completion));
    if total == 0 {
        return None;
    }

    Some(TokenBreakdown {
        input: total
            .saturating_sub(completion)
            .saturating_sub(cache_read)
            .saturating_sub(cache_write),
        output: completion.saturating_sub(reasoning),
        cache_read,
        cache_write,
        reasoning,
    })
}

fn parse_internal_chat_usage(db_path: &Path, conn: &Connection) -> Vec<UnifiedMessage> {
    // `content_json` is deliberately absent. The metadata block carries exact
    // scalar usage and the model that produced the assistant response.
    let query = r#"
        SELECT
            m.id,
            m.thread_id,
            m.created_at,
            m.metadata_json,
            t.model_id
        FROM chat_messages m
        LEFT JOIN chat_threads t ON t.id = m.thread_id
        WHERE m.role = 'assistant'
        ORDER BY m.created_at, m.id
        "#;

    let mut messages = Vec::new();
    sqlite_for_each_row_on(
        conn,
        db_path,
        query,
        Some("Unsloth Studio chat usage"),
        &mut |row| {
            let message_id: String = row.get(0)?;
            let thread_id: String = row.get(1)?;
            let created_at: i64 = row.get(2)?;
            let metadata_json: Option<String> = row.get(3)?;
            let thread_model: Option<String> = row.get(4)?;

            let Some(metadata) = metadata_json
                .as_deref()
                .and_then(|value| serde_json::from_str::<MessageMetadata>(value).ok())
            else {
                return Ok(());
            };
            let Some(usage) = metadata.context_usage else {
                return Ok(());
            };
            let Some(tokens) = normalized_tokens(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                usage.cached_tokens,
                usage.cache_write_tokens,
                usage.reasoning_tokens,
            ) else {
                return Ok(());
            };
            let timestamp = timestamp_secs_to_ms(created_at as f64);
            if timestamp <= 0 || message_id.trim().is_empty() {
                return Ok(());
            }

            let model = non_blank(usage.model_id)
                .or_else(|| non_blank(thread_model))
                .unwrap_or_else(|| "unknown".to_string());
            let session_id = if thread_id.trim().is_empty() {
                format!("unsloth:chat:{message_id}")
            } else {
                thread_id
            };
            let mut message = UnifiedMessage::new_with_dedup(
                CLIENT_ID,
                model,
                CLIENT_ID,
                session_id,
                timestamp,
                tokens,
                0.0,
                Some(format!("unsloth:chat:{message_id}")),
            );
            message.agent = Some(STUDIO_AGENT.to_string());
            message.is_turn_start = true;
            message.mark_provider_reported_cost();
            messages.push(message);
            Ok(())
        },
    );
    messages
}

fn parse_external_api_usage(db_path: &Path, conn: &Connection) -> Vec<UnifiedMessage> {
    // Older Studio builds do not have this table. Probe silently so their
    // internal chat usage still imports without a warning on every scan.
    let query = r#"
        SELECT
            id,
            endpoint,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            created_at
        FROM api_usage_events
        ORDER BY created_at, id
        "#;

    let mut messages = Vec::new();
    sqlite_for_each_row_on(conn, db_path, query, None, &mut |row| {
        let id: String = row.get(0)?;
        let endpoint: String = row.get(1)?;
        let model: String = row.get(2)?;
        let prompt_tokens: i64 = row.get(3)?;
        let completion_tokens: i64 = row.get(4)?;
        let total_tokens: i64 = row.get(5)?;
        let created_at: i64 = row.get(6)?;

        let Some(tokens) =
            normalized_tokens(prompt_tokens, completion_tokens, total_tokens, 0, 0, 0)
        else {
            return Ok(());
        };
        let timestamp = timestamp_secs_to_ms(created_at as f64);
        if timestamp <= 0 || id.trim().is_empty() {
            return Ok(());
        }

        let model = non_blank(Some(model)).unwrap_or_else(|| "unknown".to_string());
        let mut message = UnifiedMessage::new_with_dedup(
            CLIENT_ID,
            model,
            CLIENT_ID,
            "unsloth:api".to_string(),
            timestamp,
            tokens,
            0.0,
            Some(format!("unsloth:api:{id}")),
        );
        message.agent = Some(API_AGENT.to_string());
        message.session_title = non_blank(Some(endpoint));
        message.is_turn_start = true;
        message.mark_provider_reported_cost();
        messages.push(message);
        Ok(())
    });
    messages
}

/// Parse durable Unsloth Studio inference usage without selecting chat content.
pub fn parse_unsloth_sqlite(db_path: &Path) -> Vec<UnifiedMessage> {
    let Some(conn) = open_readonly_sqlite_opt(db_path) else {
        return Vec::new();
    };

    let mut messages = parse_internal_chat_usage(db_path, &conn);
    messages.extend(parse_external_api_usage(db_path, &conn));
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{params, Connection};

    fn create_database(path: &Path, include_api_usage: bool) -> Connection {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE chat_threads (
                id TEXT PRIMARY KEY,
                model_id TEXT
            );
            CREATE TABLE chat_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                metadata_json TEXT,
                created_at INTEGER NOT NULL
            );
            "#,
        )
        .unwrap();
        if include_api_usage {
            conn.execute_batch(
                r#"
                CREATE TABLE api_usage_events (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );
                "#,
            )
            .unwrap();
        }
        conn
    }

    #[test]
    fn returns_empty_for_missing_database() {
        let dir = tempfile::tempdir().unwrap();
        assert!(parse_unsloth_sqlite(&dir.path().join("missing.db")).is_empty());
    }

    #[test]
    fn parses_internal_chat_and_content_free_api_usage() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("studio.db");
        let conn = create_database(&db_path, true);
        conn.execute(
            "INSERT INTO chat_threads (id, model_id) VALUES (?1, ?2)",
            params!["thread-1", "thread-fallback"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, metadata_json, created_at) VALUES (?1, ?2, 'assistant', ?3, ?4)",
            params![
                "message-1",
                "thread-1",
                r#"{"contextUsage":{"promptTokens":100,"completionTokens":40,"totalTokens":140,"cachedTokens":30,"cacheWriteTokens":10,"reasoningTokens":5,"modelId":"unsloth/Qwen-test"}}"#,
                1_788_000_000_123_i64,
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO api_usage_events (id, subject, endpoint, model, status, prompt_tokens, completion_tokens, total_tokens, created_at) VALUES (?1, ?2, ?3, ?4, 'completed', ?5, ?6, ?7, ?8)",
            params![
                "request-1",
                "private-user",
                "/v1/chat/completions",
                "unsloth/local-api-model",
                20_i64,
                7_i64,
                27_i64,
                1_788_000_100_i64,
            ],
        )
        .unwrap();
        drop(conn);

        let messages = parse_unsloth_sqlite(&db_path);
        assert_eq!(messages.len(), 2);

        let chat = &messages[0];
        assert_eq!(chat.model_id, "unsloth/Qwen-test");
        assert_eq!(chat.session_id, "thread-1");
        assert_eq!(chat.timestamp, 1_788_000_000_123);
        assert_eq!(chat.tokens.input, 60);
        assert_eq!(chat.tokens.output, 35);
        assert_eq!(chat.tokens.cache_read, 30);
        assert_eq!(chat.tokens.cache_write, 10);
        assert_eq!(chat.tokens.reasoning, 5);
        assert_eq!(chat.tokens.total(), 140);
        assert_eq!(chat.dedup_key.as_deref(), Some("unsloth:chat:message-1"));
        assert_eq!(chat.agent.as_deref(), Some(STUDIO_AGENT));
        assert!(chat.is_turn_start);
        assert_eq!(chat.cost, 0.0);

        let api = &messages[1];
        assert_eq!(api.model_id, "unsloth/local-api-model");
        assert_eq!(api.timestamp, 1_788_000_100_000);
        assert_eq!(api.tokens.input, 20);
        assert_eq!(api.tokens.output, 7);
        assert_eq!(api.tokens.total(), 27);
        assert_eq!(api.agent.as_deref(), Some(API_AGENT));
        assert_eq!(api.session_title.as_deref(), Some("/v1/chat/completions"));
        assert_eq!(api.dedup_key.as_deref(), Some("unsloth:api:request-1"));
        assert_eq!(api.session_id, "unsloth:api");
    }

    #[test]
    fn older_schema_without_api_table_still_parses_chat_usage() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("studio.db");
        let conn = create_database(&db_path, false);
        conn.execute(
            "INSERT INTO chat_threads (id, model_id) VALUES ('thread-1', 'fallback-model')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, metadata_json, created_at) VALUES (?1, ?2, 'assistant', ?3, ?4)",
            params![
                "message-1",
                "thread-1",
                r#"{"contextUsage":{"promptTokens":5,"completionTokens":3,"totalTokens":8}}"#,
                1_788_000_000_i64,
            ],
        )
        .unwrap();
        drop(conn);

        let messages = parse_unsloth_sqlite(&db_path);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].model_id, "fallback-model");
        assert_eq!(messages[0].tokens.total(), 8);
    }

    #[test]
    fn skips_user_messages_malformed_metadata_and_zero_usage() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("studio.db");
        let conn = create_database(&db_path, false);
        for (id, role, metadata) in [
            (
                "user-1",
                "user",
                r#"{"contextUsage":{"promptTokens":10,"totalTokens":10}}"#,
            ),
            ("assistant-bad", "assistant", "not-json"),
            (
                "assistant-zero",
                "assistant",
                r#"{"contextUsage":{"promptTokens":0,"completionTokens":0,"totalTokens":0}}"#,
            ),
        ] {
            conn.execute(
                "INSERT INTO chat_messages (id, thread_id, role, metadata_json, created_at) VALUES (?1, 'thread-1', ?2, ?3, ?4)",
                params![id, role, metadata, 1_788_000_000_i64],
            )
            .unwrap();
        }
        drop(conn);

        assert!(parse_unsloth_sqlite(&db_path).is_empty());
    }
}
