use agent_client_protocol::{
    self as acp, Client, SessionNotification,
};
use anyhow::Result;
use g3_core::{Agent, ToolCall};
use g3_providers::{Message, MessageRole};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_util::compat::{TokioAsyncReadCompatExt as _, TokioAsyncWriteCompatExt as _};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

/// Represents a single G3 session for ACP
struct G3Session {
    messages: Vec<Message>,
    tool_call_ids: HashMap<String, String>, // Maps internal tool IDs to ACP tool call IDs
    cancel_token: Option<CancellationToken>, // Active cancellation token for prompt processing
}

/// G3 ACP Agent implementation that exposes G3 agent capabilities via ACP protocol
pub struct G3AcpAgent {
    session_update_tx: mpsc::UnboundedSender<(acp::SessionNotification, oneshot::Sender<()>)>,
    sessions: Arc<Mutex<HashMap<String, G3Session>>>,
    agent: Arc<Mutex<Agent>>, // Shared agent instance
}

/// Format a tool name to be more human-friendly
fn format_tool_name(tool_name: &str) -> String {
    // For simple tools like "shell", "read_file", "write_file", just capitalize
    tool_name
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl G3AcpAgent {
    pub async fn new(
        session_update_tx: mpsc::UnboundedSender<(acp::SessionNotification, oneshot::Sender<()>)>,
        config: g3_config::Config,
    ) -> Result<Self> {
        // Create agent instance
        let agent = Agent::new(config).await?;

        Ok(Self {
            session_update_tx,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            agent: Arc::new(Mutex::new(agent)),
        })
    }

    fn convert_acp_prompt_to_message(&self, prompt: Vec<acp::ContentBlock>) -> Message {
        let mut content = String::new();

        // Process all content blocks from the prompt
        for block in prompt {
            match block {
                acp::ContentBlock::Text(text) => {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(&text.text);
                }
                acp::ContentBlock::Image(_image) => {
                    // TODO: Handle image content when G3 supports it
                    content.push_str("\n[Image content not yet supported]");
                }
                acp::ContentBlock::Resource(resource) => {
                    // Embed resource content as text with context
                    match &resource.resource {
                        acp::EmbeddedResourceResource::TextResourceContents(text_resource) => {
                            let header = format!("\n--- Resource: {} ---\n", text_resource.uri);
                            content.push_str(&header);
                            content.push_str(&text_resource.text);
                            content.push_str("\n---\n");
                        }
                        _ => {
                            // Ignore non-text resources for now
                        }
                    }
                }
                acp::ContentBlock::ResourceLink(_link) => {
                    // TODO: Handle resource links
                    content.push_str("\n[Resource links not yet supported]");
                }
                acp::ContentBlock::Audio(..) => {
                    content.push_str("\n[Audio content not supported]");
                }
            }
        }

        Message {
            role: MessageRole::User,
            content,
        }
    }

    async fn send_notification(&self, notification: SessionNotification) -> Result<(), acp::Error> {
        let (tx, rx) = oneshot::channel();
        self.session_update_tx
            .send((notification, tx))
            .map_err(|_| acp::Error::internal_error())?;
        rx.await.map_err(|_| acp::Error::internal_error())?;
        Ok(())
    }

    async fn handle_tool_call(
        &self,
        tool_call: &ToolCall,
        session_id: &acp::SessionId,
        session: &mut G3Session,
    ) -> Result<(), acp::Error> {
        // Generate ACP tool call ID and track mapping
        let acp_tool_id = format!("tool_{}", uuid::Uuid::new_v4());
        
        // Store the mapping - we'll use the tool name as the internal ID for now
        session
            .tool_call_ids
            .insert(tool_call.tool.clone(), acp_tool_id.clone());

        // Send tool call notification
        self.send_notification(SessionNotification {
            session_id: session_id.clone(),
            update: acp::SessionUpdate::ToolCall(acp::ToolCall {
                id: acp::ToolCallId(acp_tool_id.clone().into()),
                title: format_tool_name(&tool_call.tool),
                kind: acp::ToolKind::default(),
                status: acp::ToolCallStatus::Pending,
                content: Vec::new(),
                locations: Vec::new(),
                raw_input: None,
                raw_output: None,
                meta: None,
            }),
            meta: None,
        })
        .await?;

        Ok(())
    }

    async fn handle_tool_result(
        &self,
        tool_name: &str,
        result: &str,
        session_id: &acp::SessionId,
        session: &mut G3Session,
        success: bool,
    ) -> Result<(), acp::Error> {
        // Look up the ACP tool call ID
        if let Some(acp_tool_id) = session.tool_call_ids.get(tool_name) {
            // Determine status
            let status = if success {
                acp::ToolCallStatus::Completed
            } else {
                acp::ToolCallStatus::Failed
            };

            let content = vec![acp::ToolCallContent::Content {
                content: acp::ContentBlock::Text(acp::TextContent {
                    annotations: None,
                    text: result.to_string(),
                    meta: None,
                }),
            }];

            // Send status update
            self.send_notification(SessionNotification {
                session_id: session_id.clone(),
                update: acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate {
                    id: acp::ToolCallId(acp_tool_id.clone().into()),
                    fields: acp::ToolCallUpdateFields {
                        status: Some(status),
                        content: Some(content),
                        ..Default::default()
                    },
                    meta: None,
                }),
                meta: None,
            })
            .await?;
        }

        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl acp::Agent for G3AcpAgent {
    async fn initialize(
        &self,
        args: acp::InitializeRequest,
    ) -> Result<acp::InitializeResponse, acp::Error> {
        info!("ACP: Received initialize request {:?}", args);

        // Advertise G3's capabilities
        let agent_capabilities = acp::AgentCapabilities {
            load_session: false, // TODO: Implement session persistence
            prompt_capabilities: acp::PromptCapabilities {
                image: false,           // TODO: Add when G3 supports images
                audio: false,           // Audio not supported
                embedded_context: true, // G3 can handle embedded context
                meta: None,
            },
            mcp_capabilities: acp::McpCapabilities {
                http: false,
                sse: false,
                meta: None,
            },
            meta: None,
        };

        Ok(acp::InitializeResponse {
            protocol_version: acp::V1,
            agent_capabilities,
            auth_methods: Vec::new(),
            meta: None,
        })
    }

    async fn authenticate(
        &self,
        args: acp::AuthenticateRequest,
    ) -> Result<acp::AuthenticateResponse, acp::Error> {
        info!("ACP: Received authenticate request {:?}", args);
        Ok(acp::AuthenticateResponse { meta: None })
    }

    async fn new_session(
        &self,
        args: acp::NewSessionRequest,
    ) -> Result<acp::NewSessionResponse, acp::Error> {
        info!("ACP: Received new session request {:?}", args);

        // Generate a unique session ID
        let session_id = uuid::Uuid::new_v4().to_string();

        let session = G3Session {
            messages: Vec::new(),
            tool_call_ids: HashMap::new(),
            cancel_token: None,
        };

        // Store the session
        let mut sessions = self.sessions.lock().await;
        sessions.insert(session_id.clone(), session);

        info!("Created new session with ID: {}", session_id);

        Ok(acp::NewSessionResponse {
            session_id: acp::SessionId(session_id.into()),
            modes: None,
            meta: None,
        })
    }

    async fn load_session(
        &self,
        args: acp::LoadSessionRequest,
    ) -> Result<acp::LoadSessionResponse, acp::Error> {
        info!("ACP: Received load session request {:?}", args);
        // Not supported yet
        Err(acp::Error::method_not_found())
    }

    async fn prompt(&self, args: acp::PromptRequest) -> Result<acp::PromptResponse, acp::Error> {
        info!("ACP: Received prompt request");

        // Get the session
        let session_id = args.session_id.0.to_string();

        // Create cancellation token for this prompt
        let cancel_token = CancellationToken::new();

        // Convert ACP prompt to G3 message
        let user_message = self.convert_acp_prompt_to_message(args.prompt);

        // Add message to session
        {
            let mut sessions = self.sessions.lock().await;
            let session = sessions
                .get_mut(&session_id)
                .ok_or_else(acp::Error::invalid_params)?;

            session.messages.push(user_message);
            session.cancel_token = Some(cancel_token.clone());
        }

        // Execute the task with the agent
        // We'll need to stream the response and handle tool calls
        let task_description = {
            let sessions = self.sessions.lock().await;
            let session = sessions.get(&session_id).unwrap();
            session
                .messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default()
        };

        // Start streaming response
        let mut agent = self.agent.lock().await;
        
        // Execute task (this is a simplified version - in a real implementation,
        // we'd want to properly stream the response chunks)
        let result = tokio::select! {
            result = agent.execute_task_with_timing_cancellable(
                &task_description,
                None,
                false,
                false,
                false,
                false,
                cancel_token.clone(),
            ) => result,
            _ = cancel_token.cancelled() => {
                // Clear the cancel token
                let mut sessions = self.sessions.lock().await;
                if let Some(session) = sessions.get_mut(&session_id) {
                    session.cancel_token = None;
                }
                
                return Ok(acp::PromptResponse {
                    stop_reason: acp::StopReason::Cancelled,
                    meta: None,
                });
            }
        };

        // Clear the cancel token
        {
            let mut sessions = self.sessions.lock().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.cancel_token = None;
            }
        }

        match result {
            Ok(response_text) => {
                // Stream the response text to the client
                self.send_notification(SessionNotification {
                    session_id: args.session_id.clone(),
                    update: acp::SessionUpdate::AgentMessageChunk {
                        content: response_text.clone().into(),
                    },
                    meta: None,
                })
                .await?;

                // Add assistant response to session
                let mut sessions = self.sessions.lock().await;
                if let Some(session) = sessions.get_mut(&session_id) {
                    session.messages.push(Message {
                        role: MessageRole::Assistant,
                        content: response_text,
                    });
                }

                Ok(acp::PromptResponse {
                    stop_reason: acp::StopReason::EndTurn,
                    meta: None,
                })
            }
            Err(e) => {
                error!("Error executing task: {}", e);
                Err(acp::Error::internal_error())
            }
        }
    }

    async fn cancel(&self, args: acp::CancelNotification) -> Result<(), acp::Error> {
        info!("ACP: Received cancel request {:?}", args);

        // Get the session and cancel its active operation
        let session_id = args.session_id.0.to_string();
        let mut sessions = self.sessions.lock().await;

        if let Some(session) = sessions.get_mut(&session_id) {
            if let Some(ref token) = session.cancel_token {
                info!("Cancelling active prompt for session {}", session_id);
                token.cancel();
            }
        } else {
            warn!("Cancel request for non-existent session: {}", session_id);
        }

        Ok(())
    }

    async fn set_session_mode(
        &self,
        _args: acp::SetSessionModeRequest,
    ) -> Result<acp::SetSessionModeResponse, acp::Error> {
        // Not supported
        Err(acp::Error::method_not_found())
    }

    async fn ext_method(
        &self,
        _args: acp::ExtRequest,
    ) -> Result<std::sync::Arc<acp::RawValue>, acp::Error> {
        // Not supported
        Err(acp::Error::method_not_found())
    }

    async fn ext_notification(&self, _args: acp::ExtNotification) -> Result<(), acp::Error> {
        // Not supported
        Ok(())
    }
}

/// Run the ACP agent server on stdio
pub async fn run_acp_server(config: g3_config::Config) -> Result<()> {
    info!("Starting G3 ACP agent server on stdio");
    eprintln!("G3 ACP agent started. Listening on stdio...");

    let outgoing = tokio::io::stdout().compat_write();
    let incoming = tokio::io::stdin().compat();

    // The AgentSideConnection will spawn futures onto our Tokio runtime.
    // LocalSet and spawn_local are used because the futures from the
    // agent-client-protocol crate are not Send.
    let local_set = tokio::task::LocalSet::new();
    local_set
        .run_until(async move {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

            // Start up the G3AcpAgent connected to stdio.
            let agent = G3AcpAgent::new(tx, config)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create ACP agent: {}", e))?;
            let (conn, handle_io) =
                acp::AgentSideConnection::new(agent, outgoing, incoming, |fut| {
                    tokio::task::spawn_local(fut);
                });

            // Kick off a background task to send the agent's session notifications to the client.
            tokio::task::spawn_local(async move {
                while let Some((session_notification, tx)) = rx.recv().await {
                    let result = conn.session_notification(session_notification).await;
                    if let Err(e) = result {
                        error!("ACP session notification error: {}", e);
                        break;
                    }
                    tx.send(()).ok();
                }
            });

            // Run until stdin/stdout are closed.
            handle_io.await
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_tool_name() {
        assert_eq!(format_tool_name("shell"), "Shell");
        assert_eq!(format_tool_name("read_file"), "Read File");
        assert_eq!(format_tool_name("write_file"), "Write File");
        assert_eq!(format_tool_name("final_output"), "Final Output");
    }
}
