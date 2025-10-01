use anyhow::Result;
use clap::Parser;
use g3_config::Config;
use g3_core::{project::Project, Agent};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

mod tui;
use tui::SimpleOutput;

#[derive(Parser)]
#[command(name = "g3")]
#[command(about = "A modular, composable AI coding agent")]
#[command(version)]
pub struct Cli {
    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,

    /// Show the system prompt being sent to the LLM
    #[arg(long)]
    pub show_prompt: bool,

    /// Show the generated code before execution
    #[arg(long)]
    pub show_code: bool,

    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<String>,

    /// Workspace directory (defaults to current directory)
    #[arg(short, long)]
    pub workspace: Option<PathBuf>,

    /// Task to execute (if provided, runs in single-shot mode instead of interactive)
    pub task: Option<String>,

    /// Enable autonomous mode with coach-player feedback loop
    #[arg(long)]
    pub autonomous: bool,

    /// Maximum number of turns in autonomous mode (default: 5)
    #[arg(long, default_value = "5")]
    pub max_turns: usize,
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging with filtering
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    // Create a filter that suppresses llama_cpp logs unless in verbose mode
    let filter = if cli.verbose {
        EnvFilter::from_default_env()
            .add_directive(format!("{}=debug", env!("CARGO_PKG_NAME")).parse().unwrap())
            .add_directive("g3_core=debug".parse().unwrap())
            .add_directive("g3_cli=debug".parse().unwrap())
            .add_directive("g3_execution=debug".parse().unwrap())
            .add_directive("g3_providers=debug".parse().unwrap())
    } else {
        EnvFilter::from_default_env()
            .add_directive(format!("{}=info", env!("CARGO_PKG_NAME")).parse().unwrap())
            .add_directive("g3_core=info".parse().unwrap())
            .add_directive("g3_cli=info".parse().unwrap())
            .add_directive("g3_execution=info".parse().unwrap())
            .add_directive("g3_providers=info".parse().unwrap())
            .add_directive("llama_cpp=off".parse().unwrap()) // Suppress all llama_cpp logs
            .add_directive("llama=off".parse().unwrap()) // Suppress all llama.cpp logs
    };

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(filter)
        .init();

    info!("Starting G3 AI Coding Agent");

    // Set up workspace directory
    let workspace_dir = if let Some(ws) = cli.workspace {
        ws
    } else if cli.autonomous {
        // For autonomous mode, use G3_WORKSPACE env var or default
        setup_workspace_directory()?
    } else {
        // Default to current directory for interactive/single-shot mode
        std::env::current_dir()?
    };

    // Create project model
    let project = if cli.autonomous {
        Project::new_autonomous(workspace_dir.clone())?
    } else {
        Project::new(workspace_dir.clone())
    };

    // Ensure workspace exists and enter it
    project.ensure_workspace_exists()?;
    project.enter_workspace()?;

    info!("Using workspace: {}", project.workspace().display());

    // Load configuration
    let config = Config::load(cli.config.as_deref())?;

    // Initialize agent
    let mut agent = Agent::new(config).await?;

    // Execute task, autonomous mode, or start interactive mode
    if cli.autonomous {
        // Autonomous mode with coach-player feedback loop
        info!("Starting autonomous mode");
        run_autonomous(
            agent,
            project,
            cli.show_prompt,
            cli.show_code,
            cli.max_turns,
        )
        .await?;
    } else if let Some(task) = cli.task {
        // Single-shot mode
        info!("Executing task: {}", task);
        let output = SimpleOutput::new();
        let result = agent
            .execute_task_with_timing(&task, None, false, cli.show_prompt, cli.show_code, true)
            .await?;
        output.print_markdown(&result);
    } else {
        let output = SimpleOutput::new();
        // Interactive mode (default)
        info!("Starting interactive mode");
        output.print(&format!("📁 Workspace: {}", project.workspace().display()));
        run_interactive(agent, cli.show_prompt, cli.show_code).await?;
    }

    Ok(())
}

async fn run_interactive(mut agent: Agent, show_prompt: bool, show_code: bool) -> Result<()> {
    let output = SimpleOutput::new();

    output.print("");
    output.print("🤖 G3 AI Coding Agent - Interactive Mode");
    output.print(
        "I solve problems by writing and executing code. Tell me what you need to accomplish!"
    );
    output.print("");

    // Display provider and model information
    match agent.get_provider_info() {
        Ok((provider, model)) => {
            output.print(&format!("🔧 Provider: {} | Model: {}", provider, model));
        }
        Err(e) => {
            error!("Failed to get provider info: {}", e);
        }
    }

    output.print("");
    output.print("Type 'exit' or 'quit' to exit, use Up/Down arrows for command history");
    output.print("For multiline input: use \\ at the end of a line to continue");
    output.print("Submit multiline with Enter (without backslash)");
    output.print("");

    // Initialize rustyline editor with history
    let mut rl = DefaultEditor::new()?;

    // Try to load history from a file in the user's home directory
    let history_file = dirs::home_dir().map(|mut path| {
        path.push(".g3_history");
        path
    });

    if let Some(ref history_path) = history_file {
        let _ = rl.load_history(history_path);
    }

    // Track multiline input
    let mut multiline_buffer = String::new();
    let mut in_multiline = false;

    loop {
        // Display context window progress bar before each prompt
        display_context_progress(&agent, &output);

        // Adjust prompt based on whether we're in multi-line mode
        let prompt = if in_multiline { "... > " } else { "g3> " };

        let readline = rl.readline(prompt);
        match readline {
            Ok(line) => {
                let trimmed = line.trim_end();
                
                // Check if line ends with backslash for continuation
                if trimmed.ends_with('\\') {
                    // Remove the backslash and add to buffer
                    let without_backslash = &trimmed[..trimmed.len() - 1];
                    multiline_buffer.push_str(without_backslash);
                    multiline_buffer.push('\n');
                    in_multiline = true;
                    continue;
                }
                
                // If we're in multiline mode and no backslash, this is the final line
                if in_multiline {
                    multiline_buffer.push_str(&line);
                    in_multiline = false;
                    // Process the complete multiline input
                    let input = multiline_buffer.trim().to_string();
                    multiline_buffer.clear();
                    
                    if input.is_empty() {
                        continue;
                    }
                    
                    // Add complete multiline to history
                    rl.add_history_entry(&input)?;
                    
                    if input == "exit" || input == "quit" {
                        break;
                    }
                    
                    // Process the multiline input
                    execute_task(&mut agent, &input, show_prompt, show_code, &output).await;
                } else {
                    // Single line input
                    let input = line.trim().to_string();
                    
                    if input.is_empty() {
                        continue;
                    }
                    
                    if input == "exit" || input == "quit" {
                        break;
                    }
                    
                    // Add to history
                    rl.add_history_entry(&input)?;
                    
                    // Process the single line input
                    execute_task(&mut agent, &input, show_prompt, show_code, &output).await;
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C pressed
                if in_multiline {
                    // Cancel multiline input
                    output.print("Multi-line input cancelled");
                    multiline_buffer.clear();
                    in_multiline = false;
                } else {
                    output.print("CTRL-C");
                }
                continue;
            }
            Err(ReadlineError::Eof) => {
                output.print("CTRL-D");
                break;
            }
            Err(err) => {
                error!("Error: {:?}", err);
                break;
            }
        }
    }

    // Save history before exiting
    if let Some(ref history_path) = history_file {
        let _ = rl.save_history(history_path);
    }

    output.print("👋 Goodbye!");
    Ok(())
}

async fn execute_task(agent: &mut Agent, input: &str, show_prompt: bool, show_code: bool, output: &SimpleOutput) {
    // Show thinking indicator immediately
    output.print("🤔 Thinking...");
    // Note: flush is handled internally by println

    // Create cancellation token for this request
    let cancellation_token = CancellationToken::new();
    let cancel_token_clone = cancellation_token.clone();

    // Execute task with cancellation support
    let execution_result = tokio::select! {
        result = agent.execute_task_with_timing_cancellable(
            input, None, false, show_prompt, show_code, true, cancellation_token
        ) => {
            result
        }
        _ = tokio::signal::ctrl_c() => {
            cancel_token_clone.cancel();
            output.print("\n⚠️  Operation cancelled by user (Ctrl+C)");
            return;
        }
    };

    match execution_result {
        Ok(response) => output.print_markdown(&response),
        Err(e) => {
            if e.to_string().contains("cancelled") {
                output.print("⚠️  Operation cancelled by user");
            } else {
                error!("Error: {}", e);
            }
        }
    }
}

fn display_context_progress(agent: &Agent, output: &SimpleOutput) {
    let context = agent.get_context_window();
    output.print_context(context.used_tokens, context.total_tokens, context.percentage_used());
}

/// Set up the workspace directory for autonomous mode
/// Uses G3_WORKSPACE environment variable or defaults to ~/tmp/workspace
fn setup_workspace_directory() -> Result<PathBuf> {
    let workspace_dir = if let Ok(env_workspace) = std::env::var("G3_WORKSPACE") {
        PathBuf::from(env_workspace)
    } else {
        // Default to ~/tmp/workspace
        let home_dir = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        home_dir.join("tmp").join("workspace")
    };

    // Create the directory if it doesn't exist
    if !workspace_dir.exists() {
        std::fs::create_dir_all(&workspace_dir)?;
        let output = SimpleOutput::new();
        output.print(&format!(
            "📁 Created workspace directory: {}",
            workspace_dir.display()
        ));
    }

    Ok(workspace_dir)
}

// Simplified autonomous mode implementation
async fn run_autonomous(
    mut agent: Agent,
    project: Project,
    show_prompt: bool,
    show_code: bool,
    max_turns: usize,
) -> Result<()> {
    let output = SimpleOutput::new();
    
    output.print("🤖 G3 AI Coding Agent - Autonomous Mode");
    output.print(&format!("📁 Using workspace: {}", project.workspace().display()));
    
    // Check if requirements exist
    if !project.has_requirements() {
        output.print("❌ Error: requirements.md not found in workspace directory");
        output.print("   Please create a requirements.md file with your project requirements at:");
        output.print(&format!("   {}/requirements.md", project.workspace().display()));
        return Ok(());
    }

    // Read requirements
    let requirements = match project.read_requirements()? {
        Some(content) => content,
        None => {
            output.print("❌ Error: Could not read requirements.md");
            return Ok(());
        }
    };

    output.print("📋 Requirements loaded from requirements.md");
    output.print("🔄 Starting coach-player feedback loop...");
    
    let mut turn = 1;
    let mut coach_feedback = String::new();
    let mut implementation_approved = false;

    loop {
        output.print(&format!("\n=== TURN {}/{} - PLAYER MODE ===", turn, max_turns));

        // Player mode: implement requirements (with coach feedback if available)
        let player_prompt = if coach_feedback.is_empty() {
            format!(
                "You are G3 in implementation mode. Read and implement the following requirements:\n\n{}\n\nImplement this step by step, creating all necessary files and code.",
                requirements
            )
        } else {
            format!(
                "You are G3 in implementation mode. You need to address the coach's feedback and improve your implementation.\n\nORIGINAL REQUIREMENTS:\n{}\n\nCOACH FEEDBACK TO ADDRESS:\n{}\n\nPlease make the necessary improvements to address the coach's feedback while ensuring all original requirements are met.",
                requirements, coach_feedback
            )
        };

        output.print("🎯 Starting player implementation...");
        let player_result = agent
            .execute_task_with_timing(&player_prompt, None, false, show_prompt, show_code, true)
            .await;

        if let Err(e) = player_result {
            output.print(&format!("❌ Player implementation failed: {}", e));
        }

        // Create a new agent instance for coach mode to ensure fresh context
        let config = g3_config::Config::load(None)?;
        let mut coach_agent = Agent::new(config).await?;

        // Ensure coach agent is also in the workspace directory
        project.enter_workspace()?;

        output.print(&format!("\n=== TURN {}/{} - COACH MODE ===", turn, max_turns));

        // Coach mode: critique the implementation
        let coach_prompt = format!(
            "You are G3 in coach mode. Your role is to critique and review implementations against requirements.

REQUIREMENTS:
{}

IMPLEMENTATION REVIEW:
Review the current state of the project and provide a concise critique focusing on:
1. Whether the requirements are correctly implemented
2. Whether the project compiles successfully
3. What requirements are missing or incorrect
4. Specific improvements needed to satisfy requirements

If the implementation correctly meets all requirements, respond with: 'IMPLEMENTATION_APPROVED'
If improvements are needed, provide specific actionable feedback. Be thorough but don't be overly critical. APPROVE the
implementation if it doesn't have compile errors, glaring omissions and generally fits the bill.

Keep your response concise and focused on actionable items.",
            requirements
        );

        output.print("🎓 Starting coach review...");
        let coach_result = coach_agent
            .execute_task_with_timing(&coach_prompt, None, false, show_prompt, show_code, true)
            .await?;

        output.print("🎓 Coach review completed");
        output.print(&format!("Coach feedback: {}", coach_result));

        // Check if coach approved the implementation
        if coach_result.contains("IMPLEMENTATION_APPROVED") {
            output.print("\n=== SESSION COMPLETED - IMPLEMENTATION APPROVED ===");
            output.print("✅ Coach approved the implementation!");
            implementation_approved = true;
            break;
        }

        // Check if we've reached max turns
        if turn >= max_turns {
            output.print("\n=== SESSION COMPLETED - MAX TURNS REACHED ===");
            output.print(&format!("⏰ Maximum turns ({}) reached", max_turns));
            break;
        }

        // Store coach feedback for next iteration
        coach_feedback = coach_result;
        turn += 1;

        output.print("🔄 Coach provided feedback for next iteration");
    }

    if implementation_approved {
        output.print("\n🎉 Autonomous mode completed successfully");
    } else {
        output.print("\n🔄 Autonomous mode completed (max iterations)");
    }

    Ok(())
}

use std::io::Write;