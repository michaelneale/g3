use anyhow::Result;
use clap::Parser;
use g3_config::Config;
use g3_core::Agent;
use indicatif::{ProgressBar, ProgressStyle};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

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

    /// Task to execute (if provided, runs in single-shot mode instead of interactive)
    pub task: Option<String>,
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::fmt().with_max_level(level).init();

    info!("Starting G3 AI Coding Agent");

    // Load configuration
    let config = Config::load(cli.config.as_deref())?;

    // Initialize agent
    let mut agent = Agent::new(config).await?;

    // Execute task or start interactive mode
    if let Some(task) = cli.task {
        // Single-shot mode
        info!("Executing task: {}", task);
        let result = agent
            .execute_task_with_timing(&task, None, false, cli.show_prompt, cli.show_code, true)
            .await?;
        println!("{}", result);
    } else {
        // Interactive mode (default)
        info!("Starting interactive mode");
        run_interactive(agent, cli.show_prompt, cli.show_code).await?;
    }

    Ok(())
}

async fn run_interactive(mut agent: Agent, show_prompt: bool, show_code: bool) -> Result<()> {
    println!("🤖 G3 AI Coding Agent - Interactive Mode");
    println!(
        "I solve problems by writing and executing code. Tell me what you need to accomplish!"
    );
    println!();

    // Display provider and model information
    match agent.get_provider_info() {
        Ok((provider, model)) => {
            println!("🔧 Provider: {} | Model: {}", provider, model);
        }
        Err(e) => {
            error!("Failed to get provider info: {}", e);
        }
    }

    println!();
    println!("Type 'exit' or 'quit' to exit, use Up/Down arrows for command history");
    println!();

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

    loop {
        // Display context window progress bar before each prompt
        display_context_progress(&agent);

        let readline = rl.readline("g3> ");
        match readline {
            Ok(line) => {
                let input = line.trim();

                if input == "exit" || input == "quit" {
                    break;
                }

                if input.is_empty() {
                    continue;
                }

                // Add to history
                rl.add_history_entry(input)?;

                // Create cancellation token for this request
                let cancellation_token = CancellationToken::new();
                let cancel_token_clone = cancellation_token.clone();

                // Spawn a task to monitor for ESC key during execution
                let esc_monitor = tokio::spawn(async move {
                    // This is a simplified approach - in a real implementation,
                    // we'd need to handle raw terminal input to detect ESC
                    // For now, we'll just provide the cancellation infrastructure
                    tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
                });

                // Execute task with cancellation support
                let execution_result = tokio::select! {
                    result = agent.execute_task_with_timing_cancellable(
                        input, None, false, show_prompt, show_code, true, cancellation_token
                    ) => {
                        esc_monitor.abort();
                        result
                    }
                    _ = tokio::signal::ctrl_c() => {
                        cancel_token_clone.cancel();
                        esc_monitor.abort();
                        println!("\n⚠️  Operation cancelled by user (Ctrl+C)");
                        continue;
                    }
                };

                match execution_result {
                    Ok(response) => println!("{}", response),
                    Err(e) => {
                        if e.to_string().contains("cancelled") {
                            println!("⚠️  Operation cancelled by user");
                        } else {
                            error!("Error: {}", e);
                        }
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
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

    println!("👋 Goodbye!");
    Ok(())
}

fn display_context_progress(agent: &Agent) {
    let context = agent.get_context_window();
    let percentage = context.percentage_used();

    // Create a simple visual progress bar using the requested characters (10 dots max)
    let bar_width = 10;
    let filled_width = ((percentage / 100.0) * bar_width as f32) as usize;
    let empty_width = bar_width - filled_width;

    let filled_chars = "●".repeat(filled_width);
    let empty_chars = "○".repeat(empty_width);
    let progress_bar = format!("{}{}", filled_chars, empty_chars);

    // Print context info with visual progress bar
    println!(
        "Context: {} {:.1}% | {}/{} tokens",
        progress_bar, percentage, context.used_tokens, context.total_tokens
    );
}
