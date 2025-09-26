use anyhow::Result;
use clap::Parser;
use g3_config::Config;
use g3_core::Agent;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::Write;
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

    /// Enable autonomous mode with coach-player feedback loop
    #[arg(long)]
    pub autonomous: bool,
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

    // Load configuration
    let config = Config::load(cli.config.as_deref())?;

    // Initialize agent
    let mut agent = Agent::new(config).await?;

    // Execute task, autonomous mode, or start interactive mode
    if cli.autonomous {
        // Autonomous mode with coach-player feedback loop
        info!("Starting autonomous mode");
        run_autonomous(agent, cli.show_prompt, cli.show_code).await?;
    } else if let Some(task) = cli.task {
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

                // Show thinking indicator immediately
                print!("🤔 Thinking...");
                std::io::stdout().flush()?;

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

async fn run_autonomous(mut agent: Agent, show_prompt: bool, show_code: bool) -> Result<()> {
    println!("🤖 G3 AI Coding Agent - Autonomous Mode");
    println!("🎯 Looking for requirements.md in current directory...");

    // Check if requirements.md exists
    let requirements_path = std::path::Path::new("requirements.md");
    if !requirements_path.exists() {
        println!("❌ Error: requirements.md not found in current directory");
        println!("   Please create a requirements.md file with your project requirements");
        return Ok(());
    }

    // Read requirements.md
    let requirements = match std::fs::read_to_string(requirements_path) {
        Ok(content) => content,
        Err(e) => {
            println!("❌ Error reading requirements.md: {}", e);
            return Ok(());
        }
    };

    println!("📋 Requirements loaded from requirements.md");
    println!("🔄 Starting coach-player feedback loop...");
    println!();

    const MAX_TURNS: usize = 5;
    let mut turn = 1;
    let mut coach_feedback = String::new();

    loop {
        println!("━━━ Turn {}/{} - Player Mode ━━━", turn, MAX_TURNS);
        
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

        let _player_result = agent
            .execute_task_with_timing(&player_prompt, None, false, show_prompt, show_code, true)
            .await?;

        println!("\n🎯 Player implementation completed");
        println!();

        // Create a new agent instance for coach mode to ensure fresh context
        let config = g3_config::Config::load(None)?;
        let mut coach_agent = Agent::new(config).await?;

        println!("━━━ Turn {}/{} - Coach Mode ━━━", turn, MAX_TURNS);
        
        // Coach mode: critique the implementation
        let coach_prompt = format!(
            "You are G3 in coach mode. Your role is to critique and review implementations against requirements.

REQUIREMENTS:
{}

IMPLEMENTATION REVIEW:
Review the current state of the project and provide a concise critique focusing on:
1. Whether the requirements are correctly implemented
2. What's missing or incorrect
3. Specific improvements needed

If the implementation correctly meets all requirements, respond with: 'IMPLEMENTATION_APPROVED'
If improvements are needed, provide specific actionable feedback.

Keep your response concise and focused on actionable items.",
            requirements
        );

        let coach_result = coach_agent
            .execute_task_with_timing(&coach_prompt, None, false, show_prompt, show_code, true)
            .await?;

        println!("\n🎓 Coach review completed");

        // Check if coach approved the implementation
        if coach_result.contains("IMPLEMENTATION_APPROVED") {
            println!("\n✅ Coach approved the implementation!");
            println!("🎉 Autonomous mode completed successfully");
            break;
        }

        // Check if we've reached max turns
        if turn >= MAX_TURNS {
            println!("\n⏰ Maximum turns ({}) reached", MAX_TURNS);
            println!("🔄 Autonomous mode completed (max iterations)");
            break;
        }

        // Store coach feedback for next iteration
        coach_feedback = coach_result;
        turn += 1;
        
        println!("\n🔄 Coach provided feedback for next iteration");
        println!("📝 Preparing to incorporate feedback in turn {}", turn);
        println!();
    }

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


