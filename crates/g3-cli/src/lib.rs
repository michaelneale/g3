use clap::Parser;
use g3_core::Agent;
use g3_config::Config;
use anyhow::Result;
use tracing::{info, error};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

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
    
    tracing_subscriber::fmt()
        .with_max_level(level)
        .init();
    
    info!("Starting G3 AI Coding Agent");
    
    // Load configuration
    let config = Config::load(cli.config.as_deref())?;
    
    // Initialize agent
    let agent = Agent::new(config).await?;
    
    // Execute task or start interactive mode
    if let Some(task) = cli.task {
        // Single-shot mode
        info!("Executing task: {}", task);
        let result = agent.execute_task_with_timing(&task, None, false, cli.show_prompt, cli.show_code, true).await?;
        println!("{}", result);
    } else {
        // Interactive mode (default)
        info!("Starting interactive mode");
        run_interactive(agent, cli.show_prompt, cli.show_code).await?;
    }
    
    Ok(())
}

async fn run_interactive(agent: Agent, show_prompt: bool, show_code: bool) -> Result<()> {
    println!("🤖 G3 AI Coding Agent - Interactive Mode");
    println!("I solve problems by writing and executing code. Tell me what you need to accomplish!");
    println!();
    println!("Type 'exit' or 'quit' to exit, use Up/Down arrows for command history");
    println!();
    
    // Initialize rustyline editor with history
    let mut rl = DefaultEditor::new()?;
    
    // Try to load history from a file in the user's home directory
    let history_file = dirs::home_dir()
        .map(|mut path| {
            path.push(".g3_history");
            path
        });
    
    if let Some(ref history_path) = history_file {
        let _ = rl.load_history(history_path);
    }
    
    loop {
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
                
                // Execute task (code-first approach)
                match agent.execute_task_with_timing(input, None, false, show_prompt, show_code, true).await {
                    Ok(response) => println!("{}", response),
                    Err(e) => error!("Error: {}", e),
                }
            },
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                continue;
            },
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            },
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
