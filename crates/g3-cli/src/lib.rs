use clap::{Parser, Subcommand};
use g3_core::Agent;
use g3_config::Config;
use anyhow::Result;
use tracing::{info, error};

#[derive(Parser)]
#[command(name = "g3")]
#[command(about = "A modular, composable AI coding agent")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
    
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
}

#[derive(Subcommand)]
pub enum Commands {
    /// Solve any task by writing and executing code
    Task {
        /// Description of the task to accomplish
        description: String,
        /// Programming language to prefer (auto-detect if not specified)
        #[arg(short, long)]
        language: Option<String>,
        /// Execute the generated code automatically (default: ask for approval)
        #[arg(short, long)]
        execute: bool,
    },
    /// Create automation scripts for recurring tasks
    Automate {
        /// Description of the workflow to automate
        workflow: String,
        /// Output file for the automation script
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Process and analyze data with code
    Data {
        /// Description of the data processing task
        operation: String,
        /// Input file or data source
        #[arg(short, long)]
        input: Option<String>,
        /// Output format (json, csv, text)
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },
    /// Web-related tasks (scraping, APIs, downloads)
    Web {
        /// Description of the web task
        task: String,
        /// Target URL (if applicable)
        #[arg(short, long)]
        url: Option<String>,
    },
    /// File system operations and management
    File {
        /// Description of the file operation
        operation: String,
        /// Target path
        #[arg(short, long)]
        path: Option<String>,
    },
    /// Legacy: Analyze code and provide insights
    Analyze {
        /// Path to analyze
        path: String,
        /// Output format (json, text)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Legacy: Generate code based on description
    Generate {
        /// Description of what to generate
        description: String,
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Legacy: Review code and suggest improvements
    Review {
        /// Path to review
        path: String,
    },
    /// Interactive mode (default)
    Interactive,
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
    
    // Execute command - default to Interactive if no command provided
    match cli.command.unwrap_or(Commands::Interactive) {
        Commands::Task { description, language, execute } => {
            info!("Executing task: {}", description);
            let result = agent.execute_task(&description, language.as_deref(), execute).await?;
            println!("{}", result);
        }
        Commands::Automate { workflow, output } => {
            info!("Creating automation: {}", workflow);
            let result = agent.create_automation(&workflow).await?;
            
            if let Some(output_path) = output {
                std::fs::write(&output_path, &result)?;
                println!("Automation script written to: {}", output_path);
            } else {
                println!("{}", result);
            }
        }
        Commands::Data { operation, input, format } => {
            info!("Processing data: {}", operation);
            let result = agent.process_data(&operation, input.as_deref()).await?;
            
            match format.as_str() {
                "json" => println!("{}", serde_json::to_string_pretty(&result)?),
                _ => println!("{}", result),
            }
        }
        Commands::Web { task, url } => {
            info!("Web task: {}", task);
            let result = agent.execute_web_task(&task, url.as_deref()).await?;
            println!("{}", result);
        }
        Commands::File { operation, path } => {
            info!("File operation: {}", operation);
            let result = agent.execute_file_operation(&operation, path.as_deref()).await?;
            println!("{}", result);
        }
        Commands::Analyze { path, format } => {
            info!("Analyzing: {}", path);
            let result = agent.analyze(&path).await?;
            
            match format.as_str() {
                "json" => println!("{}", serde_json::to_string_pretty(&result)?),
                _ => println!("{}", result),
            }
        }
        Commands::Generate { description, output } => {
            info!("Generating code: {}", description);
            let result = agent.generate(&description).await?;
            
            if let Some(output_path) = output {
                std::fs::write(&output_path, &result)?;
                println!("Generated code written to: {}", output_path);
            } else {
                println!("{}", result);
            }
        }
        Commands::Review { path } => {
            info!("Reviewing: {}", path);
            let result = agent.review(&path).await?;
            println!("{}", result);
        }
        Commands::Interactive => {
            info!("Starting interactive mode");
            run_interactive(agent, cli.show_prompt, cli.show_code).await?;
        }
    }
    
    Ok(())
}

async fn run_interactive(agent: Agent, show_prompt: bool, show_code: bool) -> Result<()> {
    println!("🤖 G3 General Purpose AI Agent - Interactive Mode");
    println!("I solve problems by writing code. Tell me what you need to accomplish!");
    println!();
    println!("Type 'exit' or 'quit' to exit");
    println!();
    
    loop {
        print!("g3> ");
        use std::io::{self, Write};
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        let input = input.trim();
        if input == "exit" || input == "quit" {
            break;
        }
        
        if input.is_empty() {
            continue;
        }
        
        // Handle legacy commands
        if let Some(path) = input.strip_prefix("analyze ") {
            match agent.analyze(path).await {
                Ok(result) => println!("{}", result),
                Err(e) => error!("Error analyzing {}: {}", path, e),
            }
            continue;
        }
        
        if let Some(description) = input.strip_prefix("generate ") {
            match agent.generate(description).await {
                Ok(result) => println!("{}", result),
                Err(e) => error!("Error generating code: {}", e),
            }
            continue;
        }
        
        if let Some(path) = input.strip_prefix("review ") {
            match agent.review(path).await {
                Ok(result) => println!("{}", result),
                Err(e) => error!("Error reviewing {}: {}", path, e),
            }
            continue;
        }
        
        // Default to task execution (code-first approach)
        match agent.execute_task_with_timing(input, None, false, show_prompt, show_code, true).await {
            Ok(response) => println!("{}", response),
            Err(e) => error!("Error: {}", e),
        }
    }
    
    println!("👋 Goodbye!");
    Ok(())
}
