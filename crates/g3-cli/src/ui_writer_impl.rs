use g3_core::ui_writer::UiWriter;
use std::io::{self, Write};

/// Console implementation of UiWriter that prints to stdout
pub struct ConsoleUiWriter;

impl ConsoleUiWriter {
    pub fn new() -> Self {
        Self
    }
}

impl UiWriter for ConsoleUiWriter {
    fn print(&self, message: &str) {
        println!("{}", message);
    }
    
    fn println(&self, message: &str) {
        println!("{}", message);
    }
    
    fn print_inline(&self, message: &str) {
        print!("{}", message);
        let _ = io::stdout().flush();
    }
    
    fn print_system_prompt(&self, prompt: &str) {
        println!("🔍 System Prompt:");
        println!("================");
        println!("{}", prompt);
        println!("================");
        println!();
    }
    
    fn print_context_status(&self, message: &str) {
        println!("{}", message);
    }
    
    fn print_tool_header(&self, tool_name: &str) {
        println!("┌─ {}", tool_name);
    }
    
    fn print_tool_arg(&self, key: &str, value: &str) {
        println!("│ {}: {}", key, value);
    }
    
    fn print_tool_output_header(&self) {
        println!("├─ output:");
    }
    
    fn print_tool_output_line(&self, line: &str) {
        println!("│ {}", line);
    }
    
    fn print_tool_output_summary(&self, hidden_count: usize) {
        println!(
            "│ ... ({} more line{} hidden)",
            hidden_count,
            if hidden_count == 1 { "" } else { "s" }
        );
    }
    
    fn print_tool_timing(&self, duration_str: &str) {
        println!("└─ ⚡️ {}", duration_str);
        println!();
    }
    
    fn print_agent_prompt(&self) {
        print!("🤖 ");
        let _ = io::stdout().flush();
    }
    
    fn print_agent_response(&self, content: &str) {
        print!("{}", content);
        let _ = io::stdout().flush();
    }
    
    fn flush(&self) {
        let _ = io::stdout().flush();
    }
}