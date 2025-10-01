use crossterm::style::Color;
use termimad::MadSkin;

/// Simple output handler with markdown support
pub struct SimpleOutput {
    mad_skin: MadSkin,
}

impl SimpleOutput {
    pub fn new() -> Self {
        let mut mad_skin = MadSkin::default();
        // Configure termimad skin for better markdown rendering
        mad_skin.set_headers_fg(Color::Cyan);
        mad_skin.bold.set_fg(Color::Yellow);
        mad_skin.italic.set_fg(Color::Magenta);
        mad_skin.code_block.set_bg(Color::Rgb { r: 40, g: 40, b: 40 });
        
        Self { mad_skin }
    }

    pub fn print(&self, text: &str) {
        println!("{}", text);
    }

    pub fn print_markdown(&self, markdown: &str) {
        self.mad_skin.print_text(markdown);
    }

    pub fn print_status(&self, status: &str) {
        println!("📊 {}", status);
    }

    pub fn print_context(&self, used: u32, total: u32, percentage: f32) {
        let bar_width = 10;
        let filled_width = ((percentage / 100.0) * bar_width as f32) as usize;
        let empty_width = bar_width - filled_width;

        let filled_chars = "●".repeat(filled_width);
        let empty_chars = "○".repeat(empty_width);
        let progress_bar = format!("{}{}", filled_chars, empty_chars);

        println!(
            "Context: {} {:.1}% | {}/{} tokens",
            progress_bar, percentage, used, total
        );
    }
}
