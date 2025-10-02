use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
    Frame, Terminal,
};
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

// Retro sci-fi color scheme inspired by Alien terminals
const TERMINAL_GREEN: Color = Color::Rgb(0, 255, 65); // Bright phosphor green
const TERMINAL_AMBER: Color = Color::Rgb(255, 176, 0); // Amber for warnings
const TERMINAL_DIM_GREEN: Color = Color::Rgb(0, 128, 32); // Dimmed green for borders
const TERMINAL_BG: Color = Color::Rgb(0, 10, 0); // Very dark green background
const TERMINAL_CYAN: Color = Color::Rgb(0, 255, 255); // Cyan for highlights
const TERMINAL_RED: Color = Color::Rgb(255, 0, 0); // Red for errors

/// Message types for communication between threads
#[derive(Debug, Clone)]
pub enum TuiMessage {
    AgentOutput(String),
    SystemStatus(String),
    ContextUpdate {
        used: u32,
        total: u32,
        percentage: f32,
    },
    Error(String),
    Exit,
}

/// Shared state for the retro terminal
struct TerminalState {
    /// Current input buffer
    input_buffer: String,
    /// Output history
    output_history: Vec<String>,
    /// Scroll position in output
    scroll_offset: usize,
    /// Cursor blink state
    cursor_blink: bool,
    /// Last cursor blink time
    last_blink: Instant,
    /// System status line
    status_line: String,
    /// Context window info
    context_info: (u32, u32, f32),
    /// Should exit
    should_exit: bool,
}

impl TerminalState {
    fn new() -> Self {
        Self {
            input_buffer: String::new(),
            output_history: vec![
                "WEYLAND-YUTANI SYSTEMS".to_string(),
                "MU/TH/UR 6000 - INTERFACE 2.4.1".to_string(),
                "".to_string(),
                "SYSTEM INITIALIZED".to_string(),
                "AWAITING COMMAND...".to_string(),
                "".to_string(),
            ],
            scroll_offset: 0,
            cursor_blink: true,
            last_blink: Instant::now(),
            status_line: "READY".to_string(),
            context_info: (0, 0, 0.0),
            should_exit: false,
        }
    }

    /// Add text to output history
    fn add_output(&mut self, text: &str) {
        // Split text by newlines and add each line
        for line in text.lines() {
            self.output_history.push(line.to_string());
        }
        // Auto-scroll to bottom
        self.scroll_offset = self.output_history.len().saturating_sub(1);
    }
}

/// Public interface for the retro terminal
#[derive(Clone)]
pub struct RetroTui {
    tx: mpsc::UnboundedSender<TuiMessage>,
    state: Arc<Mutex<TerminalState>>,
    terminal: Arc<Mutex<Terminal<CrosstermBackend<io::Stdout>>>>,
}

impl RetroTui {
    /// Create and start the retro terminal UI
    pub async fn start() -> Result<Self> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        // Create message channel
        let (tx, mut rx) = mpsc::unbounded_channel::<TuiMessage>();

        let state = Arc::new(Mutex::new(TerminalState::new()));
        let terminal = Arc::new(Mutex::new(terminal));

        // Clone for the background task
        let state_clone = state.clone();
        let terminal_clone = terminal.clone();

        // Spawn background task to handle messages and redraw
        tokio::spawn(async move {
            let mut last_draw = Instant::now();

            loop {
                // Check for messages
                while let Ok(msg) = rx.try_recv() {
                    let mut state = state_clone.lock().unwrap();
                    match msg {
                        TuiMessage::AgentOutput(text) => {
                            state.add_output(&text);
                        }
                        TuiMessage::SystemStatus(status) => {
                            state.status_line = status;
                        }
                        TuiMessage::ContextUpdate {
                            used,
                            total,
                            percentage,
                        } => {
                            state.context_info = (used, total, percentage);
                        }
                        TuiMessage::Error(err) => {
                            state.add_output(&format!("ERROR: {}", err));
                        }
                        TuiMessage::Exit => {
                            state.should_exit = true;
                            break;
                        }
                    }
                }

                // Check if we should exit
                if state_clone.lock().unwrap().should_exit {
                    break;
                }

                // Update cursor blink
                {
                    let mut state = state_clone.lock().unwrap();
                    if state.last_blink.elapsed() > Duration::from_millis(500) {
                        state.cursor_blink = !state.cursor_blink;
                        state.last_blink = Instant::now();
                    }
                }

                // Redraw at ~60fps
                if last_draw.elapsed() > Duration::from_millis(16) {
                    let state = state_clone.lock().unwrap();
                    let mut term = terminal_clone.lock().unwrap();
                    let _ = Self::draw(&mut term, &state);
                    last_draw = Instant::now();
                }

                // Small sleep to prevent busy waiting
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });

        // Initial draw
        {
            let state = state.lock().unwrap();
            let mut term = terminal.lock().unwrap();
            Self::draw(&mut term, &state)?;
        }

        Ok(Self {
            tx,
            state,
            terminal,
        })
    }

    /// Draw the terminal UI
    fn draw(
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        state: &TerminalState,
    ) -> Result<()> {
        terminal.draw(|f| {
            let size = f.area();

            // Create main layout - header, input, output
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(5), // Header/input area
                    Constraint::Min(10),   // Main output area
                    Constraint::Length(1), // Status bar
                ])
                .split(size);

            // Draw header/input area
            Self::draw_input_area(f, chunks[0], &state.input_buffer, state.cursor_blink);

            // Draw main output area
            Self::draw_output_area(f, chunks[1], &state.output_history, state.scroll_offset);

            // Draw status bar
            Self::draw_status_bar(f, chunks[2], &state.status_line, state.context_info);
        })?;

        Ok(())
    }

    /// Draw the input area with prompt
    fn draw_input_area(f: &mut Frame, area: Rect, input_buffer: &str, cursor_blink: bool) {
        // Show the actual input buffer content with prompt
        let input_text = if cursor_blink {
            format!("g3> {}█", input_buffer)
        } else {
            format!("g3> {} ", input_buffer)
        };

        let input = Paragraph::new(input_text)
            .style(Style::default().fg(TERMINAL_GREEN))
            .block(
                Block::default()
                    .title(" COMMAND INPUT ")
                    .title_alignment(Alignment::Center)
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(TERMINAL_DIM_GREEN))
                    .style(Style::default().bg(TERMINAL_BG)),
            );

        f.render_widget(input, area);
    }

    /// Draw the main output area
    fn draw_output_area(
        f: &mut Frame,
        area: Rect,
        output_history: &[String],
        scroll_offset: usize,
    ) {
        // Calculate visible lines
        let visible_height = area.height.saturating_sub(2) as usize; // Account for borders
        let total_lines = output_history.len();

        // Adjust scroll offset to ensure it's valid
        let max_scroll = total_lines.saturating_sub(visible_height);
        let scroll = scroll_offset.min(max_scroll);

        // Get visible lines
        let visible_lines: Vec<Line> = output_history
            .iter()
            .skip(scroll)
            .take(visible_height)
            .map(|line| {
                // Apply different colors based on content
                let style = if line.starts_with("ERROR:") {
                    Style::default()
                        .fg(TERMINAL_RED)
                        .add_modifier(Modifier::BOLD)
                } else if line.starts_with('>') {
                    Style::default().fg(TERMINAL_CYAN)
                } else if line.starts_with("SYSTEM:")
                    || line.starts_with("WEYLAND")
                    || line.starts_with("MU/TH/UR")
                {
                    Style::default()
                        .fg(TERMINAL_AMBER)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(TERMINAL_GREEN)
                };

                Line::from(Span::styled(line.clone(), style))
            })
            .collect();

        let output = Paragraph::new(visible_lines)
            .block(
                Block::default()
                    .title(" SYSTEM OUTPUT ")
                    .title_alignment(Alignment::Center)
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(TERMINAL_DIM_GREEN))
                    .style(Style::default().bg(TERMINAL_BG)),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(output, area);

        // Draw scrollbar if needed
        if total_lines > visible_height {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("▲"))
                .end_symbol(Some("▼"))
                .track_symbol(Some("│"))
                .thumb_symbol("█")
                .style(Style::default().fg(TERMINAL_DIM_GREEN));

            let mut scrollbar_state = ScrollbarState::new(total_lines)
                .position(scroll)
                .viewport_content_length(visible_height);

            f.render_stateful_widget(
                scrollbar,
                area.inner(ratatui::layout::Margin {
                    vertical: 1,
                    horizontal: 0,
                }),
                &mut scrollbar_state,
            );
        }
    }

    /// Draw the status bar
    fn draw_status_bar(
        f: &mut Frame,
        area: Rect,
        status_line: &str,
        context_info: (u32, u32, f32),
    ) {
        let (used, total, percentage) = context_info;

        // Create context meter
        let bar_width = 10;
        let filled = ((percentage / 100.0) * bar_width as f32) as usize;
        let meter = format!("[{}{}]", "█".repeat(filled), "░".repeat(bar_width - filled));

        let status_text = format!(
            " STATUS: {} | CONTEXT: {} {:.1}% ({}/{} tokens) | ↑↓ SCROLL | CTRL-C EXIT ",
            status_line, meter, percentage, used, total
        );

        let status = Paragraph::new(status_text)
            .style(
                Style::default()
                    .fg(TERMINAL_AMBER)
                    .bg(TERMINAL_BG)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(Alignment::Left);

        f.render_widget(status, area);
    }

    /// Send output to the terminal
    pub fn output(&self, text: &str) {
        let _ = self.tx.send(TuiMessage::AgentOutput(text.to_string()));
    }

    /// Update system status
    pub fn status(&self, status: &str) {
        let _ = self.tx.send(TuiMessage::SystemStatus(status.to_string()));
    }

    /// Update context window information
    pub fn update_context(&self, used: u32, total: u32, percentage: f32) {
        let _ = self.tx.send(TuiMessage::ContextUpdate {
            used,
            total,
            percentage,
        });
    }

    /// Send error message
    pub fn error(&self, error: &str) {
        let _ = self.tx.send(TuiMessage::Error(error.to_string()));
    }

    /// Signal exit
    pub fn exit(&self) {
        let _ = self.tx.send(TuiMessage::Exit);
    }

    /// Update input buffer (for display)
    pub fn update_input(&self, input: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.input_buffer = input.to_string();
        }
    }

    /// Handle scrolling
    pub fn scroll_up(&self) {
        if let Ok(mut state) = self.state.lock() {
            if state.scroll_offset > 0 {
                state.scroll_offset -= 1;
            }
        }
    }

    pub fn scroll_down(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.scroll_offset += 1;
        }
    }

    pub fn scroll_page_up(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.scroll_offset = state.scroll_offset.saturating_sub(10);
        }
    }

    pub fn scroll_page_down(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.scroll_offset += 10;
        }
    }

    pub fn scroll_home(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.scroll_offset = 0;
        }
    }

    pub fn scroll_end(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.scroll_offset = state.output_history.len().saturating_sub(1);
        }
    }
}

impl Drop for RetroTui {
    fn drop(&mut self) {
        // Restore terminal
        let _ = disable_raw_mode();
        if let Ok(mut term) = self.terminal.lock() {
            let _ = execute!(
                term.backend_mut(),
                LeaveAlternateScreen,
                DisableMouseCapture
            );
        }
    }
}
