use std::{
    io,
    time::{Duration, Instant},
};

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, widgets::*};

use audio_agent::{
    audio::AudioChunk,
    decoder::{Segment, Task},
    model::WhichModel,
    stream::{Stream, StreamItem},
};

pub struct App {
    // stream: Stream,
    segments: Vec<Segment>,
}

impl App {
    pub fn new() -> Result<Self> {
        Ok(Self {
            // stream,
            segments: vec![],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        cpu: bool,
        model: WhichModel,
        model_id: Option<String>,
        revision: Option<String>,
        seed: u64,
        task: Option<Task>,
        timestamps: bool,
        language: Option<String>,
    ) -> Result<()> {
        // setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let (tx, rx): (Sender<StreamItem>, Receiver<StreamItem>) = unbounded();
        let stream = Stream::new(
            cpu, model, model_id, revision, seed, task, timestamps, language,
        )?;
        std::thread::spawn(move || {
            for msg in stream.text_stream().unwrap() {
                tx.send(msg).unwrap();
            }
        });
        let res = self.event_loop(&mut terminal, rx);

        // restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        res
    }

    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        receiver: crossbeam_channel::Receiver<StreamItem>,
    ) -> Result<(), anyhow::Error> {
        terminal.draw(|f| self.ui(f, AudioChunk::default(), vec![]))?;

        let tick_rate = Duration::from_millis(250);
        let mut last_tick = Instant::now();
        loop {
            if let Ok(StreamItem { wave, segments }) = receiver.recv_timeout(tick_rate) {
                terminal.draw(|f| self.ui(f, wave, segments))?;
            }

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if crossterm::event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') {
                        return Ok(());
                    }
                }
            }
            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }
        }
    }

    fn ui(&mut self, frame: &mut Frame, wave: AudioChunk, segments: Vec<Segment>) {
        let area = frame.size();

        let vertical = Layout::vertical([Constraint::Percentage(80), Constraint::Percentage(20)]);
        let [chart, bottom] = vertical.areas(area);

        self.render_wave(frame, chart, wave);
        self.render_text(frame, bottom, segments);
    }

    fn render_wave(&self, f: &mut Frame, area: Rect, wave: AudioChunk) {
        // let (start, end) = wave.time_window();
        let payload = wave.payload();
        let start = 0_f64;
        let end = payload.len() as f64;

        let data = payload
            .iter()
            .enumerate()
            .map(|(i, d)| (i as f64, *d as f64))
            .collect::<Vec<(f64, f64)>>();

        let x_labels = vec![
            Span::styled(
                format!("{}", start),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("{}", (start + end) / 2.0)),
            Span::styled(
                format!("{}", end),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ];
        let datasets = vec![Dataset::default()
            // .name("data2")
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Cyan))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .title("Audio".cyan().bold())
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    // .title("Sample")
                    .style(Style::default().fg(Color::Gray))
                    .labels(x_labels)
                    .bounds([start, end]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .labels(vec!["-0.25".bold(), "0".into(), "0.25".bold()])
                    .bounds([-0.25, 0.25]),
            );

        f.render_widget(chart, area);
    }

    fn render_text(&mut self, f: &mut Frame, area: Rect, segments: Vec<Segment>) {
        // max up to 5
        let len = self.segments.len() + segments.len();
        if len >= 5 {
            self.segments.drain(0..len - 5);
        }
        self.segments.extend(segments);

        let text: Vec<_> = self
            .segments
            .iter()
            .map(|seg| {
                text::Line::from(format!(
                    "{:.3}s -- {:.3}s: {}",
                    seg.start,
                    seg.start + seg.duration,
                    seg.dr.text
                ))
                // text::Line::from(seg.dr.text)
            })
            .collect();

        let block = Block::default().borders(Borders::ALL).title(Span::styled(
            "Transcript",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ));
        let paragraph = Paragraph::new(text).block(block).wrap(Wrap { trim: true });
        f.render_widget(paragraph, area);
    }
}
