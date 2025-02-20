use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use std::rc::Rc;

use super::ScorerClassic;

pub fn plot_ratio(buf: &mut String, scorer: &ScorerClassic) -> anyhow::Result<()> {
    // Get all unique x-axis labels (workload descriptions)
    let mut x_labels: Vec<Rc<str>> = scorer
        .tools
        .values()
        .flat_map(|map| map.keys().cloned())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    x_labels.sort();

    // Calculate the y-axis range (ratios of derivative/primal)
    let mut min_ratio = f64::INFINITY;
    let mut max_ratio = f64::NEG_INFINITY;

    for tool_data in scorer.tools.values() {
        for pair in tool_data.values() {
            let ratio = pair.derivative.as_secs_f64() / pair.primal.as_secs_f64();
            if ratio > 0.0 {
                // Only consider positive values for log scale
                min_ratio = min_ratio.min(ratio);
                max_ratio = max_ratio.max(ratio);
            }
        }
    }

    // Add some padding to the y-axis range for log scale
    let log_min = min_ratio.ln();
    let log_max = max_ratio.ln();
    let log_padding = (log_max - log_min) * 0.1;
    let y_min = (log_min - log_padding).exp();
    let y_max = (log_max + log_padding).exp();

    // Create an SVG backend
    let backend = SVGBackend::with_string(buf, (800, 600));
    let root = backend.into_drawing_area();
    root.fill(&WHITE)?;

    // Create the chart with logarithmic y-axis
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Derivative/Primal Time Ratio by Workload",
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(120) // Increased to accommodate vertical labels
        .y_label_area_size(60)
        .build_cartesian_2d(0..(x_labels.len() - 1), (y_min..y_max).log_scale())?;

    // Configure the chart
    chart
        .configure_mesh()
        .x_labels(x_labels.len())
        .x_label_formatter(&|x| {
            x_labels
                .get(*x)
                .map(|s| s.as_ref())
                .unwrap_or("")
                .to_string()
        })
        .x_label_style(
            TextStyle::from(("sans-serif", 12))
                .transform(FontTransform::Rotate90)
                .pos(Pos::new(HPos::Center, VPos::Bottom)),
        )
        .y_desc("Derivative/Primal Ratio (log scale)")
        .draw()?;

    // Create a color series for the tools
    let colors = [
        &RGBColor(31, 119, 180),  // Blue
        &RGBColor(255, 127, 14),  // Orange
        &RGBColor(44, 160, 44),   // Green
        &RGBColor(214, 39, 40),   // Red
        &RGBColor(148, 103, 189), // Purple
        &RGBColor(140, 86, 75),   // Brown
        &RGBColor(227, 119, 194), // Pink
        &RGBColor(127, 127, 127), // Gray
    ];

    // Plot each tool's data as a series
    for (tool_idx, (tool_name, tool_data)) in scorer.tools.iter().enumerate() {
        let color = colors[tool_idx % colors.len()];

        let series_data: Vec<(usize, f64)> = x_labels
            .iter()
            .enumerate()
            .filter_map(|(i, label)| {
                tool_data.get(label).and_then(|pair| {
                    let ratio = pair.derivative.as_secs_f64() / pair.primal.as_secs_f64();
                    if ratio > 0.0 {
                        Some((i, ratio))
                    } else {
                        None
                    }
                })
            })
            .collect();

        chart
            .draw_series(LineSeries::new(
                series_data.iter().map(|(x, y)| (*x, *y)),
                color.stroke_width(2),
            ))?
            .label(tool_name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });
    }

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
