//! Shared rerun visualisation helpers for the ondrive examples.
//!
//! The library itself is viz-agnostic (rerun is a dev-dependency only).
//! Each example imports this module via `#[path = "common/viz.rs"] mod viz;`.
#![allow(dead_code)]

use ondrive::{ControllerStatus, Goal, Path as OPath, RobotState};
use rerun::{Color, RecordingStream};

pub fn red() -> Color {
    Color::from_rgb(255, 0, 0)
}
pub fn green() -> Color {
    Color::from_rgb(0, 255, 0)
}
pub fn blue() -> Color {
    Color::from_rgb(0, 120, 255)
}
pub fn yellow() -> Color {
    Color::from_rgb(255, 255, 0)
}
pub fn magenta() -> Color {
    Color::from_rgb(255, 0, 255)
}
pub fn cyan() -> Color {
    Color::from_rgb(0, 255, 255)
}
pub fn orange() -> Color {
    Color::from_rgb(255, 165, 0)
}
pub fn purple() -> Color {
    Color::from_rgb(128, 0, 128)
}

pub fn palette() -> [Color; 8] {
    [
        red(),
        green(),
        blue(),
        yellow(),
        magenta(),
        cyan(),
        orange(),
        purple(),
    ]
}

fn points_from_path(path: &OPath) -> Vec<[f32; 3]> {
    path.waypoints
        .iter()
        .map(|p| [p.point.x as f32, p.point.y as f32, 0.0])
        .collect()
}

pub fn show_path(rec: &RecordingStream, path: &OPath, entity: &str, color: Color) {
    if path.waypoints.is_empty() {
        return;
    }
    let points = points_from_path(path);
    let _ = rec.log_static(
        entity,
        &rerun::LineStrips3D::new([points]).with_colors([color]),
    );
}

pub fn show_paths(rec: &RecordingStream, paths: &[&OPath], entity_prefix: &str) {
    let pal = palette();
    for (i, p) in paths.iter().enumerate() {
        let color = pal[i % pal.len()];
        let entity = format!("{entity_prefix}/{i}");
        show_path(rec, p, &entity, color);
    }
}

pub fn show_predicted_trajectory(
    rec: &RecordingStream,
    points: &[datapod::Point],
    entity: &str,
    color: Color,
) {
    if points.is_empty() {
        return;
    }
    let pts: Vec<[f32; 3]> = points
        .iter()
        .map(|p| [p.x as f32, p.y as f32, 0.0])
        .collect();
    let _ = rec.log(
        entity,
        &rerun::LineStrips3D::new([pts]).with_colors([color]),
    );
}

pub fn show_robot_pose(
    rec: &RecordingStream,
    pose: &datapod::Pose,
    entity: &str,
    color: Color,
    scale: f32,
) {
    let yaw = pose.rotation.to_euler().yaw as f32;
    let cy = yaw.cos();
    let sy = yaw.sin();
    let width = 0.4 * scale;
    let length = 0.6 * scale;
    let cx = pose.point.x as f32;
    let cyp = pose.point.y as f32;

    let corners_local: [(f32, f32); 5] = [
        (length / 2.0, width / 2.0),
        (length / 2.0, -width / 2.0),
        (-length / 2.0, -width / 2.0),
        (-length / 2.0, width / 2.0),
        (length / 2.0, width / 2.0),
    ];
    let body: Vec<[f32; 3]> = corners_local
        .iter()
        .map(|(lx, ly)| [cx + cy * lx - sy * ly, cyp + sy * lx + cy * ly, 0.0])
        .collect();
    let _ = rec.log(
        format!("{entity}/body"),
        &rerun::LineStrips3D::new([body])
            .with_colors([color])
            .with_radii([0.025 * scale]),
    );

    let arrow_len = length * 0.4;
    let front_cx = cx + cy * length / 2.0;
    let front_cy = cyp + sy * length / 2.0;
    let end_cx = cx + cy * (length / 2.0 + arrow_len);
    let end_cy = cyp + sy * (length / 2.0 + arrow_len);
    let arrow: Vec<[f32; 3]> = vec![[front_cx, front_cy, 0.0], [end_cx, end_cy, 0.0]];
    let _ = rec.log(
        format!("{entity}/orientation"),
        &rerun::LineStrips3D::new([arrow])
            .with_colors([color])
            .with_radii([0.03 * scale]),
    );
}

pub fn show_robot_state(
    rec: &RecordingStream,
    state: &RobotState,
    entity: &str,
    color: Color,
    scale: f32,
) {
    show_robot_pose(rec, &state.pose, entity, color, scale);
}

pub fn show_goal(rec: &RecordingStream, goal: &Goal, entity: &str, color: Color) {
    let _ = rec.log_static(
        entity,
        &rerun::Points3D::new([[
            goal.target_pose.point.x as f32,
            goal.target_pose.point.y as f32,
            0.0,
        ]])
        .with_colors([color])
        .with_radii([0.15]),
    );
}

pub fn show_controller_status(rec: &RecordingStream, status: &ControllerStatus, entity: &str) {
    let text = format!(
        "Mode: {}\nDistance to goal: {:.3}\nHeading error: {:.3}\nCTE: {:.3}\nGoal reached: {}",
        status.mode,
        status.distance_to_goal,
        status.heading_error,
        status.cross_track_error,
        if status.goal_reached { "Yes" } else { "No" }
    );
    let _ = rec.log(entity, &rerun::TextLog::new(text));
}
