#![allow(clippy::approx_constant)]

use std::f64::consts::PI;

use ondrive::turners::{
    Dubins, DubinsSegmentType, Pose2D, ReedsShepp, ReedsSheppSegmentType, Sharper, point_distance,
};
use ondrive::turners::{Point2Ext, point_xy};

fn assert_pose_close(actual: Pose2D, expected: Pose2D, tol: f64) {
    assert!(point_distance(actual.point, expected.point) <= tol);
    assert!((actual.yaw - expected.yaw).abs() <= tol);
}

fn seeded_random_pose_pair() -> (Pose2D, Pose2D) {
    let mut state: u64 = 1337;
    let mut next_f64 = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (state >> 11) as f64;
        bits / ((1u64 << 53) as f64)
    };
    let start = Pose2D::new(
        1.0 + next_f64() * 2.0,
        1.0 + next_f64() * 2.0,
        next_f64() * 2.0 * PI - PI,
    );
    let end = Pose2D::new(
        start.point.x() + next_f64() * 6.0 - 3.0,
        start.point.y() + next_f64() * 6.0 - 3.0,
        next_f64() * 2.0 * PI - PI,
    );
    (start, end)
}

#[test]
fn dubins_matches_reference_case_1() {
    let start = Pose2D::new(0.0, 0.0, 0.0);
    let goal = Pose2D::new(5.0, 2.0, 1.0);

    let dubins = Dubins::new(1.0);
    let path = dubins.plan_path(start, goal, 0.2);

    assert_eq!(path.name, "LSL");
    assert_eq!(path.segments.len(), 3);
    assert_eq!(path.waypoints.len(), 29);
    assert!((path.total_length - 5.434_621_474_236_828).abs() < 1e-5);
    assert_eq!(path.segments[0].r#type, DubinsSegmentType::Left);
    assert_eq!(path.segments[1].r#type, DubinsSegmentType::Straight);
    assert_eq!(path.segments[2].r#type, DubinsSegmentType::Left);
    assert!((path.segments[0].length - 0.354_727_642_930_729_2).abs() < 1e-5);
    assert!((path.segments[1].length - 4.434_621_474_236_828).abs() < 1e-5);
    assert!((path.segments[2].length - 0.645_272_357_069_270_8).abs() < 1e-5);
    assert_pose_close(*path.waypoints.first().unwrap(), start, 1e-12);
    assert_pose_close(*path.waypoints.last().unwrap(), goal, 1e-12);
}

#[test]
fn dubins_matches_reference_case_2() {
    let start = Pose2D::new(0.0, 0.0, 0.5);
    let goal = Pose2D::new(4.0, -1.0, -0.75);

    let dubins = Dubins::new(1.0);
    let path = dubins.plan_path(start, goal, 0.2);

    assert_eq!(path.name, "RSR");
    assert_eq!(path.segments.len(), 3);
    assert_eq!(path.waypoints.len(), 23);
    assert!((path.total_length - 4.214_629_251_687_159).abs() < 1e-5);
    assert_eq!(path.segments[0].r#type, DubinsSegmentType::Right);
    assert_eq!(path.segments[1].r#type, DubinsSegmentType::Straight);
    assert_eq!(path.segments[2].r#type, DubinsSegmentType::Right);
    assert!((path.segments[0].length - 0.792_240_314_465_343_2).abs() < 1e-5);
    assert!((path.segments[1].length - 2.964_629_251_687_158_6).abs() < 1e-5);
    assert!((path.segments[2].length - 0.457_759_685_534_656_83).abs() < 1e-5);
    assert_pose_close(*path.waypoints.first().unwrap(), start, 1e-12);
    assert_pose_close(*path.waypoints.last().unwrap(), goal, 1e-12);
}

#[test]
fn dubins_get_all_paths_contains_shortest_family() {
    let start = Pose2D::new(0.0, 0.0, 0.0);
    let goal = Pose2D::new(5.0, 2.0, 1.0);

    let dubins = Dubins::new(1.0);
    let all = dubins.get_all_paths(start, goal, 0.2);
    let best = dubins.plan_path(start, goal, 0.2);

    assert!(!all.is_empty());
    assert!(all.iter().any(|path| path.name == best.name));
    assert!(all.iter().all(|path| !path.waypoints.is_empty()));
    assert!(all.iter().all(|path| path.total_length > 0.0));
}

#[test]
fn dubins_heads_example_path_families_are_stable() {
    let start = Pose2D::new(0.0, 0.0, 0.0);
    let goal = Pose2D::new(0.0, 1.0, PI);

    let dubins = Dubins::new(0.2);
    let names = dubins
        .get_all_paths(start, goal, 0.05)
        .into_iter()
        .map(|path| path.name)
        .collect::<Vec<_>>();

    assert_eq!(names, vec!["LSL", "RSR", "RSL", "LSR", "LRL"]);
}

#[test]
fn dubins_random_heads_seed_is_stable() {
    let (start, goal) = seeded_random_pose_pair();

    let dubins = Dubins::new(0.2);
    let names = dubins
        .get_all_paths(start, goal, 0.05)
        .into_iter()
        .map(|path| path.name)
        .collect::<Vec<_>>();

    assert_eq!(names, vec!["LSL", "RSR", "RSL", "LSR"]);
}

#[test]
fn reeds_shepp_matches_reference_case_1() {
    let start = Pose2D::new(0.0, 0.0, 0.0);
    let goal = Pose2D::new(3.0, -2.0, 3.14);

    let rs = ReedsShepp::new(1.0);
    let path = rs.plan_path(start, goal, 0.2);

    assert_eq!(path.name, "Lrsr");
    assert_eq!(path.segments.len(), 4);
    assert_eq!(path.waypoints.len(), 25);
    assert!((path.total_length - 4.746_880_816_460_614).abs() < 1e-5);
    assert_eq!(path.segments[0].r#type, ReedsSheppSegmentType::LeftForward);
    assert_eq!(
        path.segments[1].r#type,
        ReedsSheppSegmentType::RightBackward
    );
    assert_eq!(
        path.segments[2].r#type,
        ReedsSheppSegmentType::StraightBackward
    );
    assert_eq!(
        path.segments[3].r#type,
        ReedsSheppSegmentType::RightBackward
    );
    assert!(path.segments[0].forward);
    assert!(!path.segments[1].forward);
    assert!(!path.segments[2].forward);
    assert!(!path.segments[3].forward);
    assert_pose_close(*path.waypoints.first().unwrap(), start, 1e-12);
    assert_pose_close(*path.waypoints.last().unwrap(), goal, 1e-12);
}

#[test]
fn reeds_shepp_get_all_paths_contains_shortest_family() {
    let start = Pose2D::new(0.0, 0.0, 0.0);
    let goal = Pose2D::new(3.0, -2.0, 3.14);

    let rs = ReedsShepp::new(1.0);
    let all = rs.get_all_paths(start, goal, 0.2);
    let best = rs.plan_path(start, goal, 0.2);

    assert!(!all.is_empty());
    assert!(all.iter().any(|path| path.name == best.name));
    assert!(all.iter().all(|path| !path.waypoints.is_empty()));
    assert!(all.iter().all(|path| path.total_length > 0.0));
}

#[test]
fn reeds_shepp_heads_example_path_families_are_stable() {
    let start = Pose2D::new(0.0, 0.0, 0.0);
    let goal = Pose2D::new(0.0, 1.0, PI);

    let rs = ReedsShepp::new(0.2);
    let names = rs
        .get_all_paths(start, goal, 0.05)
        .into_iter()
        .map(|path| path.name)
        .collect::<Vec<_>>();

    assert_eq!(names, vec!["LSL", "lsl", "rlsl", "rLSL", "lslR", "LSLR"]);
}

#[test]
fn reeds_shepp_random_heads_seed_is_stable() {
    let (start, goal) = seeded_random_pose_pair();

    let rs = ReedsShepp::new(0.2);
    let names = rs
        .get_all_paths(start, goal, 0.05)
        .into_iter()
        .map(|path| path.name)
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec!["RSL", "rsl", "Lrsl", "lRSL", "Rlsl", "RSLr", "RSRl"]
    );
}

#[test]
fn reeds_shepp_matches_reference_case_2() {
    let start = Pose2D::new(0.0, 0.0, 0.3);
    let goal = Pose2D::new(-1.0, 0.5, -2.2);

    let rs = ReedsShepp::new(1.0);
    let path = rs.plan_path(start, goal, 0.2);

    assert_eq!(path.name, "lRl");
    assert_eq!(path.segments.len(), 3);
    assert_eq!(path.waypoints.len(), 14);
    assert!((path.total_length - 2.500_000_254_363_117).abs() < 1e-5);
    assert_eq!(path.segments[0].r#type, ReedsSheppSegmentType::LeftBackward);
    assert_eq!(path.segments[1].r#type, ReedsSheppSegmentType::RightForward);
    assert_eq!(path.segments[2].r#type, ReedsSheppSegmentType::LeftBackward);
    assert!(!path.segments[0].forward);
    assert!(path.segments[1].forward);
    assert!(!path.segments[2].forward);
    assert_pose_close(*path.waypoints.first().unwrap(), start, 1e-12);
    assert_pose_close(*path.waypoints.last().unwrap(), goal, 1e-12);
}

#[test]
fn sharper_auto_same_point_matches_reference_pattern() {
    let start = Pose2D::from_point(point_xy(0.0, 0.0), 0.0);
    let goal = Pose2D::from_point(point_xy(0.0, 0.0), PI);

    let sharper = Sharper::new(1.0, 4.0, 2.0);
    let path = sharper.plan_sharp_turn(start, goal, "auto");

    assert_eq!(path.pattern_name, "three_point");
    assert_eq!(
        path.segment_types,
        vec![
            "start",
            "forward_along_old_heading",
            "reverse_with_turn",
            "forward_to_turning_point",
        ]
    );
    assert_eq!(path.waypoints.len(), 4);
    assert!((path.total_length - 8.0).abs() < 0.01);
    assert_pose_close(*path.waypoints.first().unwrap(), start, 1e-12);
    assert_pose_close(*path.waypoints.last().unwrap(), goal, 1e-12);
}

#[test]
fn sharper_auto_far_matches_reference_pattern() {
    let start = Pose2D::from_point(point_xy(0.0, 0.0), 0.0);
    let goal = Pose2D::from_point(point_xy(6.0, 2.0), 1.0);

    let sharper = Sharper::new(1.0, 4.0, 2.0);
    let path = sharper.plan_sharp_turn(start, goal, "auto");

    assert_eq!(path.pattern_name, "fishtail");
    assert_eq!(
        path.segment_types,
        vec![
            "start",
            "approach",
            "tail_out",
            "transition",
            "final_approach",
            "end",
        ]
    );
    assert_eq!(path.waypoints.len(), 6);
    assert!((path.total_length - 17.3778).abs() < 1e-3);
    assert_pose_close(*path.waypoints.first().unwrap(), start, 1e-12);
    assert_pose_close(*path.waypoints.last().unwrap(), goal, 1e-12);
}

#[test]
fn sharper_explicit_bulb_matches_reference_pattern() {
    let start = Pose2D::from_point(point_xy(0.0, 0.0), 0.0);
    let goal = Pose2D::from_point(point_xy(6.0, 2.0), 1.0);

    let sharper = Sharper::new(1.0, 4.0, 2.0);
    let path = sharper.plan_sharp_turn(start, goal, "bulb");

    assert_eq!(path.pattern_name, "bulb");
    assert_eq!(path.segment_types.len(), 7);
    assert_eq!(path.waypoints.len(), 7);
    assert_eq!(path.segment_types.first().unwrap(), "start");
    assert_eq!(path.segment_types.last().unwrap(), "end");
    assert!(
        path.segment_types[1..6]
            .iter()
            .all(|segment| segment == "bulb_arc")
    );
    assert!((path.total_length - 11.5898).abs() < 1e-3);
    assert_pose_close(*path.waypoints.first().unwrap(), start, 1e-12);
    assert_pose_close(*path.waypoints.last().unwrap(), goal, 1e-12);
}
