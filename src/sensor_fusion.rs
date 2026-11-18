use super::filter::{
    AccelerationObservation, AngularVelocityObservation, Ekf, LinearVelocityObservation,
    TwistObservation, UpdateOutcome, WorldLinearVelocityObservation,
};
use super::pose::{
    Cov6, Cov15, Nominal, State6, Transform3, Velocity6, cov3_in_6, pose_to_state, state7_to_pose,
    transform_to_velocity,
};
use super::traits::{EkfDataSource, Measurement, SourceRole};
use log::warn as rr_warn;
use nalgebra::{Translation3, UnitQuaternion};
use std::collections::{HashMap, VecDeque};

const HISTORY_WINDOW_S: f64 = 0.5;

const MIN_DELTA_DT_S: f64 = 1e-6;

#[derive(Clone, Copy)]
struct LastReading {
    pose: Transform3,
    t: f64,
    r: Cov6,
}

pub struct IntegrationResult {
    pub abs: State6,
    pub outcome: Option<UpdateOutcome>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
enum HistoryEvent {
    Reading {
        source_name: &'static str,
        role: SourceRole,
        measurement: Measurement,
    },
    Tick,
}

type LastRawDelta = Option<(&'static str, LastReading)>;

struct HistoryEntry {
    t: f64,
    event: HistoryEvent,
    nominal_after: Nominal,
    cov_after: Cov15,
    last_raw_delta: LastRawDelta,
}

struct Snapshot {
    t: f64,
    nominal: Nominal,
    cov: Cov15,
    last_raw: HashMap<&'static str, LastReading>,
}

pub struct SensorFusion {
    pub ekf: Ekf,
    last_raw: HashMap<&'static str, LastReading>,
    history: VecDeque<HistoryEntry>,
    base: Snapshot,
    latest_t: f64,
    pending_delta: LastRawDelta,
}

impl SensorFusion {
    pub fn new(ekf: Ekf) -> Self {
        let base = Snapshot {
            t: f64::NEG_INFINITY,
            nominal: ekf.nominal,
            cov: ekf.covariance(),
            last_raw: HashMap::new(),
        };
        Self {
            ekf,
            last_raw: HashMap::new(),
            history: VecDeque::new(),
            base,
            latest_t: f64::NEG_INFINITY,
            pending_delta: None,
        }
    }

    pub fn tick(&mut self, t: f64) {
        self.insert_event(t, HistoryEvent::Tick);
    }

    pub fn integrate(&mut self, source: &mut dyn EkfDataSource) -> Vec<IntegrationResult> {
        let readings = source.sample();
        if readings.is_empty() {
            return Vec::new();
        }
        let role = source.role();
        let name = source.name();
        readings
            .into_iter()
            .filter_map(|reading| {
                self.insert_event(
                    reading.t,
                    HistoryEvent::Reading {
                        source_name: name,
                        role,
                        measurement: reading.measurement,
                    },
                )
            })
            .collect()
    }

    fn insert_event(&mut self, t: f64, event: HistoryEvent) -> Option<IntegrationResult> {
        self.latest_t = self.latest_t.max(t);
        let idx = self.history.iter().take_while(|e| e.t <= t).count();

        if idx == 0 && t < self.base.t {
            match &event {
                HistoryEvent::Reading { source_name, .. } => rr_warn!(
                    "SensorFusion: dropping reading from '{source_name}' at t={t:.3}s — older \
                     than the {HISTORY_WINDOW_S}s replay window (earliest replayable t={:.3}s)",
                    self.base.t
                ),
                HistoryEvent::Tick => rr_warn!(
                    "SensorFusion: dropping tick at t={t:.3}s — older than the replay window \
                     (earliest replayable t={:.3}s)",
                    self.base.t
                ),
            }
            return None;
        }

        if idx < self.history.len() {
            if idx == 0 {
                self.ekf.restore(self.base.nominal, self.base.cov);
            } else {
                let prev = &self.history[idx - 1];
                self.ekf.restore(prev.nominal_after, prev.cov_after);
            }
            self.last_raw.clear();
            self.last_raw.extend(self.base.last_raw.iter());
            for entry in self.history.iter().take(idx) {
                if let Some((name, reading)) = entry.last_raw_delta {
                    self.last_raw.insert(name, reading);
                }
            }
        }

        let mut prev_t = if idx == 0 {
            self.base.t
        } else {
            self.history[idx - 1].t
        };

        let tail = self.history.split_off(idx);
        let mut result = None;
        for (i, (ev_t, ev)) in std::iter::once((t, event))
            .chain(tail.into_iter().map(|e| (e.t, e.event)))
            .enumerate()
        {
            let dt = if prev_t.is_finite() {
                (ev_t - prev_t).max(0.0)
            } else {
                0.0 // nothing ever came before this — no elapsed time to coast over
            };
            self.ekf.predict(dt);
            self.pending_delta = None;
            let r = match &ev {
                HistoryEvent::Tick => None,
                HistoryEvent::Reading {
                    source_name,
                    role,
                    measurement,
                } => Some(self.integrate_one(source_name, ev_t, *role, *measurement)),
            };
            if i == 0 {
                result = r;
            }
            self.history.push_back(HistoryEntry {
                t: ev_t,
                event: ev,
                nominal_after: self.ekf.nominal,
                cov_after: self.ekf.covariance(),
                last_raw_delta: self.pending_delta.take(),
            });
            prev_t = ev_t;
        }

        self.prune();
        result
    }

    fn prune(&mut self) {
        while let Some(front) = self.history.front() {
            if front.t >= self.latest_t - HISTORY_WINDOW_S {
                break;
            }
            let front = self.history.pop_front().unwrap();
            self.base.t = front.t;
            self.base.nominal = front.nominal_after;
            self.base.cov = front.cov_after;
            if let Some((name, reading)) = front.last_raw_delta {
                self.base.last_raw.insert(name, reading);
            }
        }
    }

    fn integrate_one(
        &mut self,
        source_name: &'static str,
        t: f64,
        role: SourceRole,
        reading: Measurement,
    ) -> IntegrationResult {
        match reading {
            Measurement::Twist { v, r } => {
                let outcome = self.ekf.update(&v, &r, &TwistObservation);
                self.rate_update(outcome)
            }
            Measurement::LinearVelocity { v, r } => {
                let outcome = self.ekf.update(&v, &r, &LinearVelocityObservation);
                self.rate_update(outcome)
            }
            Measurement::AngularVelocity { w, r } => {
                let outcome = self.ekf.update(&w, &r, &AngularVelocityObservation);
                self.rate_update(outcome)
            }
            Measurement::Acceleration { a, r } => {
                let outcome = self.ekf.update(&a, &r, &AccelerationObservation);
                self.rate_update(outcome)
            }
            Measurement::Pose { z, r, r_scatter } => {
                let abs_pose = state7_to_pose(&z);
                let mut outcome = None;
                if role.corrects_pose() {
                    outcome = Some(self.ekf.update_pose_gated(&z, &r));
                }
                if role.derives_rate() {
                    let rate =
                        self.differential_delta(source_name, t, abs_pose, r_scatter, role, true);
                    if let Some((twist, r_twist)) = rate {
                        let rate_outcome = self.ekf.update(&twist, &r_twist, &TwistObservation);
                        outcome = outcome.or(Some(rate_outcome));
                    }
                }
                IntegrationResult {
                    abs: pose_to_state(&abs_pose),
                    outcome,
                }
            }
            Measurement::Position { z, r } => {
                let mut outcome = None;
                if role.corrects_pose() {
                    outcome = Some(self.ekf.update_gps_gated(&z, &r));
                }
                if role.derives_rate() {
                    let abs_pose = Transform3::from_parts(
                        Translation3::new(z[0], z[1], z[2]),
                        UnitQuaternion::identity(),
                    );
                    let rate = self.differential_delta(
                        source_name,
                        t,
                        abs_pose,
                        cov3_in_6(&r),
                        role,
                        false, // identity rotation is a device, not a measurement
                    );
                    if let Some((twist, r_twist)) = rate {
                        let v_world = twist.fixed_rows::<3>(0).into_owned();
                        let r3 = r_twist.fixed_view::<3, 3>(0, 0).into_owned();
                        let rate_outcome =
                            self.ekf
                                .update(&v_world, &r3, &WorldLinearVelocityObservation);
                        outcome = outcome.or(Some(rate_outcome));
                    }
                }
                IntegrationResult {
                    abs: State6::new(z[0], z[1], z[2], 0.0, 0.0, 0.0),
                    outcome,
                }
            }
        }
    }

    fn rate_update(&self, outcome: UpdateOutcome) -> IntegrationResult {
        IntegrationResult {
            abs: pose_to_state(&self.ekf.nominal.pose),
            outcome: Some(outcome),
        }
    }

    fn differential_delta(
        &mut self,
        source_name: &'static str,
        t: f64,
        abs_pose: Transform3,
        r: Cov6,
        role: SourceRole,
        has_orientation: bool,
    ) -> Option<(Velocity6, Cov6)> {
        let current = LastReading {
            pose: abs_pose,
            t,
            r,
        };
        self.pending_delta = Some((source_name, current));
        let Some(prev) = self.last_raw.insert(source_name, current) else {
            if !role.corrects_pose() {
                self.ekf.nominal.pose.translation = abs_pose.translation;
                if has_orientation {
                    self.ekf.nominal.pose.rotation = abs_pose.rotation;
                }
            }
            return None;
        };
        let dt = t - prev.t;
        if dt <= MIN_DELTA_DT_S {
            rr_warn!(
                "SensorFusion: '{source_name}' reported twice within {MIN_DELTA_DT_S}s \
                 (dt={dt:.6}s) — cannot derive a rate, skipping"
            );
            return None;
        }
        let delta = prev.pose.inverse() * abs_pose;
        Some((transform_to_velocity(&delta) / dt, (prev.r + r) / (dt * dt)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::pose::{
        Cov3, Cov6, Cov15, Pos3, Vec3, Velocity6, ate_m, pose_to_state7, state_to_pose,
    };
    use super::super::traits::Reading;
    use super::*;
    use approx::assert_abs_diff_eq;

    struct FixedPoseSource {
        name: &'static str,
        role: SourceRole,
        next: Vec<Reading>,
    }

    impl EkfDataSource for FixedPoseSource {
        fn name(&self) -> &'static str {
            self.name
        }
        fn sample(&mut self) -> Vec<Reading> {
            std::mem::take(&mut self.next)
        }
        fn role(&self) -> SourceRole {
            self.role
        }
    }

    fn pose_reading(t: f64, euler: State6) -> Reading {
        Reading {
            t,
            measurement: Measurement::Pose {
                z: pose_to_state7(&state_to_pose(&euler)),
                r: Cov6::identity() * 0.01,
                r_scatter: Cov6::identity() * 0.01,
            },
        }
    }

    fn position_reading(t: f64, x: f64, y: f64, z: f64) -> Reading {
        Reading {
            t,
            measurement: Measurement::Position {
                z: Pos3::new(x, y, z),
                r: Cov3::identity() * 0.01,
            },
        }
    }

    #[test]
    fn differential_first_tick_falls_through_to_absolute() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "predict",
            role: SourceRole::Differential,
            next: vec![pose_reading(1.0, State6::new(2.0, 1.0, 0.0, 0.0, 0.0, 0.0))],
        };

        fusion.integrate(&mut source);

        assert_abs_diff_eq!(fusion.ekf.state()[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fusion.ekf.state()[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn a_both_source_does_not_snap_away_its_own_gated_pose_update() {
        const Z_X: f64 = 0.2;
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity() * 0.01,
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "cam",
            role: SourceRole::Both,
            next: vec![pose_reading(1.0, State6::new(Z_X, 0.0, 0.0, 0.0, 0.0, 0.0))],
        };

        let results = fusion.integrate(&mut source);
        assert_eq!(
            results[0].outcome,
            Some(UpdateOutcome::Accepted),
            "the pose update must be admitted, or this test says nothing"
        );

        let x = fusion.ekf.state()[0];
        assert!(
            (0.4 * Z_X..0.6 * Z_X).contains(&x),
            "expected a Kalman-weighted ~{} from a P=R update, got {x} — {Z_X} means the \
             bootstrap snapped over the gated update",
            0.5 * Z_X
        );
    }

    #[test]
    fn a_differential_source_still_bootstraps_its_pose() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity() * 0.01,
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "vio",
            role: SourceRole::Differential,
            next: vec![pose_reading(1.0, State6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))],
        };

        fusion.integrate(&mut source);

        assert_abs_diff_eq!(fusion.ekf.state()[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn differential_delta_becomes_a_gated_velocity_not_a_pose_composition() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "vio",
            role: SourceRole::Differential,
            next: vec![pose_reading(1.0, State6::zeros())],
        };
        fusion.integrate(&mut source); // bootstrap: snaps pose, no velocity yet
        assert_abs_diff_eq!(fusion.ekf.nominal.v.norm(), 0.0, epsilon = 1e-12);

        source.next = vec![pose_reading(1.5, State6::new(0.5, 0.0, 0.0, 0.0, 0.0, 0.0))];
        let results = fusion.integrate(&mut source);

        assert_eq!(
            results[0].outcome,
            Some(UpdateOutcome::Accepted),
            "a differential delta is now a gated measurement, not ungated motion"
        );
        assert!(
            (0.9..1.0).contains(&fusion.ekf.nominal.v.x),
            "expected v to move most of the way to 1 m/s, got {}",
            fusion.ekf.nominal.v.x
        );
    }

    #[test]
    fn both_role_corrects_the_pose_and_derives_a_rate_from_one_reading() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "odom",
            role: SourceRole::Both,
            next: vec![pose_reading(1.0, State6::zeros())],
        };
        fusion.integrate(&mut source); // bootstrap

        source.next = vec![pose_reading(1.5, State6::new(0.5, 0.0, 0.0, 0.0, 0.0, 0.0))];
        let results = fusion.integrate(&mut source);

        assert_eq!(results[0].outcome, Some(UpdateOutcome::Accepted));
        assert!(
            fusion.ekf.nominal.v.x > 0.5,
            "the rate half of `Both` must still fire, got v={}",
            fusion.ekf.nominal.v.x
        );
        assert!(
            fusion.ekf.state()[0] > 0.1,
            "the pose half of `Both` must still fire — `Differential` alone \
             leaves the pose at 0 here, got x={}",
            fusion.ekf.state()[0]
        );
    }

    #[test]
    fn differential_moves_the_pose_only_indirectly_via_cross_covariance() {
        let run = |role: SourceRole| {
            let mut fusion = SensorFusion::new(Ekf::new(
                Transform3::identity(),
                Cov15::identity(),
                Cov15::zeros(),
            ));
            let mut source = FixedPoseSource {
                name: "vio",
                role,
                next: vec![pose_reading(1.0, State6::zeros())],
            };
            fusion.integrate(&mut source);
            source.next = vec![pose_reading(1.5, State6::new(0.5, 0.0, 0.0, 0.0, 0.0, 0.0))];
            fusion.integrate(&mut source);
            (fusion.ekf.state()[0], fusion.ekf.nominal.v.x)
        };

        let (x_diff, v_diff) = run(SourceRole::Differential);
        let (x_both, v_both) = run(SourceRole::Both);

        assert!(
            x_diff > 0.0,
            "cross-covariance should pull the pose: {x_diff}"
        );
        assert!(
            x_both > x_diff,
            "the direct pose correction must move the pose further than the \
             cross-covariance path alone: {x_diff} then {x_both}"
        );
        assert!(v_diff > 0.5 && v_both > 0.5, "both roles correct v");
    }

    #[test]
    fn differential_rate_uses_the_sources_own_elapsed_time() {
        let velocity_after = |gap: f64| {
            let mut fusion = SensorFusion::new(Ekf::new(
                Transform3::identity(),
                Cov15::identity(),
                Cov15::zeros(),
            ));
            let mut source = FixedPoseSource {
                name: "vio",
                role: SourceRole::Differential,
                next: vec![pose_reading(1.0, State6::zeros())],
            };
            fusion.integrate(&mut source);
            source.next = vec![pose_reading(
                1.0 + gap,
                State6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )];
            fusion.integrate(&mut source);
            fusion.ekf.nominal.v.x
        };
        let (fast, slow) = (velocity_after(1.0), velocity_after(2.0));
        assert!(
            slow < fast * 0.6,
            "halving the rate must halve the derived velocity: {fast} then {slow}"
        );
    }

    #[test]
    fn differential_rotation_delta_uses_quaternion_composition_not_euler_subtraction() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "vio",
            role: SourceRole::Differential,
            next: vec![pose_reading(1.0, State6::new(0.0, 0.0, 0.0, 0.0, 0.0, 3.0))],
        };
        fusion.integrate(&mut source);

        source.next = vec![pose_reading(
            2.0,
            State6::new(0.0, 0.0, 0.0, 0.0, 0.0, -3.13),
        )];
        fusion.integrate(&mut source);

        let expected = 2.0 * std::f64::consts::PI - 6.13;
        assert!(
            (fusion.ekf.nominal.w.z - expected).abs() < 0.05,
            "expected ~{expected:.3} rad/s across the seam, got {}",
            fusion.ekf.nominal.w.z
        );
    }

    fn vio_fusion() -> (SensorFusion, FixedPoseSource) {
        (
            SensorFusion::new(Ekf::new(
                Transform3::identity(),
                Cov15::identity(),
                Cov15::zeros(),
            )),
            FixedPoseSource {
                name: "vio",
                role: SourceRole::Differential,
                next: Vec::new(),
            },
        )
    }

    fn x_reading(t: f64, x: f64) -> Reading {
        pose_reading(t, State6::new(x, 0.0, 0.0, 0.0, 0.0, 0.0))
    }

    #[test]
    fn rewind_rebuilds_last_raw_from_the_entries_before_the_insertion_point() {
        let (mut fusion, mut source) = vio_fusion();

        source.next = vec![x_reading(0.0, 0.0)];
        fusion.integrate(&mut source); // bootstrap
        source.next = vec![x_reading(0.2, 0.02)];
        fusion.integrate(&mut source);

        source.next = vec![x_reading(0.1, 0.01)];
        let results = fusion.integrate(&mut source);

        assert_eq!(
            results[0].outcome,
            Some(UpdateOutcome::Accepted),
            "the late reading derived no rate, so the rewind lost `last_raw`"
        );
    }

    #[test]
    fn pruned_entries_advance_the_base_last_raw_they_age_out_of() {
        let (mut fusion, mut source) = vio_fusion();

        for (i, t) in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].iter().enumerate() {
            source.next = vec![x_reading(*t, 0.02 * i as f64)];
            fusion.integrate(&mut source);
        }

        source.next = vec![x_reading(0.5, 0.05)];
        let results = fusion.integrate(&mut source);

        assert_eq!(
            results[0].outcome,
            Some(UpdateOutcome::Accepted),
            "replaying against `base` derived no rate, so `base.last_raw` was \
             not advanced as entries aged out"
        );
    }

    #[test]
    fn differential_position_source_derives_a_world_frame_velocity() {
        let mut fusion = SensorFusion::new(Ekf::new(
            state_to_pose(&State6::new(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                std::f64::consts::FRAC_PI_2,
            )),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "gps_diff",
            role: SourceRole::Differential,
            next: vec![position_reading(1.0, 0.0, 0.0, 0.0)],
        };
        fusion.integrate(&mut source); // bootstrap, nothing to diff yet
        assert_abs_diff_eq!(fusion.ekf.nominal.v.norm(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            fusion.ekf.state()[5],
            std::f64::consts::FRAC_PI_2,
            epsilon = 1e-12
        );

        source.next = vec![position_reading(1.5, 0.5, 0.0, 0.0)];
        let results = fusion.integrate(&mut source);
        assert_eq!(results[0].outcome, Some(UpdateOutcome::Accepted));

        let v = fusion.ekf.nominal.v;
        assert!(
            v.y < -0.3,
            "world +x at a 90° yaw must appear on body -y, got {v:?} — the \
             velocity is being treated as body-frame"
        );
        assert!(
            v.x.abs() < v.y.abs() * 0.5,
            "body x should carry little of a world-x velocity at 90° yaw, got {v:?}"
        );
    }

    #[test]
    fn absolute_position_source_updates_ekf_directly() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity() * 100.0,
            Cov15::identity() * 1e-3,
        ));
        let mut source = FixedPoseSource {
            name: "gps",
            role: SourceRole::Absolute,
            next: vec![position_reading(1.0, 4.0, -2.0, 0.0)],
        };

        let results = fusion.integrate(&mut source);
        assert_eq!(results.len(), 1, "source had exactly one queued reading");
        assert_eq!(results[0].outcome, Some(UpdateOutcome::Accepted));
        assert_abs_diff_eq!(
            ate_m(
                &fusion.ekf.state(),
                &State6::new(4.0, -2.0, 0.0, 0.0, 0.0, 0.0)
            ),
            0.0,
            epsilon = 1.0
        );
    }

    #[test]
    fn nothing_new_this_tick_returns_empty() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "gps",
            role: SourceRole::Absolute,
            next: Vec::new(),
        };
        assert!(fusion.integrate(&mut source).is_empty());
    }

    #[test]
    fn multiple_queued_readings_are_each_gated_individually() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity() * 100.0,
            Cov15::identity() * 1e-3,
        ));
        let mut source = FixedPoseSource {
            name: "gps",
            role: SourceRole::Absolute,
            next: vec![
                position_reading(1.0, 1.0, 0.0, 0.0),
                position_reading(1.01, 1.0, 0.0, 0.0),
            ],
        };

        let results = fusion.integrate(&mut source);
        assert_eq!(
            results.len(),
            2,
            "both queued readings should be integrated"
        );
        assert_eq!(results[0].outcome, Some(UpdateOutcome::Accepted));
        assert_eq!(results[1].outcome, Some(UpdateOutcome::Accepted));
        assert_abs_diff_eq!(fusion.ekf.state()[0], 1.0, epsilon = 0.05);
    }

    #[test]
    fn twist_reading_is_a_gated_correction_not_dead_reckoning() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "wheel_odom",
            role: SourceRole::Absolute,
            next: vec![Reading {
                t: 1.0,
                measurement: Measurement::Twist {
                    v: Velocity6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                    r: Cov6::identity() * 0.01,
                },
            }],
        };

        let results = fusion.integrate(&mut source);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, Some(UpdateOutcome::Accepted));
        assert_abs_diff_eq!(fusion.ekf.nominal.v.x, 1.0, epsilon = 0.05);
        assert_abs_diff_eq!(fusion.ekf.nominal.w.z, 0.5, epsilon = 0.05);
        assert_abs_diff_eq!(
            fusion.ekf.state()[0],
            0.0,
            epsilon = 1e-12 // no motion model yet, so a rate cannot move the pose
        );
    }

    #[test]
    fn every_rate_variant_dispatches_to_its_own_state_block() {
        let measurements = [
            Measurement::LinearVelocity {
                v: Vec3::new(0.4, 0.0, 0.0),
                r: Cov3::identity() * 0.01,
            },
            Measurement::AngularVelocity {
                w: Vec3::new(0.0, 0.0, 0.4),
                r: Cov3::identity() * 0.01,
            },
            Measurement::Acceleration {
                a: Vec3::new(0.0, 0.4, 0.0),
                r: Cov3::identity() * 0.01,
            },
        ];
        let expected = [(0.4, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, 0.4)];

        for (measurement, (want_v, want_w, want_a)) in measurements.into_iter().zip(expected) {
            let mut fusion = SensorFusion::new(Ekf::new(
                Transform3::identity(),
                Cov15::identity(),
                Cov15::zeros(),
            ));
            let mut source = FixedPoseSource {
                name: "imu",
                role: SourceRole::Absolute,
                next: vec![Reading {
                    t: 1.0,
                    measurement,
                }],
            };

            let results = fusion.integrate(&mut source);
            assert_eq!(results[0].outcome, Some(UpdateOutcome::Accepted));
            assert_abs_diff_eq!(fusion.ekf.nominal.v.norm(), want_v, epsilon = 0.05);
            assert_abs_diff_eq!(fusion.ekf.nominal.w.norm(), want_w, epsilon = 0.05);
            assert_abs_diff_eq!(fusion.ekf.nominal.a.norm(), want_a, epsilon = 0.05);
        }
    }

    #[test]
    fn rate_readings_apply_regardless_of_source_role() {
        for differential in [false, true] {
            let mut fusion = SensorFusion::new(Ekf::new(
                Transform3::identity(),
                Cov15::identity(),
                Cov15::zeros(),
            ));
            let mut source = FixedPoseSource {
                name: "gyro",
                role: if differential {
                    SourceRole::Differential
                } else {
                    SourceRole::Absolute
                },
                next: vec![Reading {
                    t: 1.0,
                    measurement: Measurement::AngularVelocity {
                        w: Vec3::new(0.0, 0.0, 0.4),
                        r: Cov3::identity() * 0.01,
                    },
                }],
            };

            let results = fusion.integrate(&mut source);
            assert_eq!(results[0].outcome, Some(UpdateOutcome::Accepted));
            assert_abs_diff_eq!(fusion.ekf.nominal.w.z, 0.4, epsilon = 0.05);
        }
    }

    #[test]
    fn a_late_rate_reading_replays_correctly() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "odom",
            role: SourceRole::Absolute,
            next: vec![Reading {
                t: 0.2,
                measurement: Measurement::LinearVelocity {
                    v: Vec3::new(1.0, 0.0, 0.0),
                    r: Cov3::identity() * 0.01,
                },
            }],
        };
        fusion.integrate(&mut source);
        let after_first = fusion.ekf.nominal.v.x;

        source.next = vec![Reading {
            t: 0.1,
            measurement: Measurement::LinearVelocity {
                v: Vec3::new(1.0, 0.0, 0.0),
                r: Cov3::identity() * 0.01,
            },
        }];
        let results = fusion.integrate(&mut source);

        assert_eq!(results.len(), 1, "the late rate reading was applied");
        assert!(
            fusion.ekf.nominal.v.x > after_first,
            "replaying both updates should pull v further toward the measurement"
        );
    }

    #[test]
    fn tick_grows_uncertainty_scaled_by_elapsed_time() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::zeros(),
            Cov15::identity() * 1e-3,
        ));
        fusion.tick(0.0); // first-ever event — no prior reference, zero growth
        let trace_before = fusion.ekf.covariance().trace();
        fusion.tick(2.0); // dt = 2.0 since the previous tick
        let grown = fusion.ekf.covariance().trace() - trace_before;
        assert_abs_diff_eq!(grown, 1e-3 * 15.0 * 2.0, epsilon = 1e-9);
    }

    #[test]
    fn tick_grows_uncertainty_even_when_no_source_reports() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::zeros(),
            Cov15::identity() * 1e-3,
        ));
        fusion.tick(0.0);
        let trace_before = fusion.ekf.covariance().trace();
        fusion.tick(1.0);
        assert!(fusion.ekf.covariance().trace() > trace_before);
    }

    #[test]
    fn late_reading_is_rewound_and_replayed_not_just_appended() {
        let run = |late: Option<f64>| {
            let mut fusion = SensorFusion::new(Ekf::new(
                Transform3::identity(),
                Cov15::identity(),
                Cov15::zeros(),
            ));
            let mut s1 = FixedPoseSource {
                name: "s1",
                role: SourceRole::Differential,
                next: vec![pose_reading(0.1, State6::zeros())],
            };
            fusion.integrate(&mut s1); // bootstrap
            s1.next = vec![pose_reading(0.3, State6::new(0.2, 0.0, 0.0, 0.0, 0.0, 0.0))];
            fusion.integrate(&mut s1);
            if let Some(x) = late {
                s1.next = vec![pose_reading(0.2, State6::new(x, 0.0, 0.0, 0.0, 0.0, 0.0))];
                let results = fusion.integrate(&mut s1);
                assert_eq!(results.len(), 1, "the late reading itself was accepted");
            }
            fusion.ekf.nominal.v.x
        };

        let appended_only = run(None);
        assert!(
            appended_only > 0.5,
            "sanity: the 1 m/s measurement must not be gated, got {appended_only}"
        );

        let split_evenly = run(Some(0.1));
        assert!(
            split_evenly < appended_only - 0.05,
            "subdividing the same motion should lose information, not gain it: \
             {appended_only} then {split_evenly}"
        );
    }

    #[test]
    fn late_reading_changes_the_derived_rate_of_the_reading_after_it() {
        let run = |late: Option<f64>| {
            let mut fusion = SensorFusion::new(Ekf::new(
                Transform3::identity(),
                Cov15::identity(),
                Cov15::zeros(),
            ));
            let mut s1 = FixedPoseSource {
                name: "s1",
                role: SourceRole::Differential,
                next: vec![pose_reading(0.1, State6::zeros())],
            };
            fusion.integrate(&mut s1);
            s1.next = vec![pose_reading(0.3, State6::new(0.2, 0.0, 0.0, 0.0, 0.0, 0.0))];
            fusion.integrate(&mut s1);
            if let Some(x) = late {
                s1.next = vec![pose_reading(0.2, State6::new(x, 0.0, 0.0, 0.0, 0.0, 0.0))];
                fusion.integrate(&mut s1);
            }
            fusion.ekf.nominal.v.x
        };

        let baseline = run(None);
        let with_stalled_tail = run(Some(0.2));
        assert!(
            with_stalled_tail < baseline,
            "the replayed reading must re-derive its velocity against the newly \
             inserted one: {baseline} then {with_stalled_tail}"
        );
    }

    #[test]
    fn rewind_restores_the_rate_states_not_just_the_pose() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "gps",
            role: SourceRole::Absolute,
            next: vec![position_reading(0.1, 1.0, 0.0, 0.0)],
        };
        fusion.integrate(&mut source); // snapshot at t=0.1 has v = 0

        fusion.ekf.nominal.v = Vec3::new(7.0, 0.0, 0.0);

        source.next = vec![position_reading(0.2, 2.0, 0.0, 0.0)];
        fusion.integrate(&mut source); // snapshot at t=0.2 captures v = 7

        source.next = vec![position_reading(0.15, 1.5, 0.0, 0.0)];
        fusion.integrate(&mut source);

        assert_abs_diff_eq!(fusion.ekf.nominal.v.x, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn reading_older_than_history_window_is_dropped() {
        let mut fusion = SensorFusion::new(Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::zeros(),
        ));
        let mut source = FixedPoseSource {
            name: "gps",
            role: SourceRole::Absolute,
            next: vec![position_reading(0.0, 1.0, 0.0, 0.0)],
        };
        fusion.integrate(&mut source);

        source.next = vec![position_reading(HISTORY_WINDOW_S + 1.0, 2.0, 0.0, 0.0)];
        fusion.integrate(&mut source);

        source.next = vec![position_reading(-1.0, 99.0, 0.0, 0.0)];
        let results = fusion.integrate(&mut source);
        assert!(
            results.is_empty(),
            "reading older than the replay window should be dropped, not applied"
        );
    }
}
