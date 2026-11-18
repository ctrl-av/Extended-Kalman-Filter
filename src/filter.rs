use super::pose::{
    Cov3, Cov6, Cov15, Nominal, Pos3, State6, State7, Tangent15, Transform3, Vec3, Velocity6,
    pose_to_state, state7_to_pose,
};
use log::warn as rr_warn;
use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateOutcome {
    Accepted,
    Gated,
    Singular,
    NotFinite,
}

impl UpdateOutcome {
    pub fn accepted(self) -> bool {
        matches!(self, Self::Accepted)
    }
}

const CHI2_6DOF: f64 = 12.592; // df=6, full 6-DOF pose update
const CHI2_3DOF: f64 = 7.815; // df=3, position-only (GPS) update

const REJECTION_INFLATION: f64 = 1.1;

pub const IP: usize = 0; // position         (world)
pub const ITH: usize = 3; // orientation error (tangent)
pub const IV: usize = 6; // linear velocity   (body)
pub const IW: usize = 9; // angular velocity  (body)
pub const IA: usize = 12; // acceleration      (body)

fn selector<const M: usize>(offset: usize) -> SMatrix<f64, M, 15> {
    SMatrix::<f64, M, 15>::from_fn(|row, col| if col == offset + row { 1.0 } else { 0.0 })
}

fn block3(v: &Tangent15, offset: usize) -> Vec3 {
    Vec3::new(v[offset], v[offset + 1], v[offset + 2])
}

pub trait ObservationModel<const M: usize> {
    type Measurement;

    fn h(&self, x: &Nominal) -> SMatrix<f64, M, 15>;

    fn chi2_gate(&self) -> f64;

    fn innovation(&self, z: &Self::Measurement, x: &Nominal) -> SVector<f64, M>;
}

pub struct PoseObservation;

impl ObservationModel<6> for PoseObservation {
    type Measurement = State7;

    fn h(&self, _x: &Nominal) -> SMatrix<f64, 6, 15> {
        selector::<6>(IP) // p and θ are contiguous, so one selector covers both
    }

    fn chi2_gate(&self) -> f64 {
        CHI2_6DOF
    }

    fn innovation(&self, z: &State7, x: &Nominal) -> State6 {
        let z_pose = state7_to_pose(z);
        let dp = z_pose.translation.vector - x.pose.translation.vector;
        let dtheta = (x.pose.rotation.inverse() * z_pose.rotation).scaled_axis();
        State6::new(dp[0], dp[1], dp[2], dtheta[0], dtheta[1], dtheta[2])
    }
}

pub struct TwistObservation;

impl ObservationModel<6> for TwistObservation {
    type Measurement = Velocity6;

    fn h(&self, _x: &Nominal) -> SMatrix<f64, 6, 15> {
        selector::<6>(IV) // v and w are contiguous
    }

    fn chi2_gate(&self) -> f64 {
        CHI2_6DOF
    }

    fn innovation(&self, z: &Velocity6, x: &Nominal) -> Velocity6 {
        Velocity6::new(
            z[0] - x.v.x,
            z[1] - x.v.y,
            z[2] - x.v.z,
            z[3] - x.w.x,
            z[4] - x.w.y,
            z[5] - x.w.z,
        )
    }
}

pub struct LinearVelocityObservation;

impl ObservationModel<3> for LinearVelocityObservation {
    type Measurement = Vec3;

    fn h(&self, _x: &Nominal) -> SMatrix<f64, 3, 15> {
        selector::<3>(IV)
    }

    fn chi2_gate(&self) -> f64 {
        CHI2_3DOF
    }

    fn innovation(&self, z: &Vec3, x: &Nominal) -> Vec3 {
        z - x.v
    }
}

pub struct AngularVelocityObservation;

impl ObservationModel<3> for AngularVelocityObservation {
    type Measurement = Vec3;

    fn h(&self, _x: &Nominal) -> SMatrix<f64, 3, 15> {
        selector::<3>(IW)
    }

    fn chi2_gate(&self) -> f64 {
        CHI2_3DOF
    }

    fn innovation(&self, z: &Vec3, x: &Nominal) -> Vec3 {
        z - x.w
    }
}

pub struct AccelerationObservation;

impl ObservationModel<3> for AccelerationObservation {
    type Measurement = Vec3;

    fn h(&self, _x: &Nominal) -> SMatrix<f64, 3, 15> {
        selector::<3>(IA)
    }

    fn chi2_gate(&self) -> f64 {
        CHI2_3DOF
    }

    fn innovation(&self, z: &Vec3, x: &Nominal) -> Vec3 {
        z - x.a
    }
}

pub struct WorldLinearVelocityObservation;

impl ObservationModel<3> for WorldLinearVelocityObservation {
    type Measurement = Vec3;

    fn h(&self, x: &Nominal) -> SMatrix<f64, 3, 15> {
        let r = x.pose.rotation.to_rotation_matrix().into_inner();
        let mut h = SMatrix::<f64, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, ITH)
            .copy_from(&(-r * x.v.cross_matrix()));
        h.fixed_view_mut::<3, 3>(0, IV).copy_from(&r);
        h
    }

    fn chi2_gate(&self) -> f64 {
        CHI2_3DOF
    }

    fn innovation(&self, z: &Vec3, x: &Nominal) -> Vec3 {
        z - x.pose.rotation * x.v
    }
}

pub struct PositionObservation;

impl ObservationModel<3> for PositionObservation {
    type Measurement = Pos3;

    fn h(&self, _x: &Nominal) -> SMatrix<f64, 3, 15> {
        selector::<3>(IP)
    }

    fn chi2_gate(&self) -> f64 {
        CHI2_3DOF
    }

    fn innovation(&self, z: &Pos3, x: &Nominal) -> Pos3 {
        z - x.pose.translation.vector
    }
}

pub fn motion_process_noise() -> Cov15 {
    motion_process_noise_with(SIGMA_V, SIGMA_W, SIGMA_A)
}

pub const SIGMA_V: f64 = 1.0; // (m/s)/√s
pub const SIGMA_W: f64 = 1.5; // (rad/s)/√s
pub const SIGMA_A: f64 = 0.5; // (m/s²)/√s

pub fn motion_process_noise_with(sigma_v: f64, sigma_w: f64, sigma_a: f64) -> Cov15 {
    const FLOOR_P: f64 = 1e-3; // m/√s
    const FLOOR_TH: f64 = 1e-3; // rad/√s
    let sq = |x: f64| x * x;
    Cov15::from_diagonal(&Tangent15::from_fn(|i, _| match i {
        0..=2 => sq(FLOOR_P),
        3..=5 => sq(FLOOR_TH),
        6..=8 => sq(sigma_v),
        9..=11 => sq(sigma_w),
        _ => sq(sigma_a),
    }))
}

fn right_jacobian_so3(phi: &Vec3) -> Matrix3<f64> {
    const SMALL_ANGLE: f64 = 1e-4;
    let theta = phi.norm();
    let px = phi.cross_matrix();
    if theta < SMALL_ANGLE {
        return Matrix3::identity() - px * 0.5 + px * px * (1.0 / 6.0);
    }
    let t2 = theta * theta;
    Matrix3::identity() - px * ((1.0 - theta.cos()) / t2)
        + px * px * ((theta - theta.sin()) / (t2 * theta))
}

pub struct Ekf {
    pub nominal: Nominal,
    cov: Cov15,
    q: Cov15,
}

impl Ekf {
    pub fn new(init_pose: Transform3, init_cov: Cov15, process_noise: Cov15) -> Self {
        Self {
            nominal: Nominal::at_rest(init_pose),
            cov: init_cov,
            q: process_noise,
        }
    }

    pub fn state(&self) -> State6 {
        pose_to_state(&self.nominal.pose)
    }

    pub fn predict(&mut self, dt: f64) {
        let f = self.jacobian(dt);
        self.nominal = self.nominal.propagated(dt);
        let p = f * self.cov * f.transpose() + self.q * dt;
        self.cov = (p + p.transpose()) * 0.5;
    }

    pub fn jacobian(&self, dt: f64) -> Cov15 {
        let x = &self.nominal;
        let r = x.pose.rotation.to_rotation_matrix().into_inner();
        let i3 = Matrix3::identity();
        let mut f = Cov15::identity();
        let half_dt2 = 0.5 * dt * dt;
        f.fixed_view_mut::<3, 3>(IP, ITH)
            .copy_from(&(-r * x.position_step(dt).cross_matrix()));
        f.fixed_view_mut::<3, 3>(IP, IV)
            .copy_from(&(r * (i3 * dt + x.w.cross_matrix() * half_dt2)));
        f.fixed_view_mut::<3, 3>(IP, IW)
            .copy_from(&(-r * x.v.cross_matrix() * half_dt2));
        f.fixed_view_mut::<3, 3>(IP, IA).copy_from(&(r * half_dt2));
        let phi = x.w * dt;
        f.fixed_view_mut::<3, 3>(ITH, ITH).copy_from(
            &UnitQuaternion::from_scaled_axis(-phi)
                .to_rotation_matrix()
                .into_inner(),
        );
        f.fixed_view_mut::<3, 3>(ITH, IW)
            .copy_from(&(right_jacobian_so3(&phi) * dt));
        f.fixed_view_mut::<3, 3>(IV, IA).copy_from(&(i3 * dt));
        f
    }

    pub fn covariance(&self) -> Cov15 {
        self.cov
    }

    pub fn restore(&mut self, nominal: Nominal, cov: Cov15) {
        self.nominal = nominal;
        self.cov = cov;
    }

    pub fn update<const M: usize, Obs: ObservationModel<M>>(
        &mut self,
        z: &Obs::Measurement,
        r: &SMatrix<f64, M, M>,
        obs: &Obs,
    ) -> UpdateOutcome {
        let h = obs.h(&self.nominal);
        let innovation = obs.innovation(z, &self.nominal);
        if !innovation.iter().all(|v| v.is_finite()) {
            rr_warn!("Ekf: dropping measurement — innovation is not finite");
            return UpdateOutcome::NotFinite;
        }
        let s = h * self.cov * h.transpose() + r; // S = HPH^T + R
        let Some(s_inv) = s.try_inverse() else {
            rr_warn!("Ekf: dropping measurement — innovation covariance S is singular");
            return UpdateOutcome::Singular;
        };
        let nis = (innovation.transpose() * s_inv * innovation)[(0, 0)];
        if !nis.is_finite() {
            rr_warn!("Ekf: dropping measurement — NIS is not finite");
            return UpdateOutcome::NotFinite;
        }
        if nis > obs.chi2_gate() {
            self.cov *= REJECTION_INFLATION;
            return UpdateOutcome::Gated; // outlier — skip update
        }
        let k = self.cov * h.transpose() * s_inv;
        let correction: Tangent15 = k * innovation;
        self.nominal.pose.translation.vector += block3(&correction, IP);
        let composed =
            self.nominal.pose.rotation * UnitQuaternion::from_scaled_axis(block3(&correction, ITH));
        self.nominal.pose.rotation = UnitQuaternion::new_normalize(composed.into_inner());
        self.nominal.v += block3(&correction, IV);
        self.nominal.w += block3(&correction, IW);
        self.nominal.a += block3(&correction, IA);
        let p = (Cov15::identity() - k * h) * self.cov;
        self.cov = (p + p.transpose()) * 0.5;
        UpdateOutcome::Accepted
    }

    pub fn update_pose_gated(&mut self, z: &State7, r: &Cov6) -> UpdateOutcome {
        self.update(z, r, &PoseObservation)
    }

    pub fn update_gps_gated(&mut self, z: &Pos3, r: &Cov3) -> UpdateOutcome {
        self.update(z, r, &PositionObservation)
    }
}

#[cfg(test)]
mod tests {
    use super::super::pose::{ate_m, pose_to_state7, state_to_pose};
    use super::*;
    use approx::assert_abs_diff_eq;

    fn euler7(x: f64, y: f64, z: f64, roll: f64, pitch: f64, yaw: f64) -> State7 {
        pose_to_state7(&state_to_pose(&State6::new(x, y, z, roll, pitch, yaw)))
    }

    #[test]
    fn rate_process_noise_can_admit_a_real_manoeuvre_within_a_few_ticks() {
        const DT_S: f64 = 0.05;
        let q = motion_process_noise();

        let admissible_v = (q[(6, 6)] * DT_S).sqrt() * 3.0;
        assert!(
            admissible_v >= 0.5,
            "Q_v admits only a {admissible_v:.3} m/s step; a robot stopping \
             from 0.5 m/s would be gated out and the filter would coast on a \
             stale velocity"
        );

        let admissible_w = (q[(9, 9)] * DT_S).sqrt() * 3.0;
        assert!(
            admissible_w >= 1.0,
            "Q_w admits only a {admissible_w:.3} rad/s step; starting a 1 rad/s \
             turn would be gated out"
        );
    }

    #[test]
    fn motion_process_noise_puts_the_uncertainty_on_the_rate_states() {
        let q = motion_process_noise();
        let (p, th) = (q[(0, 0)], q[(3, 3)]);
        let (v, w, a) = (q[(6, 6)], q[(9, 9)], q[(12, 12)]);
        assert!(p < v && p < w && p < a, "position Q must be a floor: {p}");
        assert!(
            th < v && th < w && th < a,
            "orientation Q must be a floor: {th}"
        );
        let _ = a;
        for i in 0..15 {
            assert!(q[(i, i)] > 0.0, "Q diagonal {i} must be positive");
            for j in 0..15 {
                if i != j {
                    assert_abs_diff_eq!(q[(i, j)], 0.0, epsilon = 1e-15);
                }
            }
        }
    }

    #[test]
    fn predict_at_rest_does_not_move_but_still_grows_covariance() {
        let q = Cov15::identity() * 1e-3;
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity() * 0.1, q);
        let trace_before = ekf.cov.trace();

        ekf.predict(2.0);

        assert_abs_diff_eq!(ekf.state()[0], 0.0, epsilon = 1e-12);
        assert!(
            ekf.cov.trace() > trace_before,
            "covariance must grow over elapsed time even at rest"
        );
    }

    #[test]
    fn process_noise_contribution_scales_linearly_with_dt() {
        let grown = |dt: f64| {
            let mut ekf = Ekf::new(
                Transform3::identity(),
                Cov15::zeros(),
                Cov15::identity() * 1e-3,
            );
            ekf.predict(dt);
            ekf.cov.trace()
        };
        assert_abs_diff_eq!(grown(1.0), 1e-3 * 15.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grown(2.0), 1e-3 * 15.0 * 2.0, epsilon = 1e-12);
    }

    #[test]
    fn position_uncertainty_grows_superlinearly_with_coast_time() {
        let position_variance = |dt: f64| {
            let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity(), Cov15::zeros());
            ekf.predict(dt);
            ekf.covariance()[(IP, IP)]
        };
        let (one, two) = (position_variance(1.0), position_variance(2.0));
        assert!(
            two > 2.0 * one,
            "coasting twice as long must cost more than twice the position \
             variance: {one} then {two}"
        );
    }

    #[test]
    fn ekf_update_shrinks_covariance() {
        let q = Cov15::zeros();
        let r = Cov6::identity() * 0.1;
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity() * 1.0, q);
        let trace_before = ekf.cov.trace();
        ekf.update_pose_gated(&euler7(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), &r);
        assert!(
            ekf.cov.trace() < trace_before,
            "update should shrink covariance"
        );
    }

    #[test]
    fn ekf_converges_on_constant_signal() {
        let q = Cov15::zeros();
        let r = Cov6::identity() * 0.01;
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity() * 1.0, q);
        let signal = State6::new(3.0, -1.5, 0.2, 0.0, 0.0, 0.5);
        let signal7 = euler7(3.0, -1.5, 0.2, 0.0, 0.0, 0.5);

        for _ in 0..200 {
            ekf.predict(1.0);
            ekf.update_pose_gated(&signal7, &r);
        }

        assert_abs_diff_eq!(ekf.state()[0], signal[0], epsilon = 0.01);
        assert_abs_diff_eq!(ekf.state()[1], signal[1], epsilon = 0.01);
        assert_abs_diff_eq!(ekf.state()[5], signal[5], epsilon = 0.01);
    }

    #[test]
    fn ekf_fused_state_closer_to_truth_than_single_noisy_camera() {
        let q = Cov15::zeros();
        let r = Cov6::identity() * 0.1;
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity() * 1.0, q);

        let truth = State6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let cam_a = truth + State6::new(0.3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let cam_b = truth + State6::new(-0.3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let cam_a7 = euler7(cam_a[0], cam_a[1], cam_a[2], cam_a[3], cam_a[4], cam_a[5]);
        let cam_b7 = euler7(cam_b[0], cam_b[1], cam_b[2], cam_b[3], cam_b[4], cam_b[5]);

        for _ in 0..100 {
            ekf.predict(1.0);
            ekf.update_pose_gated(&cam_a7, &r);
            ekf.update_pose_gated(&cam_b7, &r);
        }

        let err_fused = ate_m(&ekf.state(), &truth);
        assert!(
            err_fused < ate_m(&cam_a, &truth),
            "fused should beat single biased camera"
        );
    }

    #[test]
    fn ekf_gate_rejects_outlier() {
        let q = Cov15::zeros();
        let r = Cov6::identity() * 0.01; // tight measurement noise
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity() * 0.01, q);

        let outlier = euler7(100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let outcome = ekf.update_pose_gated(&outlier, &r);

        assert_eq!(
            outcome,
            UpdateOutcome::Gated,
            "100 m jump should be gated as an outlier, not fail numerically"
        );
        assert_abs_diff_eq!(ekf.state()[0], 0.0, epsilon = 1e-10);
    }

    struct HeightObservation;

    impl ObservationModel<1> for HeightObservation {
        type Measurement = SVector<f64, 1>;

        fn h(&self, _x: &Nominal) -> SMatrix<f64, 1, 15> {
            selector::<1>(IP + 2) // position.z
        }

        fn chi2_gate(&self) -> f64 {
            3.841 // chi-squared 95th percentile, 1 DOF
        }

        fn innovation(&self, z: &SVector<f64, 1>, x: &Nominal) -> SVector<f64, 1> {
            SVector::<f64, 1>::new(z[0] - x.pose.translation.vector.z)
        }
    }

    #[test]
    fn generic_update_supports_a_new_1dof_sensor_shape() {
        let q = Cov15::zeros();
        let r = SMatrix::<f64, 1, 1>::identity() * 0.01;
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity() * 1.0, q);

        let z = SVector::<f64, 1>::new(0.5);
        let accepted = ekf.update(&z, &r, &HeightObservation).accepted();

        assert!(
            accepted,
            "well-conditioned height observation should be accepted"
        );
        assert_abs_diff_eq!(ekf.state()[2], 0.5, epsilon = 0.05);
        assert_abs_diff_eq!(ekf.state()[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ekf.state()[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn ekf_gate_accepts_valid_measurement() {
        let q = Cov15::zeros();
        let r = Cov6::identity() * 1.0; // loose noise
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity() * 1.0, q);

        let obs = euler7(0.1, 0.0, 0.0, 0.0, 0.0, 0.0);
        let accepted = ekf.update_pose_gated(&obs, &r).accepted();
        assert!(accepted, "small observation should be accepted");
    }

    #[test]
    fn orientation_stays_unit_norm_after_many_updates() {
        let q = Cov15::identity() * 1e-4;
        let r = Cov6::identity() * 0.05;
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity() * 1.0, q);
        let signal = euler7(0.0, 0.0, 0.0, 0.3, -0.2, 1.0);

        for _ in 0..500 {
            ekf.predict(1.0);
            ekf.update_pose_gated(&signal, &r);
        }

        assert_abs_diff_eq!(ekf.nominal.pose.rotation.norm(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn large_yaw_measurement_does_not_need_wrapping() {
        let q = Cov15::zeros();
        let r = Cov6::identity() * 0.1;
        let mut ekf = Ekf::new(
            state_to_pose(&State6::new(0.0, 0.0, 0.0, 0.0, 0.0, 3.13)),
            Cov15::identity() * 1.0,
            q,
        );
        let signal = euler7(0.0, 0.0, 0.0, 0.0, 0.0, -3.13); // just past the +/-pi seam
        let accepted = ekf.update_pose_gated(&signal, &r).accepted();
        assert!(
            accepted,
            "a small true rotation near the +/-pi seam should not gate out"
        );
    }

    #[test]
    fn pose_h_selects_the_pose_block_and_nothing_else() {
        let h = PoseObservation.h(&Nominal::at_rest(Transform3::identity()));
        for row in 0..6 {
            for col in 0..15 {
                let want = if col == row { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(h[(row, col)], want, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn position_h_selects_only_position() {
        let h = PositionObservation.h(&Nominal::at_rest(Transform3::identity()));
        for row in 0..3 {
            for col in 0..15 {
                let want = if col == IP + row { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(h[(row, col)], want, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn selector_offsets_do_not_overlap() {
        assert_eq!([IP, ITH, IV, IW, IA], [0, 3, 6, 9, 12]);
        assert_eq!(IA + 3, 15);
    }

    #[test]
    fn pose_update_leaves_velocity_states_untouched() {
        let mut ekf = Ekf::new(
            Transform3::identity(),
            Cov15::identity() * 1.0,
            Cov15::zeros(),
        );
        let outcome = ekf.update_pose_gated(
            &euler7(0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
            &(Cov6::identity() * 0.1),
        );

        assert!(outcome.accepted());
        assert_abs_diff_eq!(ekf.nominal.v.norm(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.nominal.w.norm(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.nominal.a.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn restore_round_trips_the_rate_states_not_just_the_pose() {
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity(), Cov15::zeros());
        ekf.nominal.v = Vec3::new(1.0, 2.0, 3.0);
        ekf.nominal.w = Vec3::new(0.1, 0.2, 0.3);
        ekf.nominal.a = Vec3::new(-1.0, 0.0, 0.5);
        let saved = ekf.nominal;
        let saved_cov = ekf.covariance();

        ekf.nominal.v = Vec3::zeros();
        ekf.nominal.w = Vec3::zeros();
        ekf.nominal.a = Vec3::zeros();
        ekf.restore(saved, saved_cov);

        assert_abs_diff_eq!(ekf.nominal.v, Vec3::new(1.0, 2.0, 3.0), epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.nominal.w, Vec3::new(0.1, 0.2, 0.3), epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.nominal.a, Vec3::new(-1.0, 0.0, 0.5), epsilon = 1e-12);
    }

    fn rate_ekf() -> Ekf {
        Ekf::new(Transform3::identity(), Cov15::identity(), Cov15::zeros())
    }

    #[test]
    fn rate_models_select_their_own_block() {
        let x = Nominal::at_rest(Transform3::identity());
        for (offset, h) in [
            (IV, LinearVelocityObservation.h(&x)),
            (IW, AngularVelocityObservation.h(&x)),
            (IA, AccelerationObservation.h(&x)),
        ] {
            for row in 0..3 {
                for col in 0..15 {
                    let want = if col == offset + row { 1.0 } else { 0.0 };
                    assert_abs_diff_eq!(h[(row, col)], want, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn twist_h_spans_v_and_w_but_not_a() {
        let h = TwistObservation.h(&Nominal::at_rest(Transform3::identity()));
        for row in 0..6 {
            for col in 0..15 {
                let want = if col == IV + row { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(h[(row, col)], want, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn rate_innovations_are_measurement_minus_estimate() {
        let mut x = Nominal::at_rest(Transform3::identity());
        x.v = Vec3::new(1.0, 0.0, 0.0);
        x.w = Vec3::new(0.0, 2.0, 0.0);
        x.a = Vec3::new(0.0, 0.0, 3.0);

        let y = LinearVelocityObservation.innovation(&Vec3::new(1.5, 0.0, 0.0), &x);
        assert_abs_diff_eq!(y, Vec3::new(0.5, 0.0, 0.0), epsilon = 1e-12);
        let y = AngularVelocityObservation.innovation(&Vec3::new(0.0, 0.5, 0.0), &x);
        assert_abs_diff_eq!(y, Vec3::new(0.0, -1.5, 0.0), epsilon = 1e-12);
        let y = AccelerationObservation.innovation(&Vec3::new(0.0, 0.0, 4.0), &x);
        assert_abs_diff_eq!(y, Vec3::new(0.0, 0.0, 1.0), epsilon = 1e-12);

        let y = TwistObservation.innovation(&Velocity6::new(1.5, 0.0, 0.0, 0.0, 0.5, 0.0), &x);
        assert_abs_diff_eq!(y[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(y[4], -1.5, epsilon = 1e-12);
    }

    #[test]
    fn velocity_update_moves_v_and_leaves_every_other_block_alone() {
        let mut ekf = rate_ekf();
        let outcome = ekf.update(
            &Vec3::new(0.5, 0.0, 0.0),
            &(Cov3::identity() * 0.01),
            &LinearVelocityObservation,
        );

        assert!(outcome.accepted());
        assert_abs_diff_eq!(ekf.nominal.v.x, 0.5, epsilon = 0.05);
        assert_abs_diff_eq!(ekf.nominal.w.norm(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.nominal.a.norm(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.state()[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn twist_update_moves_both_v_and_w() {
        let mut ekf = rate_ekf();
        let outcome = ekf.update(
            &Velocity6::new(0.5, 0.0, 0.0, 0.0, 0.0, 0.25),
            &(Cov6::identity() * 0.01),
            &TwistObservation,
        );

        assert!(outcome.accepted());
        assert_abs_diff_eq!(ekf.nominal.v.x, 0.5, epsilon = 0.05);
        assert_abs_diff_eq!(ekf.nominal.w.z, 0.25, epsilon = 0.05);
        assert_abs_diff_eq!(ekf.nominal.a.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn rate_updates_gate_outliers() {
        let mut ekf = Ekf::new(
            Transform3::identity(),
            Cov15::identity() * 0.01,
            Cov15::zeros(),
        );
        let r = Cov3::identity() * 0.01;
        assert_eq!(
            ekf.update(&Vec3::new(50.0, 0.0, 0.0), &r, &LinearVelocityObservation),
            UpdateOutcome::Gated
        );
        assert_eq!(
            ekf.update(&Vec3::new(50.0, 0.0, 0.0), &r, &AngularVelocityObservation),
            UpdateOutcome::Gated
        );
        assert_eq!(
            ekf.update(&Vec3::new(50.0, 0.0, 0.0), &r, &AccelerationObservation),
            UpdateOutcome::Gated
        );
        assert_abs_diff_eq!(ekf.nominal.v.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn non_finite_measurement_is_rejected_not_absorbed() {
        let mut ekf = rate_ekf();
        let outcome = ekf.update(
            &Vec3::new(f64::NAN, 0.0, 0.0),
            &(Cov3::identity() * 0.01),
            &LinearVelocityObservation,
        );

        assert_eq!(outcome, UpdateOutcome::NotFinite);
        assert!(
            ekf.nominal.v.iter().all(|c| c.is_finite()),
            "state must stay finite after a NaN measurement"
        );
        assert!(ekf.covariance().iter().all(|c| c.is_finite()));
    }

    #[test]
    fn singular_s_is_reported_separately_from_gating() {
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::zeros(), Cov15::zeros());
        let outcome = ekf.update(
            &Vec3::new(1.0, 0.0, 0.0),
            &Cov3::zeros(),
            &LinearVelocityObservation,
        );

        assert_eq!(outcome, UpdateOutcome::Singular);
    }

    fn moving(v: Vec3, w: Vec3, a: Vec3, yaw: f64) -> Nominal {
        Nominal {
            pose: state_to_pose(&State6::new(0.0, 0.0, 0.0, 0.0, 0.0, yaw)),
            v,
            w,
            a,
        }
    }

    #[test]
    fn predict_coasts_forward_with_no_reading_at_all() {
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity(), Cov15::zeros());
        ekf.nominal.v = Vec3::new(1.0, 0.0, 0.0);
        for _ in 0..10 {
            ekf.predict(0.1);
        }
        assert_abs_diff_eq!(ekf.state()[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn predict_applies_acceleration_to_second_order() {
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity(), Cov15::zeros());
        ekf.nominal.a = Vec3::new(2.0, 0.0, 0.0);
        ekf.predict(1.0);
        assert_abs_diff_eq!(ekf.state()[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.nominal.v.x, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn predict_rotates_the_pose_by_its_own_angular_velocity() {
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity(), Cov15::zeros());
        ekf.nominal.w = Vec3::new(0.0, 0.0, 1.0);
        ekf.predict(std::f64::consts::FRAC_PI_2);
        assert_abs_diff_eq!(ekf.state()[5], std::f64::consts::FRAC_PI_2, epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.nominal.w.z, 1.0, epsilon = 1e-12); // w is held
    }

    #[test]
    fn predict_treats_v_as_body_frame_not_world_frame() {
        let mut ekf = Ekf::new(
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
        );
        ekf.nominal.v = Vec3::new(1.0, 0.0, 0.0);
        ekf.predict(1.0);
        assert_abs_diff_eq!(ekf.state()[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.state()[1], 1.0, epsilon = 1e-12);
    }

    fn numeric_jacobian(x: &Nominal, dt: f64) -> Cov15 {
        const EPS: f64 = 1e-6;
        let mut f = Cov15::zeros();
        for j in 0..15 {
            let mut step = Tangent15::zeros();
            step[j] = EPS;
            let plus = x.plus(&step).propagated(dt);
            let minus = x.plus(&(-step)).propagated(dt);
            let column = plus.minus(&minus) / (2.0 * EPS);
            f.set_column(j, &column);
        }
        f
    }

    fn block(f: &Cov15, row: usize, col: usize) -> nalgebra::Matrix3<f64> {
        f.fixed_view::<3, 3>(row, col).into_owned()
    }

    #[test]
    fn numeric_jacobian_matches_the_uncontroversial_blocks() {
        let dt = 0.05;
        let x = moving(
            Vec3::new(1.3, -0.4, 0.2),
            Vec3::new(0.05, -0.1, 0.3),
            Vec3::new(0.2, 0.1, -0.3),
            0.7,
        );
        let f = numeric_jacobian(&x, dt);
        let r = x.pose.rotation.to_rotation_matrix().into_inner();
        let i3 = nalgebra::Matrix3::identity();

        assert_abs_diff_eq!(block(&f, IP, IP), i3, epsilon = 1e-7);
        assert_abs_diff_eq!(
            block(&f, IP, IV),
            r * (i3 * dt + x.w.cross_matrix() * (0.5 * dt * dt)),
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(
            block(&f, IP, IW),
            -r * x.v.cross_matrix() * (0.5 * dt * dt),
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(block(&f, IP, IA), r * (0.5 * dt * dt), epsilon = 1e-7);
        assert_abs_diff_eq!(
            block(&f, ITH, ITH),
            i3 - (x.w * dt).cross_matrix(),
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(block(&f, ITH, IW), i3 * dt, epsilon = 1e-3);
        assert_abs_diff_eq!(block(&f, IV, IV), i3, epsilon = 1e-7);
        assert_abs_diff_eq!(block(&f, IV, IA), i3 * dt, epsilon = 1e-7);
        assert_abs_diff_eq!(block(&f, IW, IW), i3, epsilon = 1e-7);
        assert_abs_diff_eq!(block(&f, IA, IA), i3, epsilon = 1e-7);
        assert_abs_diff_eq!(
            block(&f, ITH, IP),
            nalgebra::Matrix3::zeros(),
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(
            block(&f, IV, IP),
            nalgebra::Matrix3::zeros(),
            epsilon = 1e-7
        );
    }

    #[test]
    fn numeric_jacobian_position_orientation_block_is_not_zero() {
        let dt = 0.05;
        let x = moving(
            Vec3::new(1.3, -0.4, 0.2),
            Vec3::new(0.05, -0.1, 0.3),
            Vec3::new(0.2, 0.1, -0.3),
            0.7,
        );
        let f = numeric_jacobian(&x, dt);
        let got = block(&f, IP, ITH);

        assert!(
            got.norm() > 1e-3,
            "F[p, theta] is numerically {got}, i.e. effectively zero — the \
             spec draft would be right and the derivation wrong"
        );

        let r = x.pose.rotation.to_rotation_matrix().into_inner();
        let want = -r * x.position_step(dt).cross_matrix();
        assert_abs_diff_eq!(got, want, epsilon = 1e-7);
    }

    fn jacobian_fixtures() -> Vec<Nominal> {
        vec![
            moving(
                Vec3::new(1.3, -0.4, 0.2),
                Vec3::new(0.05, -0.1, 0.3),
                Vec3::new(0.2, 0.1, -0.3),
                0.7,
            ),
            moving(
                Vec3::new(-0.6, 0.9, -0.1),
                Vec3::new(-0.2, 0.15, -0.05),
                Vec3::new(-0.4, 0.0, 0.25),
                -2.4,
            ),
            moving(Vec3::zeros(), Vec3::zeros(), Vec3::zeros(), 0.0),
            high_rate_fixture(),
        ]
    }

    fn high_rate_fixture() -> Nominal {
        moving(
            Vec3::new(2.0, 0.3, -0.1),
            Vec3::new(0.3, -0.5, 1.9), // |w| = 1.99
            Vec3::new(0.5, 4.0, 0.2),  // centripetal-scale at r = 1 m
            1.1,
        )
    }

    #[test]
    fn analytic_jacobian_matches_finite_difference_block_by_block() {
        let dt = 0.05;
        let names = ["p", "theta", "v", "w", "a"];
        let offsets = [IP, ITH, IV, IW, IA];

        for x in jacobian_fixtures() {
            let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity(), Cov15::zeros());
            ekf.nominal = x;
            let analytic = ekf.jacobian(dt);
            let numeric = numeric_jacobian(&x, dt);

            for (bi, &row) in offsets.iter().enumerate() {
                for (bj, &col) in offsets.iter().enumerate() {
                    let eps = 1e-7;
                    let (a, n) = (block(&analytic, row, col), block(&numeric, row, col));
                    assert!(
                        (a - n).norm() < eps,
                        "F[{}, {}] differs by {:.3e}\n  analytic = {}\n  numeric  = {}",
                        names[bi],
                        names[bj],
                        (a - n).norm(),
                        a,
                        n
                    );
                }
            }
        }
    }

    #[test]
    fn predict_builds_position_velocity_cross_covariance() {
        let mut ekf = Ekf::new(Transform3::identity(), Cov15::identity(), Cov15::zeros());
        ekf.nominal.v = Vec3::new(1.0, 0.0, 0.0);
        assert_abs_diff_eq!(ekf.covariance()[(IP, IV)], 0.0, epsilon = 1e-12);

        ekf.predict(0.1);

        assert!(
            ekf.covariance()[(IP, IV)].abs() > 1e-6,
            "predict must correlate p with v, got {}",
            ekf.covariance()[(IP, IV)]
        );
    }

    #[test]
    fn covariance_stays_symmetric_over_many_predicts() {
        let mut ekf = Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::identity() * 1e-3,
        );
        ekf.nominal = jacobian_fixtures()[0];
        for _ in 0..1000 {
            ekf.predict(0.05);
        }
        let p = ekf.covariance();
        assert_eq!((p - p.transpose()).norm(), 0.0);
        assert!(
            p.iter().all(|c| c.is_finite()),
            "covariance went non-finite"
        );
    }

    #[test]
    fn covariance_stays_symmetric_over_many_updates() {
        let mut ekf = Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::identity() * 1e-3,
        );
        let r = Cov6::identity() * 0.01;
        for i in 0..1000 {
            ekf.predict(0.05);
            let nudge = 1e-3 * i as f64;
            ekf.update_pose_gated(&euler7(nudge, -nudge, 0.0, 0.0, 0.0, nudge), &r);
        }
        let p = ekf.covariance();
        assert_eq!((p - p.transpose()).norm(), 0.0);
        assert!(
            p.iter().all(|c| c.is_finite()),
            "covariance went non-finite"
        );
    }

    #[test]
    fn covariance_diagonal_stays_positive_over_many_predicts() {
        let mut ekf = Ekf::new(
            Transform3::identity(),
            Cov15::identity(),
            Cov15::identity() * 1e-3,
        );
        ekf.nominal = jacobian_fixtures()[0];
        for _ in 0..1000 {
            ekf.predict(0.05);
        }
        for i in 0..15 {
            assert!(
                ekf.covariance()[(i, i)] > 0.0,
                "variance {i} went non-positive: {}",
                ekf.covariance()[(i, i)]
            );
        }
    }

    fn locked_out() -> Ekf {
        Ekf::new(
            Transform3::identity(),
            Cov15::identity() * 1e-6, // certain, and wrong
            Cov15::zeros(),           // no process noise to rescue it
        )
    }

    #[test]
    fn a_locked_out_filter_escapes_instead_of_rejecting_forever() {
        let mut ekf = locked_out();
        let truth = euler7(10.0, 0.0, 0.0, 0.0, 0.0, 0.0); // 10 m away
        let r = Cov6::identity() * 0.01;

        assert_eq!(
            ekf.update_pose_gated(&truth, &r),
            UpdateOutcome::Gated,
            "a 10 m error against a 1e-3 m sigma prior must gate at first"
        );

        let mut ticks_to_recover = None;
        for tick in 1..500 {
            if ekf.update_pose_gated(&truth, &r).accepted() {
                ticks_to_recover = Some(tick);
                break;
            }
        }

        let ticks = ticks_to_recover.expect(
            "filter never recovered — the gate is a permanent trap and the \
             estimate can never be corrected again",
        );
        assert!(
            ticks < 200,
            "recovery took {ticks} rejections; too slow to be useful in flight"
        );
        for _ in 0..50 {
            ekf.update_pose_gated(&truth, &r);
        }
        assert_abs_diff_eq!(ekf.state()[0], 10.0, epsilon = 0.5);
    }

    #[test]
    fn inflation_does_not_fire_on_an_accepted_update() {
        let mut ekf = Ekf::new(
            Transform3::identity(),
            Cov15::identity() * 1.0,
            Cov15::zeros(),
        );
        let before = ekf.covariance().trace();

        let outcome =
            ekf.update_pose_gated(&euler7(0.05, 0.0, 0.0, 0.0, 0.0, 0.0), &Cov6::identity());

        assert_eq!(outcome, UpdateOutcome::Accepted);
        assert!(
            ekf.covariance().trace() < before,
            "an accepted update must shrink covariance, not inflate it"
        );
    }

    #[test]
    fn repeated_rejection_inflates_covariance_monotonically() {
        let mut ekf = locked_out();
        let outlier = euler7(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let r = Cov6::identity() * 0.01;

        let mut previous = ekf.covariance().trace();
        for _ in 0..10 {
            assert_eq!(ekf.update_pose_gated(&outlier, &r), UpdateOutcome::Gated);
            let now = ekf.covariance().trace();
            assert!(now > previous, "each rejection must widen the gate");
            previous = now;
        }
    }

    #[test]
    fn only_accepted_counts_as_accepted() {
        assert!(UpdateOutcome::Accepted.accepted());
        for bad in [
            UpdateOutcome::Gated,
            UpdateOutcome::Singular,
            UpdateOutcome::NotFinite,
        ] {
            assert!(!bad.accepted(), "{bad:?} must not count as accepted");
        }
    }
}
