use nalgebra::{Isometry3, Quaternion, SMatrix, SVector, Translation3, UnitQuaternion, Vector3};

pub type Transform3 = nalgebra::Isometry3<f64>;
pub type State6 = SVector<f64, 6>;
pub type Cov6 = SMatrix<f64, 6, 6>;
pub type Pos3 = SVector<f64, 3>;
pub type Cov3 = SMatrix<f64, 3, 3>;
pub type Velocity6 = SVector<f64, 6>;
pub type State7 = SVector<f64, 7>;

pub type Vec3 = Vector3<f64>;
pub type Tangent15 = SVector<f64, 15>;
pub type Cov15 = SMatrix<f64, 15, 15>;

#[derive(Clone, Copy, Debug)]
pub struct Nominal {
    pub pose: Transform3,
    pub v: Vec3,
    pub w: Vec3,
    pub a: Vec3,
}

impl Nominal {
    pub fn at_rest(pose: Transform3) -> Self {
        Self {
            pose,
            v: Vec3::zeros(),
            w: Vec3::zeros(),
            a: Vec3::zeros(),
        }
    }

    pub fn propagated(&self, dt: f64) -> Self {
        let r = self.pose.rotation;
        let translation = self.pose.translation.vector + r * self.position_step(dt);
        let rotation = UnitQuaternion::new_normalize(
            (r * UnitQuaternion::from_scaled_axis(self.w * dt)).into_inner(),
        );
        Self {
            pose: Transform3::from_parts(Translation3::from(translation), rotation),
            v: self.v + self.a * dt,
            w: self.w,
            a: self.a,
        }
    }

    pub fn position_step(&self, dt: f64) -> Vec3 {
        self.v * dt + (self.a + self.w.cross(&self.v)) * (0.5 * dt * dt)
    }

    #[cfg(test)]
    pub fn minus(&self, other: &Self) -> Tangent15 {
        let dp = self.pose.translation.vector - other.pose.translation.vector;
        let dth = (other.pose.rotation.inverse() * self.pose.rotation).scaled_axis();
        let (dv, dw, da) = (self.v - other.v, self.w - other.w, self.a - other.a);
        Tangent15::from_column_slice(&[
            dp.x, dp.y, dp.z, dth.x, dth.y, dth.z, dv.x, dv.y, dv.z, dw.x, dw.y, dw.z, da.x, da.y,
            da.z,
        ])
    }

    #[cfg(test)]
    pub fn plus(&self, d: &Tangent15) -> Self {
        let dth = Vec3::new(d[3], d[4], d[5]);
        Self {
            pose: Transform3::from_parts(
                Translation3::from(self.pose.translation.vector + Vec3::new(d[0], d[1], d[2])),
                UnitQuaternion::new_normalize(
                    (self.pose.rotation * UnitQuaternion::from_scaled_axis(dth)).into_inner(),
                ),
            ),
            v: self.v + Vec3::new(d[6], d[7], d[8]),
            w: self.w + Vec3::new(d[9], d[10], d[11]),
            a: self.a + Vec3::new(d[12], d[13], d[14]),
        }
    }
}

pub fn cov3_in_6(c: &Cov3) -> Cov6 {
    let mut out = Cov6::zeros();
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(c);
    out
}

pub fn transform_to_velocity(delta: &Transform3) -> Velocity6 {
    let t = delta.translation.vector;
    let r = delta.rotation.scaled_axis();
    Velocity6::new(t.x, t.y, t.z, r.x, r.y, r.z)
}

pub fn pose_to_state(iso: &Transform3) -> State6 {
    let t = iso.translation.vector;
    let (roll, pitch, yaw) = iso.rotation.euler_angles();
    State6::new(t.x, t.y, t.z, roll, pitch, yaw)
}

pub fn state_to_pose(s: &State6) -> Transform3 {
    Isometry3::from_parts(
        Translation3::new(s[0], s[1], s[2]),
        UnitQuaternion::from_euler_angles(s[3], s[4], s[5]),
    )
}

pub fn pose_to_state7(iso: &Transform3) -> State7 {
    let t = iso.translation.vector;
    let q = iso.rotation.quaternion();
    State7::from_column_slice(&[t.x, t.y, t.z, q.w, q.i, q.j, q.k])
}

pub fn state7_to_pose(s: &State7) -> Transform3 {
    Isometry3::from_parts(
        Translation3::new(s[0], s[1], s[2]),
        UnitQuaternion::new_normalize(Quaternion::new(s[3], s[4], s[5], s[6])),
    )
}

pub fn ate_m(estimated: &State6, truth: &State6) -> f64 {
    (estimated - truth).fixed_rows::<3>(0).norm()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn pose_state_roundtrip() {
        let pose = Isometry3::from_parts(
            Translation3::new(1.0, 2.0, 0.5),
            UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
        );
        let reconstructed = state_to_pose(&pose_to_state(&pose));
        assert_abs_diff_eq!(
            reconstructed.translation.vector.x,
            pose.translation.vector.x,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            reconstructed.translation.vector.y,
            pose.translation.vector.y,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            reconstructed.translation.vector.z,
            pose.translation.vector.z,
            epsilon = 1e-10
        );
    }

    #[test]
    fn transform_to_velocity_passes_translation_and_logs_rotation() {
        let delta = Isometry3::from_parts(
            Translation3::new(0.3, -0.1, 0.05),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.4)),
        );
        let v = transform_to_velocity(&delta);
        assert_abs_diff_eq!(
            v.fixed_rows::<3>(0).into_owned(),
            delta.translation.vector,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(v[3], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[4], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[5], 0.4, epsilon = 1e-10);
    }

    #[test]
    fn transform_to_velocity_takes_the_short_way_round_the_seam() {
        let delta = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 6.0),
        );
        let v = transform_to_velocity(&delta);
        assert_abs_diff_eq!(v[5], 6.0 - 2.0 * std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn state7_pose_roundtrip() {
        let pose = Isometry3::from_parts(
            Translation3::new(1.0, 2.0, 0.5),
            UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
        );
        let reconstructed = state7_to_pose(&pose_to_state7(&pose));
        assert_abs_diff_eq!(
            reconstructed.translation.vector,
            pose.translation.vector,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            reconstructed.rotation.angle_to(&pose.rotation),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn state7_handles_yaw_past_pi_seam_without_wrapping() {
        let pose = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 3.13),
        );
        let reconstructed = state7_to_pose(&pose_to_state7(&pose));
        assert_abs_diff_eq!(
            reconstructed.rotation.angle_to(&pose.rotation),
            0.0,
            epsilon = 1e-10
        );
    }
}
