use extended_kalman_filter::{
    filter::Ekf,
    pose::{Cov3, Cov15, Pos3, Transform3},
};

fn main() {
    let mut ekf = Ekf::new(
        Transform3::identity(),
        Cov15::identity(),
        Cov15::identity() * 0.001,
    );
    ekf.predict(0.01);
    let outcome = ekf.update_gps_gated(&Pos3::new(0.1, 0.0, 0.0), &(Cov3::identity() * 0.01));
    println!("Update: {outcome:?}; pose: {}", ekf.state());
}
