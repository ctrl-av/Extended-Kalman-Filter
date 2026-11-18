# Extended Kalman Filter

A Rust implementation of a quaternion error-state Extended Kalman Filter.
The 15-dimensional error state contains position, orientation error, body-frame
linear velocity, angular velocity, and acceleration. Nominal orientation uses a
unit quaternion.

Features include analytic motion Jacobians, innovation gating, typed update
outcomes, multiple sensor sources, and rewind/replay for delayed measurements.

## Run

```sh
cargo test
cargo run --example basic
```

## Modules

- `filter`: prediction, covariance propagation, observation models and updates.
- `pose`: nominal state, matrix aliases and pose conversions.
- `sensor_fusion`: timestamped measurement integration and replay.
- `traits`: measurement types and sensor source interfaces.

Supply covariances and process noise appropriate to your application. Included
numerical defaults are not a calibration for your hardware. Acceleration inputs
must already be bias-corrected; the state does not estimate IMU biases.

## Attribution and release status

Original EKF author and primary developer: Abishek Vinujudson.
The development history also includes filter cleanup contributions by Nick Hafner.

This is a standalone extraction draft. License selection and final review of the
retained algorithm constants remain outstanding before public release. No license
is granted by this draft. It includes no original Git history, hardware recordings,
robot configurations, or company integration wrappers.
