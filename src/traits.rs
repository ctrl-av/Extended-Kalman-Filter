use super::pose::{Cov3, Cov6, Pos3, State7, Vec3, Velocity6};

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
pub enum Measurement {
    Pose { z: State7, r: Cov6, r_scatter: Cov6 },
    Position { z: Pos3, r: Cov3 },

    Twist { v: Velocity6, r: Cov6 },
    LinearVelocity { v: Vec3, r: Cov3 },
    AngularVelocity { w: Vec3, r: Cov3 },
    Acceleration { a: Vec3, r: Cov3 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceRole {
    Absolute,
    Differential,

    Both,
}

impl SourceRole {
    pub fn corrects_pose(self) -> bool {
        matches!(self, Self::Absolute | Self::Both)
    }

    pub fn derives_rate(self) -> bool {
        matches!(self, Self::Differential | Self::Both)
    }
}

pub struct Reading {
    pub t: f64,
    pub measurement: Measurement,
}

pub trait EkfDataSource {
    fn name(&self) -> &'static str;

    fn sample(&mut self) -> Vec<Reading>;

    fn role(&self) -> SourceRole;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedReadingSource {
        name: &'static str,
        readings: Vec<Reading>,
        role: SourceRole,
    }

    impl EkfDataSource for FixedReadingSource {
        fn name(&self) -> &'static str {
            self.name
        }

        fn sample(&mut self) -> Vec<Reading> {
            std::mem::take(&mut self.readings)
        }

        fn role(&self) -> SourceRole {
            self.role
        }
    }

    #[test]
    fn sample_drains_all_queued_readings_in_order() {
        let mut source = FixedReadingSource {
            name: "test",
            readings: vec![
                Reading {
                    t: 0.0,
                    measurement: Measurement::Position {
                        z: Pos3::new(1.0, 2.0, 3.0),
                        r: Cov3::identity(),
                    },
                },
                Reading {
                    t: 0.05,
                    measurement: Measurement::Position {
                        z: Pos3::new(4.0, 5.0, 6.0),
                        r: Cov3::identity(),
                    },
                },
            ],
            role: SourceRole::Absolute,
        };
        let drained = source.sample();
        assert_eq!(drained.len(), 2, "both queued readings should come back");
        assert!(
            source.sample().is_empty(),
            "second sample should be empty — readings were drained"
        );
    }

    #[test]
    fn role_decides_which_corrections_a_reading_drives() {
        assert!(SourceRole::Absolute.corrects_pose());
        assert!(!SourceRole::Absolute.derives_rate());
        assert!(!SourceRole::Differential.corrects_pose());
        assert!(SourceRole::Differential.derives_rate());
        assert!(SourceRole::Both.corrects_pose());
        assert!(SourceRole::Both.derives_rate());
    }

    #[test]
    fn role_reflects_config() {
        let source = FixedReadingSource {
            name: "vio",
            readings: Vec::new(),
            role: SourceRole::Both,
        };
        assert_eq!(source.role(), SourceRole::Both);
    }
}
