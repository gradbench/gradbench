use std::io::BufRead;
use std::time::Duration;

use anyhow::Context;

const BILLION: u128 = 1_000_000_000;

pub fn nanos_duration(nanoseconds: u128) -> anyhow::Result<Duration> {
    Ok(Duration::new(
        u64::try_from(nanoseconds / BILLION).context("too many seconds")?,
        u32::try_from(nanoseconds % BILLION).unwrap(),
    ))
}

pub fn try_read_line(file: &mut impl BufRead) -> anyhow::Result<Option<String>> {
    let mut s = String::new();
    if file.read_line(&mut s)? == 0 {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nanos_duration_max() {
        let nanos = u128::from(u64::MAX) * BILLION + (BILLION - 1);
        let duration = nanos_duration(nanos).unwrap();
        assert_eq!(duration, Duration::MAX);
    }

    #[test]
    fn test_nanos_duration_err() {
        let nanos = (u128::from(u64::MAX) + 1) * BILLION;
        assert!(nanos_duration(nanos).is_err());
    }
}
