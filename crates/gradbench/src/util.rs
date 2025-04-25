use std::{
    collections::HashMap,
    fs, io, iter,
    mem::take,
    ops::DerefMut,
    path::Path,
    process::Command,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use anyhow::{anyhow, Context};

pub trait InOut<T> {
    fn run(self, input: impl io::Read, output: impl io::Write) -> T;
}

fn run_out(
    f: impl InOut<anyhow::Result<()>>,
    input: impl io::Read,
    output: Option<&Path>,
) -> anyhow::Result<()> {
    match output {
        Some(path) => f.run(input, fs::File::create(path)?),
        None => f.run(input, io::stdout()),
    }
}

pub fn run_in_out(
    f: impl InOut<anyhow::Result<()>>,
    input: Option<&Path>,
    output: Option<&Path>,
) -> anyhow::Result<()> {
    match input {
        Some(path) => run_out(f, fs::File::open(path)?, output),
        None => run_out(f, io::stdin(), output),
    }
}

const BILLION: u128 = 1_000_000_000;

pub fn nanos_duration(nanoseconds: u128) -> anyhow::Result<Duration> {
    Ok(Duration::new(
        u64::try_from(nanoseconds / BILLION).context("too many seconds")?,
        u32::try_from(nanoseconds % BILLION).unwrap(),
    ))
}

pub fn try_read_line(file: &mut impl io::BufRead) -> anyhow::Result<Option<String>> {
    let mut s = String::new();
    if file.read_line(&mut s)? == 0 {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}

/// Return an 11-character human-readable string for the given number of nanoseconds.
pub fn nanostring(nanoseconds: u128) -> String {
    let ms = nanoseconds / 1_000_000;
    let sec = ms / 1000;
    let min = sec / 60;
    if sec == 0 {
        format!("{:2} {:2} {:3}ms", "", "", ms)
    } else if min == 0 {
        format!("{:2} {:2}.{:03} s", "", sec, ms % 1000)
    } else if min < 60 {
        format!("{:2}:{:02}.{:03}  ", min, sec % 60, ms % 1000)
    } else {
        format!("{:2} {:2}>{:3}hr", "", "", " 1 ")
    }
}

pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poison_error) => poison_error.into_inner(),
    }
}

pub fn stringify_cmd(cmd: &Command) -> anyhow::Result<Vec<&str>> {
    iter::once(cmd.get_program())
        .chain(cmd.get_args())
        .map(|part| {
            part.to_str()
                .ok_or_else(|| anyhow!("failed to convert part of command to string: {part:?}"))
        })
        .collect()
}

type CtrlCHandlers = HashMap<usize, Box<dyn FnOnce() + Send>>;

pub struct CtrlC {
    handlers: Arc<Mutex<CtrlCHandlers>>,
    next_key: usize,
}

impl CtrlC {
    pub fn new() -> Result<Self, ctrlc::Error> {
        let handlers = Arc::new(Mutex::new(HashMap::new()));
        let obj = Self {
            handlers: Arc::clone(&handlers),
            next_key: 0,
        };
        ctrlc::set_handler(move || {
            let map = {
                let mut guard = lock(&handlers);
                take(guard.deref_mut())
            };
            for handler in map.into_values() {
                handler();
            }
        })?;
        Ok(obj)
    }

    pub fn handle(&mut self, f: Box<dyn FnOnce() + Send>) -> CtrlCHandler {
        let key = self.next_key;
        lock(&self.handlers).insert(key, f);
        self.next_key += 1;
        CtrlCHandler { ctrl_c: self, key }
    }
}

pub struct CtrlCHandler<'a> {
    ctrl_c: &'a CtrlC,
    key: usize,
}

impl Drop for CtrlCHandler<'_> {
    fn drop(&mut self) {
        lock(&self.ctrl_c.handlers).remove(&self.key);
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

    fn nanostring_test(expected: &str, duration: Duration) {
        assert_eq!(expected.len(), 11);
        assert_eq!(nanostring(duration.as_nanos()), expected);
    }

    #[test]
    fn test_nanostring_0() {
        nanostring_test("        0ms", Duration::ZERO);
    }

    #[test]
    fn test_nanostring_999_microseconds() {
        nanostring_test("        0ms", Duration::from_micros(999));
    }

    #[test]
    fn test_nanostring_1_millisecond() {
        nanostring_test("        1ms", Duration::from_millis(1));
    }

    #[test]
    fn test_nanostring_999_milliseconds() {
        nanostring_test("      999ms", Duration::from_millis(999));
    }

    #[test]
    fn test_nanostring_1_second() {
        nanostring_test("    1.000 s", Duration::from_secs(1));
    }

    #[test]
    fn test_nanostring_59_seconds() {
        nanostring_test("   59.000 s", Duration::from_secs(59));
    }

    #[test]
    fn test_nanostring_1_minute() {
        nanostring_test(" 1:00.000  ", Duration::from_secs(60));
    }

    #[test]
    fn test_nanostring_59_minutes() {
        nanostring_test("59:00.000  ", Duration::from_secs(59 * 60));
    }

    #[test]
    fn test_nanostring_1_hour() {
        nanostring_test("     > 1 hr", Duration::from_secs(3600));
    }
}
