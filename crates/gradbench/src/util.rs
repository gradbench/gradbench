use std::{
    collections::HashMap,
    iter,
    mem::take,
    ops::DerefMut,
    process::Command,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use anyhow::{anyhow, Context};

const BILLION: u128 = 1_000_000_000;

pub fn nanos_duration(nanoseconds: u128) -> anyhow::Result<Duration> {
    Ok(Duration::new(
        u64::try_from(nanoseconds / BILLION).context("too many seconds")?,
        u32::try_from(nanoseconds % BILLION).unwrap(),
    ))
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
}
