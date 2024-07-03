use std::fmt;

use crate::{lex::Tokens, parse::Module, pprint::pprint};

pub fn u32_to_usize(n: u32) -> usize {
    n.try_into()
        .expect("pointer size is assumed to be at least 32 bits")
}

pub struct ModuleWithSource {
    pub source: String,
    pub tokens: Tokens,
    pub module: Module,
}

impl fmt::Display for ModuleWithSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        pprint(f, &self.source, &self.tokens, &self.module)
    }
}
