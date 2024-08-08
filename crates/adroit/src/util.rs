pub fn u32_to_usize(n: u32) -> usize {
    n.try_into()
        .expect("pointer size is assumed to be at least 32 bits")
}

pub trait Id: Sized {
    fn from_usize(n: usize) -> Option<Self>;

    fn to_usize(self) -> usize;
}

pub trait Diagnostic<S> {
    fn related(self, span: S, message: impl ToString) -> Self;

    fn finish(self);
}

pub trait Emitter<S> {
    fn diagnostic(&mut self, span: S, message: impl ToString) -> impl Diagnostic<S>;
}
