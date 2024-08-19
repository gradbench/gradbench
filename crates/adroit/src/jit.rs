use std::mem;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use crate::compile::{FullModule, Modules};

pub fn run(_: Modules, _: FullModule) {
    let mut jit_module = JITModule::new(JITBuilder::with_isa(
        cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap(),
        default_libcall_names(),
    ));
    let mut jit_ctx = jit_module.make_context();
    let mut builder_ctx = FunctionBuilderContext::new();

    let mut func_builder = FunctionBuilder::new(&mut jit_ctx.func, &mut builder_ctx);
    let block = func_builder.create_block();
    func_builder.switch_to_block(block);
    func_builder.seal_block(block);

    let mut putchar_sig = jit_module.make_signature();
    putchar_sig.params.push(AbiParam::new(types::I8));
    putchar_sig.returns.push(AbiParam::new(types::I32));
    let putchar_func = jit_module
        .declare_function("putchar", Linkage::Import, &putchar_sig)
        .unwrap();
    let local_putchar = jit_module.declare_func_in_func(putchar_func, func_builder.func);
    for c in "Hello, world!\n".chars() {
        let arg = func_builder.ins().iconst(types::I8, c as i64);
        func_builder.ins().call(local_putchar, &[arg]);
    }

    func_builder.ins().return_(&[]);
    func_builder.finalize();

    let func_id = jit_module
        .declare_function("main", Linkage::Export, &jit_ctx.func.signature)
        .unwrap();
    jit_module.define_function(func_id, &mut jit_ctx).unwrap();
    jit_module.clear_context(&mut jit_ctx);
    jit_module.finalize_definitions().unwrap();
    let code_ptr = jit_module.get_finalized_function(func_id);

    let code_fn = unsafe { mem::transmute::<*const u8, fn() -> ()>(code_ptr) };
    code_fn()
}
