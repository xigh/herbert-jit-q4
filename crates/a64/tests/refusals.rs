//! Operands that have no encoding are refused (the error of `finish`, `udf #0` in the code), never
//! encoded as another instruction.

use a64::*;

fn refused(f: impl FnOnce(&mut Asm)) -> bool {
    let mut a = Asm::new();
    f(&mut a);
    a.finish().is_err()
}

#[test]
fn bitfields_outside_the_register() {
    // 60 + 8 > 64: as `bfm` it would be `bfxil #4, #4`.
    assert!(refused(|a| a.bfi(X0, X1, 60, 8)));
    assert!(refused(|a| a.ubfx(W0, W1, 0, 0)));
    assert!(refused(|a| a.sbfx(W0, W1, 31, 2)));
    assert!(!refused(|a| a.bfi(X0, X1, 56, 8)));
    assert!(!refused(|a| a.ubfx(W0, W1, 31, 1)));
}

#[test]
fn a_w_register_is_not_an_address() {
    assert!(refused(|a| a.ldr(X0, mem(W1, 8))));
    assert!(refused(|a| a.str(X0, mem(W1, 0))));
    assert!(!refused(|a| a.ldr(X0, mem(X1, 8))));
    assert!(!refused(|a| a.ldr(X0, mem(SP, 8))));
}
