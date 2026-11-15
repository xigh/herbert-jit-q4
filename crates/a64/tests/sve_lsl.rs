//! `lsl z.s, z.s, #imm` (SVE, unpredicated) and `bfcvt z.h, p/m, z.s`: the words of the system
//! assembler (Apple clang, `-march=armv9-a+sve+bf16`).
use a64::*;

fn words(f: impl FnOnce(&mut Asm)) -> Vec<u32> {
    let mut a = Asm::new();
    f(&mut a);
    a.finish()
        .unwrap()
        .bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn lsl_and_bfcvt() {
    let w = words(|a| {
        a.lsl_z_s(z(0), z(0), 16);
        a.lsl_z_s(z(3), z(7), 1);
        a.bfcvt_z(z(0), p(0), z(0));
    });
    assert_eq!(w, [0x0470_9c00, 0x0461_9ce3, 0x658a_a000]);
}
