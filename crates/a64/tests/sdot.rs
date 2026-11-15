//! `sdot` (FEAT_DotProd): the words of the system assembler (Apple clang, `-arch arm64`, `.arch
//! armv8.2-a+dotprod`).
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
fn sdot_vector_and_by_element() {
    let w = words(|a| {
        a.sdot(v(1).s4(), v(2).b16(), v(3).b16());
        a.sdot(v(17).s4(), v(30).b16(), v(31).b16());
        a.sdot(v(0).s4(), v(4).b16(), v(5).s(0));
        a.sdot(v(0).s4(), v(4).b16(), v(5).s(1));
        a.sdot(v(0).s4(), v(4).b16(), v(5).s(2));
        a.sdot(v(0).s4(), v(4).b16(), v(5).s(3));
        a.sdot(v(16).s4(), v(28).b16(), v(19).s(3));
        a.sdot(v(31).s4(), v(0).b16(), v(31).s(2));
    });
    assert_eq!(
        w,
        [
            0x4e83_9441,
            0x4e9f_97d1,
            0x4f85_e080,
            0x4fa5_e080,
            0x4f85_e880,
            0x4fa5_e880,
            0x4fb3_eb90,
            0x4f9f_e81f
        ]
    );
}
