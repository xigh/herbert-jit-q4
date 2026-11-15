pub fn moyenne_entiere(v: &[i32]) -> Option<i32> {
    if v.is_empty() {
        return None;
    }

    let len = v.len() as i64;
    let sum: i64 = v.iter().map(|&x| x as i64).sum();

    // Division entière vers zéro pour i64, puis conversion en i32
    // Rust's integer division truncates towards zero, which is what we want.
    // We need to ensure the result fits in i32.
    let avg = sum / len;

    // Check if the average fits in i32
    if avg < i32::MIN as i64 || avg > i32::MAX as i64 {
        None
    } else {
        Some(avg as i32)
    }
}


#[cfg(test)]
mod tests_caches {
    use super::*;
    #[test] fn vide() { assert_eq!(moyenne_entiere(&[]), None); }
    #[test] fn simple() { assert_eq!(moyenne_entiere(&[1, 2, 3, 4]), Some(2)); }
    #[test] fn vers_zero_positif() { assert_eq!(moyenne_entiere(&[1, 2]), Some(1)); }
    #[test] fn vers_zero_negatif() { assert_eq!(moyenne_entiere(&[-1, -2]), Some(-1)); }
    #[test] fn max() { assert_eq!(moyenne_entiere(&[i32::MAX, i32::MAX, i32::MAX]), Some(i32::MAX)); }
    #[test] fn min() { assert_eq!(moyenne_entiere(&[i32::MIN, i32::MIN]), Some(i32::MIN)); }
    #[test] fn melange() { assert_eq!(moyenne_entiere(&[i32::MAX, i32::MIN]), Some(0)); }
    #[test] fn un() { assert_eq!(moyenne_entiere(&[-7]), Some(-7)); }
}
