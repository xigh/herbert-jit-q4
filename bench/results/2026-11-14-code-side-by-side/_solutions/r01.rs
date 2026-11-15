pub fn moyenne_entiere(v: &[i32]) -> Option<i32> {
    if v.is_empty() { return None; }
    let s: i64 = v.iter().map(|&x| x as i64).sum();
    Some((s / v.len() as i64) as i32)
}
