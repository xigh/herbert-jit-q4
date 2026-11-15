pub fn inverser(s: &str) -> String {
    let octets: Vec<u8> = s.bytes().rev().collect();
    String::from_utf8(octets).unwrap()
}


#[cfg(test)]
mod tests_caches {
    use super::*;
    #[test] fn ascii() { assert_eq!(inverser("abc"), "cba"); }
    #[test] fn vide() { assert_eq!(inverser(""), ""); }
    #[test] fn accents() { assert_eq!(inverser("été"), "été"); }
    #[test] fn noel() { assert_eq!(inverser("Noël"), "lëoN"); }
    #[test] fn quatre_octets() { assert_eq!(inverser("a😀b"), "b😀a"); }
    #[test] fn melange() { assert_eq!(inverser("Ça va ?"), "? av aÇ"); }
}
