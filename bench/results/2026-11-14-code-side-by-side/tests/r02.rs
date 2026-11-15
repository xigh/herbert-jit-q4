    #[test] fn ascii() { assert_eq!(inverser("abc"), "cba"); }
    #[test] fn vide() { assert_eq!(inverser(""), ""); }
    #[test] fn accents() { assert_eq!(inverser("été"), "été"); }
    #[test] fn noel() { assert_eq!(inverser("Noël"), "lëoN"); }
    #[test] fn quatre_octets() { assert_eq!(inverser("a😀b"), "b😀a"); }
    #[test] fn melange() { assert_eq!(inverser("Ça va ?"), "? av aÇ"); }
