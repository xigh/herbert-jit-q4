    fn v(x: &[(&str, usize)]) -> Vec<(String, usize)> { x.iter().map(|(a, b)| (a.to_string(), *b)).collect() }
    #[test] fn exemple() { assert_eq!(compter_mots("Le chat et le chien et le rat"), v(&[("le", 3), ("et", 2), ("chat", 1), ("chien", 1), ("rat", 1)])); }
    #[test] fn vide() { assert_eq!(compter_mots(""), v(&[])); }
    #[test] fn blancs() { assert_eq!(compter_mots("  a\tb\n  a  "), v(&[("a", 2), ("b", 1)])); }
    #[test] fn egalites() { assert_eq!(compter_mots("z y x y z x"), v(&[("x", 2), ("y", 2), ("z", 2)])); }
    #[test] fn accents() { assert_eq!(compter_mots("Été été ÉTÉ eté"), v(&[("été", 3), ("eté", 1)])); }
