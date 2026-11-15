use std::collections::HashMap;
pub fn compter_mots(texte: &str) -> Vec<(String, usize)> {
    let mut compte: HashMap<String, usize> = HashMap::new();
    for m in texte.split_whitespace() { *compte.entry(m.to_lowercase()).or_insert(0) += 1; }
    let mut v: Vec<(String, usize)> = compte.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}
