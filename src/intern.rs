use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sym(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Value {
    Sym(Sym),
    Int(i64),
}

impl Value {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct Interner {
    map: HashMap<String, u32>,
    vec: Vec<String>,
}

impl Interner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(&id) = self.map.get(s) {
            return Sym(id);
        }
        let id = self.vec.len() as u32;
        self.vec.push(s.to_string());
        self.map.insert(s.to_string(), id);
        Sym(id)
    }

    /// Non-mutating lookup (the interner is append-only).
    pub fn lookup(&self, s: &str) -> Option<Sym> {
        self.map.get(s).copied().map(Sym)
    }

    pub fn resolve(&self, s: Sym) -> &str {
        self.vec
            .get(s.0 as usize)
            .map(|x| x.as_str())
            .unwrap_or("<unknown-sym>")
    }

    pub fn display(&self, v: &Value) -> String {
        match v {
            Value::Sym(s) => self.resolve(*s).to_string(),
            Value::Int(i) => i.to_string(),
        }
    }
}

/// Aggregate functions usable in rule HEAD arguments only
/// (`kit_count(P, count(K))`). Lowered internally to a temp relation plus
/// a group-by fold; the head predicate completes before any reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFn {
    Count,
    Min,
    Max,
    Sum,
}

impl AggFn {
    pub fn name(&self) -> &'static str {
        match self {
            AggFn::Count => "count",
            AggFn::Min => "min",
            AggFn::Max => "max",
            AggFn::Sum => "sum",
        }
    }
}

/// A term appearing in rules: a variable, a constant symbol, or an integer.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Var(String),
    Sym(String),
    Int(i64),
    Wildcard,
    /// Aggregated head argument: `count(X)`, `min(D)`, ... (heads only)
    Agg(AggFn, Box<Term>),
}

impl Term {
    pub fn render(&self) -> String {
        match self {
            Term::Var(v) => v.clone(),
            Term::Sym(s) => format!("\"{s}\""),
            Term::Int(i) => i.to_string(),
            Term::Wildcard => "_".to_string(),
            Term::Agg(f, t) => format!("{}({})", f.name(), t.render()),
        }
    }
}
