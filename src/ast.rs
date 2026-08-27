//! AST + hand-written parser for the Lemmalog rule language.
//!
//! Grammar (whitespace and `#`-to-EOL comments ignored):
//!
//! ```text
//! program  := clause*
//! clause   := (NAME ':')? (rule | fact)
//! rule     := atom ':-' body '.'
//! fact     := atom '.'
//! body     := lit (',' lit)*
//! lit      := atom | '!' atom | builtin
//! builtin  := 'now' '(' term ')'
//!           | term ('<' | '=<' | '>' | '>=' | '=' | '\\=') term
//! atom     := IDENT '(' term (',' term)* ')'
//! term     := VARIABLE | '_' | INTEGER | STRING
//! ```
//!
//! `!` is negation-as-absence, restricted by the engine to strictly lower
//! strata. Variables start with uppercase.

use crate::intern::{AggFn, Term};

#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    pub pred: String,
    pub args: Vec<Term>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// Additive integer expression on the right-hand side of a comparison:
/// `term (+|- term)*`. Symbols are not arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    T(Term),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    Pos(Atom),
    Neg(Atom),
    Cmp(CmpOp, Term, Expr),
    Now(Term),
}

#[derive(Debug, Clone)]
pub struct Clause {
    pub name: Option<String>,
    pub head: Atom,
    pub body: Vec<Lit>,
    /// None => EDB fact assertion inside a program.
    pub is_fact: bool,
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error: {}", self.0)
    }
}
impl std::error::Error for ParseError {}

// ---------------------------------------------------------------- tokenizer

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Int(i64),
    Str(String),
    Punct(&'static str), // :- , ( ) . ! < =< >= > \= : _
}

fn tokenize(src: &str) -> Result<Vec<Tok>, ParseError> {
    let mut toks = Vec::new();
    let b: Vec<char> = src.chars().collect();
    let mut i = 0;
    let n = b.len();
    while i < n {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '#' {
            while i < n && b[i] != '\n' {
                i += 1;
            }
        } else if c == '"' {
            i += 1;
            let mut s = String::new();
            while i < n && b[i] != '"' {
                s.push(b[i]);
                i += 1;
            }
            if i >= n {
                return Err(ParseError("unterminated string".into()));
            }
            i += 1;
            toks.push(Tok::Str(s));
        } else if c.is_ascii_digit() || (c == '-' && i + 1 < n && b[i + 1].is_ascii_digit()) {
            let start = i;
            if c == '-' {
                i += 1;
            }
            while i < n && b[i].is_ascii_digit() {
                i += 1;
            }
            let s: String = b[start..i].iter().collect();
            toks.push(Tok::Int(
                s.parse().map_err(|_| ParseError(format!("bad integer {s}")))?,
            ));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < n && (b[i].is_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            let s: String = b[start..i].iter().collect();
            toks.push(Tok::Ident(s));
        } else {
            let two: String = b[i..(i + 2).min(n)].iter().collect();
            let punct: &'static str = match two.as_str() {
                ":-" => ":-",
                "=<" => "=<",
                ">=" => ">=",
                "\\=" => "\\=",
                _ => match c {
                    ',' => ",",
                    '(' => "(",
                    ')' => ")",
                    '.' => ".",
                    '!' => "!",
                    '<' => "<",
                    '>' => ">",
                    '=' => "=",
                    ':' => ":",
                    '+' => "+",
                    '-' => "-",
                    _ => return Err(ParseError(format!("unexpected character {c:?}"))),
                },
            };
            i += punct.len();
            toks.push(Tok::Punct(punct));
        }
    }
    Ok(toks)
}

// ------------------------------------------------------------------- parser

pub fn parse_program(src: &str) -> Result<Vec<Clause>, ParseError> {
    let toks = tokenize(src)?;
    let mut p = Parser { toks, pos: 0 };
    let mut clauses = Vec::new();
    while p.pos < p.toks.len() {
        clauses.push(p.clause()?);
    }
    Ok(clauses)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat_punct(&mut self, p: &str) -> Result<(), ParseError> {
        match self.peek() {
            Some(Tok::Punct(q)) if *q == p => {
                self.pos += 1;
                Ok(())
            }
            other => Err(ParseError(format!("expected {p:?}, found {other:?}"))),
        }
    }

    fn is_punct(&self, p: &str) -> bool {
        matches!(self.peek(), Some(Tok::Punct(q)) if *q == p)
    }

    fn clause(&mut self) -> Result<Clause, ParseError> {
        // optional rule name:  name : head :- body.
        let mut name = None;
        if let Some(Tok::Ident(_)) = self.peek() {
            if matches!(self.toks.get(self.pos + 1), Some(Tok::Punct(":"))) {
                if let Tok::Ident(s) = self.toks[self.pos].clone() {
                    name = Some(s);
                    self.pos += 2;
                }
            }
        }
        let head = self.atom()?;
        if self.is_punct(":-") {
            self.pos += 1;
            let body = self.body()?;
            self.eat_punct(".")?;
            Ok(Clause {
                name,
                head,
                body,
                is_fact: false,
            })
        } else {
            self.eat_punct(".")?;
            if name.is_some() {
                return Err(ParseError("rule name on EDB fact".into()));
            }
            Ok(Clause {
                name: None,
                head,
                body: Vec::new(),
                is_fact: true,
            })
        }
    }

    fn body(&mut self) -> Result<Vec<Lit>, ParseError> {
        let mut lits = vec![self.lit()?];
        while self.is_punct(",") {
            self.pos += 1;
            lits.push(self.lit()?);
        }
        Ok(lits)
    }

    fn lit(&mut self) -> Result<Lit, ParseError> {
        if self.is_punct("!") {
            self.pos += 1;
            let a = self.atom()?;
            return Ok(Lit::Neg(a));
        }
        if let Some(Tok::Ident(s)) = self.peek() {
            if s == "now" && matches!(self.toks.get(self.pos + 1), Some(Tok::Punct("("))) {
                self.pos += 2;
                let t = self.term()?;
                self.eat_punct(")")?;
                return Ok(Lit::Now(t));
            }
        }
        let t1 = self.term()?;
        if let Some(Tok::Punct(op)) = self.peek() {
            let cmp = match *op {
                "<" => Some(CmpOp::Lt),
                "=<" => Some(CmpOp::Le),
                ">" => Some(CmpOp::Gt),
                ">=" => Some(CmpOp::Ge),
                "=" => Some(CmpOp::Eq),
                "\\=" => Some(CmpOp::Ne),
                _ => None,
            };
            if let Some(op) = cmp {
                self.pos += 1;
                let mut e = Expr::T(self.term()?);
                while self.is_punct("+") || self.is_punct("-") {
                    let add = self.is_punct("+");
                    self.pos += 1;
                    let rhs = Expr::T(self.term()?);
                    e = if add {
                        Expr::Add(Box::new(e), Box::new(rhs))
                    } else {
                        Expr::Sub(Box::new(e), Box::new(rhs))
                    };
                }
                return Ok(Lit::Cmp(op, t1, e));
            }
        }
        // otherwise it must be a relational atom: term was an Ident (predicate)
        match t1 {
            Term::Var(_) | Term::Wildcard | Term::Int(_) | Term::Agg(..) => {
                Err(ParseError(format!("expected atom, got term {t1:?}")))
            }
            Term::Sym(pred) => {
                let args = self.args()?;
                Ok(Lit::Pos(Atom { pred, args }))
            }
        }
    }

    fn atom(&mut self) -> Result<Atom, ParseError> {
        match self.peek() {
            Some(Tok::Ident(s)) => {
                let pred = s.clone();
                self.pos += 1;
                let args = self.args()?;
                Ok(Atom { pred, args })
            }
            other => Err(ParseError(format!("expected predicate, found {other:?}"))),
        }
    }

    fn args(&mut self) -> Result<Vec<Term>, ParseError> {
        self.eat_punct("(")?;
        let mut args = Vec::new();
        if !self.is_punct(")") {
            args.push(self.term()?);
            while self.is_punct(",") {
                self.pos += 1;
                args.push(self.term()?);
            }
        }
        self.eat_punct(")")?;
        Ok(args)
    }

    fn term(&mut self) -> Result<Term, ParseError> {
        match self.peek().cloned() {
            Some(Tok::Ident(s)) => {
                self.pos += 1;
                if s == "_" {
                    Ok(Term::Wildcard)
                } else if s
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_uppercase())
                    .unwrap_or(false)
                {
                    Ok(Term::Var(s))
                }
                // aggregate terms: count(X) / min(X) / max(X) / sum(X)
                else if matches!(s.as_str(), "count" | "min" | "max" | "sum")
                    && self.is_punct("(")
                {
                    let f = match s.as_str() {
                        "count" => AggFn::Count,
                        "min" => AggFn::Min,
                        "max" => AggFn::Max,
                        _ => AggFn::Sum,
                    };
                    self.pos += 1;
                    let inner = self.term()?;
                    self.eat_punct(")")?;
                    Ok(Term::Agg(f, Box::new(inner)))
                } else {
                    Ok(Term::Sym(s))
                }
            }
            Some(Tok::Int(i)) => {
                self.pos += 1;
                Ok(Term::Int(i))
            }
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(Term::Sym(s))
            }
            other => Err(ParseError(format!("expected term, found {other:?}"))),
        }
    }
}

