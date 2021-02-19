pub trait Thunk {
    type Output;

    fn value(&self) -> Self::Output;
}

pub struct Cell<'a, T: 'a>{
    value: &'a T,
}

impl<'a, T> Cell<'a, T> {
    pub const fn new(value: &'a T) -> Self {
        Self {
            value,
        }
    }
}

impl<'a, T> Thunk for Cell<'a, T> {
    type Output = &'a T;

    fn value(&self) -> Self::Output {
        self.value
    }
}

pub struct Comp<'a, I: 'a, O: Clone + 'a> {
    i: &'a I,
    f: &'a dyn Fn(&I) -> O,
    last: O,
}

impl<'a, I, O: Clone> Comp<'a, I, O> {
    pub fn new(i: &'a I, f: &'a dyn Fn(&I) -> O) -> Self {
        Self {
            i,
            f,
            last: f(i).clone(),
        }
    }
}

impl<'a, I, O: Clone> Thunk for Comp<'a, I, O> {
    type Output = O;

    fn value(&self) -> Self::Output {
        self.last.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_creates_cells() {
        let a = Cell::new(&1);
        assert_eq!(a.value(), &1);
    }

    fn adder(t: &(&Cell<u32>, &Cell<u32>)) -> u32 {
        t.0.value() + t.1.value()
    }

    #[test]
    fn it_accepts_function_pointers() {
        let a = Cell::new(&1);
        let b = Cell::new(&2);
        let ab = (&a, &b);
        let add = Comp::new(&ab, &adder);
        assert_eq!(add.value(), 3);
    }

    #[test]
    fn it_computes_constants() {
        let a = Cell::new(&1);
        let c = Comp::new(&a, &|x| x.value());
        assert_eq!(c.value(), &1);
    }

    #[test]
    fn it_computes_binary_addition() {
        let a = Cell::new(&1);
        let b = Cell::new(&2);
        let ab = (&a, &b);
        let add = Comp::new(&ab, &|(x, y)| x.value() + y.value());
        assert_eq!(add.value(), 3);
    }

    #[test]
    fn it_computes_trinary_addition() {
        let a = Cell::new(&1);
        let b = Cell::new(&2);
        let c = Cell::new(&3);
        let abc = (&a, &b, &c);
        let add = Comp::new(&abc, &|(x, y, z)| x.value() + y.value() + z.value());
        assert_eq!(add.value(), 6);
    }

    #[test]
    fn it_nests_computations() {
        let a = Cell::new(&1);
        let b = Cell::new(&2);
        let c = Cell::new(&3);
        let ab = (&a, &b);
        let add_ab = Comp::new(&ab, &|(x, y)| x.value() + y.value());
        let ab_c = (&add_ab, &c);
        let a_plus_b_div_c = Comp::new(&ab_c, &|(x, y)| x.value() / y.value());
        assert_eq!(a_plus_b_div_c.value(), 1);
    }

    #[test]
    fn it_allows_computation_reuse() {
        let a = Cell::new(&1);
        let aa = (&a, &a);
        let add_aa = Comp::new(&aa, &|(x, y)| x.value() + y.value());
        assert_eq!(add_aa.value(), 2);
    }
}
