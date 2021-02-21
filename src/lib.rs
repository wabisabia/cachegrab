pub trait Thunk<T> {
    fn value(&self) -> T;
}

pub struct Cell<'a, T>
where
    T: Clone,
{
    value: &'a T,
}

impl<'a, T> Cell<'a, T>
where
    T: Clone,
{
    pub fn new(value: &'a T) -> Self {
        Self { value }
    }
}

impl<'a, T> Thunk<T> for Cell<'a, T>
where
    T: Clone,
{
    fn value(&self) -> T {
        self.value.clone()
    }
}

pub struct Comp<'a, 'b, F, I, O>
where
    F: Fn(&I) -> O,
    O: Clone,
{
    last: O,
    f: &'a F,
    i: &'b I,
}

impl<'a, 'b, F, I, O> Comp<'a, 'b, F, I, O>
where
    F: Fn(&I) -> O,
    O: Clone,
{
    pub fn new(i: &'b I, f: &'a F) -> Self {
        Self {
            last: f(i).clone(),
            f,
            i,
        }
    }
}

impl<'a, 'b, F, I, O> Thunk<O> for Comp<'a, 'b, F, I, O>
where
    F: Fn(&I) -> O,
    O: Clone,
{
    fn value(&self) -> O {
        self.last.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_creates_cells() {
        let a = Cell::new(&1);
        assert_eq!(a.value(), 1);
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
        assert_eq!(c.value(), 1);
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
