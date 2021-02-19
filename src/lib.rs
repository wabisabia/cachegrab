pub trait Thunk {
    type Output;

    fn value(&self) -> Self::Output;
}

pub struct Cell<'a, T: 'a>{
    value: &'a T,
}

impl<'a, T> Cell<'a, T> {
    pub const fn new(value: &'a T) -> Self {
        Self{
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

pub struct Comp<'a, I: 'a, O: 'a> {
    i: &'a I,
    f: &'a dyn Fn(&I) -> O,
}

impl<'a, I, O> Comp<'a, I, O> {
    pub fn new(i: &'a I, f: &'a dyn Fn(&I) -> O) -> Self {
        Self {
            i,
            f,
        }
    }
}

impl<'a, I, O> Thunk for Comp<'a, I, O> {
    type Output = O;

    fn value(&self) -> Self::Output {
        (self.f)(self.i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell() {
        let a = Cell::new(&1);
        assert_eq!(a.value(), &1);
    }

    #[test]
    fn constant() {
        let a = Cell::new(&1);
        let c = Comp::new(&a, &|x| x.value());
        assert_eq!(c.value(), &1);
    }

    #[test]
    fn binary_addition() {
        let a = Cell::new(&1);
        let b = Cell::new(&2);
        let ab = (&a, &b);
        let add = Comp::new(&ab, &|(x, y)| x.value() + y.value());
        assert_eq!(add.value(), 3);
    }

    #[test]
    fn trinary_addition() {
        let a = Cell::new(&1);
        let b = Cell::new(&2);
        let c = Cell::new(&3);
        let abc = (&a, &b, &c);
        let add = Comp::new(&abc, &|(x, y, z)| x.value() + y.value() + z.value());
        assert_eq!(add.value(), 6);
    }
}
