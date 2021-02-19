pub trait Thunk {
    type Output;

    fn value(&self) -> Self::Output;
}

pub struct Cell<T: 'static>{
    value: &'static T,
}

impl<T> Cell<T> {
    pub const fn new(value: &'static T) -> Self {
        Self{
            value,
        }
    }
}

impl<T> Thunk for Cell<T> {
    type Output = &'static T;

    fn value(&self) -> Self::Output {
        self.value
    }
}

pub struct Comp<I: 'static, O: 'static> {
    i: &'static I,
    f: &'static dyn Fn(&I) -> O,
}

impl<I, O> Comp<I, O> {
    pub fn new(i: &'static I, f: &'static dyn Fn(&I) -> O) -> Self {
        Self {
            i,
            f,
        }
    }
}

impl<I, O> Thunk for Comp<I, O> {
    type Output = O;

    fn value(&self) -> Self::Output {
        (self.f)(self.i)
    }
}

static A: Cell<u32> = Cell::new(&1);
static B: Cell<u32> = Cell::new(&2);
static AB: (&Cell<u32>, &Cell<u32>) = (&A, &B);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_addition() {
        let add = Comp::new(&AB, &|(x, y)| x.value() + y.value());
        assert_eq!(add.value(), 3);
    }
}
