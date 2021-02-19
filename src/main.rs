// Allows implementors to generate sized values
trait Thunk {
    type Output;

    fn value(&self) -> Self::Output;
}

struct Cell<T: 'static>{
    value: &'static T,
}

impl<T> Cell<T> {
    const fn new(value: &'static T) -> Self {
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

struct Comp<I: 'static, O: 'static> {
    i: &'static I,
    f: &'static dyn Fn(&I) -> O,
}

impl<I, O> Comp<I, O> {
    fn new(i: &'static I, f: &'static dyn Fn(&I) -> O) -> Self {
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
static C: (&Cell<u32>, &Cell<u32>) = (&A, &B);

fn main() {
    let add = Comp::new(&C, &|(x, y): &(&Cell<u32>, &Cell<u32>)| x.value() + y.value());
    println!("{:?}", add.value());
}
