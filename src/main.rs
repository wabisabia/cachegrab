mod lib;
use lib::{Cell, Comp, Thunk};

fn main() {
    let a = Cell::new(&1);
    let b = Cell::new(&2);
    let ab = (&a, &b);
    let add = Comp::new(&ab, &|(x, y)| x.value() + y.value());
    println!("{:?}", add.value());
}
