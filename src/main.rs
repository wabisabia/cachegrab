mod thunk;
use thunk::{Thunk, Cell, Comp};

static A: Cell<u32> = Cell::new(&1);
static B: Cell<u32> = Cell::new(&2);
static AB: (&Cell<u32>, &Cell<u32>) = (&A, &B);

fn main() {
    let add = Comp::new(&AB, &|(x, y)| x.value() + y.value());
    println!("{:?}", add.value());
}
