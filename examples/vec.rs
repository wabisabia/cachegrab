use cachegrab::{buffer, incremental::Incremental, Dcg};
use std::rc::Rc;

fn main() {
    let dcg = Dcg::new();
    let ones = Rc::new((0..10).map(|_| dcg.var(1)).collect::<Vec<_>>());
    let identity = buffer!(dcg, ones => {
        println!("computing...");
        ones
    });
    let incrs = buffer!(dcg, identity => {
        identity.iter().map(|x| x + 1).collect::<Vec<_>>()
    });
    println!("{:?}", incrs.read()); // "computing..."
    println!("{:?}", incrs.read()); // ""
    ones[2].write(3);
    println!("{:?}", incrs.read()); // "computing..."
    println!("{:?}", incrs.read()); // ""
}
