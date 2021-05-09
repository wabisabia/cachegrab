use cachegrab::{incremental::Incremental, thunk, Dcg};

fn main() {
    let dcg = Dcg::new();
    let numerator = dcg.var(1);
    let denominator = dcg.var(1);
    let safe_div = thunk!(dcg, (denominator; numerator) => {
        if denominator == 0 {
            None
        } else {
            Some(numerator.read() / denominator)
        }
    });

    println!("{:?}", safe_div.read()); // Some(1)

    denominator.write(0);

    println!("{:?}", safe_div.read()); // None
}
