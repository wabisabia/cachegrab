use cachegrab::{thunk, Dcg, Incremental};

fn main() {
    let dcg = Dcg::new();
    let numerator = dcg.cell(1);
    let denominator = dcg.cell(1);
    let safe_div = thunk!(dcg, {
        if denominator == 0 {
            None
        } else {
            Some(numerator.read() / denominator)
        }
    }, denominator; numerator);

    println!("{:?}", safe_div.read()); // Some(1)

    denominator.write(0);

    println!("{:?}", safe_div.read()); // None
}
