use cachegrab::{buffer, incremental::Incremental, Dcg};

fn main() {
    let dcg = Dcg::default();

    let a = dcg.var(String::from("Hello "));

    let b = dcg.var("World");

    let ab = buffer!(dcg, (a, b) => a + b);

    println!("{:?}", ab.read());
}
