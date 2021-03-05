mod dcg;
use dcg::Dcg;
use petgraph::dot::Dot;

fn main() {
    let dcg = Dcg::new();

    let a = dcg.cell(1);
    let b = dcg.cell(2);

    let add_ab = || dcg.get(a) + dcg.get(b);
    let thunk1 = dcg.thunk(&add_ab, &[a, b]);

    let add_one = || dcg.get(thunk1) + 1;
    let thunk2 = dcg.thunk(&add_one, &[thunk1]);

    let add_two = || dcg.get(thunk1) + 2;
    let thunk3 = dcg.thunk(&add_two, &[thunk1]);

    let two_times_thunk_plus_three = || dcg.get(thunk2) + dcg.get(thunk3);
    let _ = dcg.thunk(&two_times_thunk_plus_three, &[thunk2, thunk3]);

    dcg.set(a, 2).unwrap();

    println!("{:?}", Dot::new(&*dcg.graph.borrow()));
}
