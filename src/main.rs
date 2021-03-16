use std::f32::consts::PI;

use dcg::Dcg;
use petgraph::dot::Dot;

fn main() {
    let circle = Dcg::new();

    let radius = circle.cell(1.0);

    let calc_circum = || 2.0 * PI * radius.get();
    let circumference = circle.memo(&calc_circum, &[radius.idx]);

    let calc_area = || {
        let r = radius.get();
        PI * r * r
    };
    let area = circle.memo(&calc_area, &[radius.idx]);

    let position = circle.cell((0.0, 0.0));

    let calc_left_border = || {
        let (x, y) = position.get();
        (x - radius.get(), y)
    };
    let left_border = circle.memo(&calc_left_border, &[position.idx]);

    circumference.query();
    area.query();
    left_border.query();

    // radius.set(2.0);

    println!("{:?}", Dot::new(&*circle.graph.borrow()));
}
