use std::f32::consts::PI;

use dcg::Dcg;
use petgraph::dot::Dot;

fn main() {
    let circle = Dcg::new();

    let radius = circle.cell(1.0);

    let get_circum = || 2.0 * PI * radius.get();
    let circumference = circle.memo(&get_circum, &[radius]);

    let get_area = || {
        let r = radius.get();
        PI * r * r
    };
    let area = circle.memo(&get_area, &[radius]);

    let canvas = Dcg::new();
    let position = canvas.cell((0.0, 0.0));

    let get_leftmost_point = || {
        let pos = position.get();
        (pos.0 - radius.query(), pos.1)
    };
    let leftmost_point = canvas.memo(&get_leftmost_point, &[position]);

    circumference.query();
    area.query();
    leftmost_point.query();

    // radius.set(2.0);

    println!(
        "{:?}",
        // Dot::new(&*circle.borrow()),
        Dot::new(&*canvas.borrow())
    );
}
