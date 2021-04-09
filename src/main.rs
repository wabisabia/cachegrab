use std::rc::Rc;

use dcg::{CellTy, Dcg, DcgNode, MemoTy};
use petgraph::dot::Dot;

struct Circle {
    dcg: Dcg,
    radius: Rc<DcgNode<f64, CellTy>>,
    circumference: DcgNode<f64, MemoTy>,
    area: DcgNode<f64, MemoTy>,
}

impl Circle {
    pub fn from_radius(radius: f64) -> Self {
        let dcg = Dcg::new();

        let radius = dcg.cell(radius);

        let radius1 = radius.clone();
        let calc_circum = Rc::new(move || 2.0 * std::f64::consts::PI * radius1.get());
        let circumference = dcg.memo(calc_circum, &[radius.idx]);

        let radius2 = radius.clone();
        let calc_area = Rc::new(move || {
            let r = radius2.get();
            std::f64::consts::PI * r * r
        });
        let area = dcg.memo(calc_area, &[radius.idx]);

        Circle {
            dcg,
            radius,
            circumference,
            area,
        }
    }

    pub fn radius(&self) -> f64 {
        self.radius.query()
    }

    pub fn circumference(&self) -> f64 {
        self.circumference.query()
    }

    pub fn area(&self) -> f64 {
        self.area.query()
    }
}

fn main() {
    let circle = Circle::from_radius(1.);

    println!(
        "{} {} {}",
        circle.radius(),
        circle.circumference(),
        circle.area()
    );

    circle.radius.set(2.);

    println!("{:?}", Dot::new(&*circle.dcg.graph.borrow()));

    println!(
        "{} {} {}",
        circle.radius(),
        circle.circumference(),
        circle.area()
    );

    println!("{:?}", Dot::new(&*circle.dcg.graph.borrow()));
}
