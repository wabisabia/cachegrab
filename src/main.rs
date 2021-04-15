use dcg::{Dcg, IncCell, IncMemo, Incremental};
use petgraph::dot::Dot;

struct Circle {
    dcg: Dcg,
    pos: IncCell<(f64, f64)>,
    radius: IncCell<f64>,
    circumference: IncMemo<f64>,
    area: IncMemo<f64>,
}

impl Circle {
    pub fn from_radius(radius: f64) -> Self {
        let dcg = Dcg::new();

        let radius = dcg.cell(radius);
        let pos = dcg.cell((0., 0.));

        let radius_inc = radius.clone();
        let circumference = dcg.memo(
            move || 2. * std::f64::consts::PI * radius_inc.read(),
            &[radius.idx()],
        );

        let radius_inc = radius.clone();
        let area = dcg.memo(
            move || {
                let r = radius_inc.read();
                std::f64::consts::PI * r * r
            },
            &[radius.idx()],
        );

        Circle {
            dcg,
            pos,
            radius,
            circumference,
            area,
        }
    }

    pub fn radius(&self) -> f64 {
        self.radius.query()
    }

    pub fn pos(&self) -> (f64, f64) {
        self.pos.query()
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
        "{} {:?} {} {}",
        circle.radius(),
        circle.pos(),
        circle.circumference(),
        circle.area()
    );

    circle.radius.write(2.);

    println!("{:?}", Dot::new(&*circle.dcg.graph.borrow()));

    println!(
        "{} {:?} {} {}",
        circle.radius(),
        circle.pos(),
        circle.circumference(),
        circle.area()
    );

    println!("{:?}", Dot::new(&*circle.dcg.graph.borrow()));
}
