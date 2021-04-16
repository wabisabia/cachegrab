use std::f64::consts::PI;

use dcg::{memo, Dcg, IncCell, IncMemo, Incremental};
use petgraph::dot::Dot;

type Point = (f64, f64);

struct Circle {
    dcg: Dcg,
    pos: IncCell<Point>,
    radius: IncCell<f64>,
    circumference: IncMemo<f64>,
    area: IncMemo<f64>,
    bounding_box: IncMemo<(Point, f64, f64)>,
}

impl Circle {
    pub fn from_radius(radius: f64) -> Self {
        let dcg = Dcg::new();

        let radius = dcg.cell(radius);
        let pos = dcg.cell((0., 0.));

        let radius_inc = radius.clone();
        let circumference = memo!(dcg, 2. * PI * radius_inc.read(), radius);

        let radius_inc = radius.clone();
        let area = memo!(
            dcg,
            {
                let r = radius_inc.read();
                std::f64::consts::PI * r * r
            },
            radius
        );

        let pos_inc = pos.clone();
        let radius_inc = radius.clone();
        let bounding_box = memo!(
            dcg,
            {
                let (x, y) = pos_inc.read();
                let radius = radius_inc.read();
                let half_radius = radius / 2.;
                ((x - half_radius, y - half_radius), radius, radius)
            },
            pos,
            radius
        );

        Circle {
            dcg,
            pos,
            radius,
            circumference,
            area,
            bounding_box,
        }
    }
}

fn main() {
    let circle = Circle::from_radius(1.);

    println!(
        "{} {:?} {} {} {:?}",
        circle.radius.query(),
        circle.pos.query(),
        circle.circumference.query(),
        circle.area.query(),
        circle.bounding_box.query(),
    );

    circle.radius.write(2.);

    println!("{:?}", Dot::new(&*circle.dcg.graph.borrow()));

    println!(
        "{} {:?} {} {} {:?}",
        circle.radius.query(),
        circle.pos.query(),
        circle.circumference.query(),
        circle.area.query(),
        circle.bounding_box.query(),
    );

    println!("{:?}", Dot::new(&*circle.dcg.graph.borrow()));
}
