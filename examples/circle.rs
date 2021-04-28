use std::f64::consts::PI;

use dcg::{memo, Cell, Dcg, Incremental, Memo};

type Point = (f64, f64);

struct Circle {
    dcg: Dcg,
    pos: Cell<Point>,
    radius: Cell<f64>,
    circumference: Memo<f64>,
    area: Memo<f64>,
    bounding_box: Memo<(Point, f64, f64)>,
}

impl Circle {
    pub fn from_radius(radius: f64) -> Self {
        let dcg = Dcg::new();
        let radius = dcg.cell(radius);
        let pos = dcg.cell((0., 0.));
        let circumference = memo!(dcg, 2. * PI * radius, radius);
        let area = memo!(dcg, PI * radius * radius, radius);
        let bounding_box = memo!(
            dcg,
            {
                let (x, y) = pos;
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
        circle.radius.read(),
        circle.pos.read(),
        circle.circumference.read(),
        circle.area.read(),
        circle.bounding_box.read(),
    );

    circle.radius.write(2.);

    println!("{:?}", circle.dcg);

    println!(
        "{} {:?} {} {} {:?}",
        circle.radius.read(),
        circle.pos.read(),
        circle.circumference.read(),
        circle.area.read(),
        circle.bounding_box.read(),
    );

    println!("{:?}", circle.dcg);
}
