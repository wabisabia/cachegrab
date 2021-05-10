use std::f64::consts::PI;

use cachegrab::{buffer, incremental::Incremental, Buffer, Dcg, Var};

type Point = (f64, f64);

struct Circle {
    dcg: Dcg,
    pos: Var<Point>,
    radius: Var<f64>,
    circumference: Buffer<f64>, // cheap computation
    area: Buffer<f64>,
    bounding_box: Buffer<(Point, f64, f64)>, // expensive computation
}

impl Circle {
    pub fn from_radius(radius: f64) -> Self {
        let dcg = Dcg::new();
        let radius = dcg.var(radius);
        let pos = dcg.var((0., 0.));
        let circumference = buffer!(dcg, radius => 2. * PI * radius);
        let area = buffer!(dcg, radius => PI * radius * radius);
        let bounding_box = buffer!(dcg, (pos, radius) => {
            let (x, y) = pos;
            let half_radius = radius / 2.;
            ((x - half_radius, y - half_radius), radius, radius)
        });

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
