# CacheGraB (Cache-Graph Backend)

Examples can be found at `cachegrab/examples`.

## Usage

Follow the instructions [here](https://www.rust-lang.org/tools/install) to install `rustup` (which comes packaged with `cargo`).

Create a new project:

```bash
cargo new my_project
cd my_project
```

Edit `Cargo.toml`:

```toml
[dependencies]
cachegrab = { path = "../my/path/to/cachegrab" }
```

Edit `main.rs`:

```rust
use cachegrab::{buffer, incremental::Incremental, Dcg};

fn main() {
	let dcg = Dcg::new();
	let a = dcg.var(String::from("Hello "));
	let b = dcg.var("World");
	let ab = buffer!(dcg, (a, b) => a + b);
	println!("{:?}", ab.read());
}
```

## Read the Docs

Generate docs:

```bash
cd cachegrab
cargo doc
```

Read the generated docs at `cachegrab/target/doc/cachegrab/index.html`.
