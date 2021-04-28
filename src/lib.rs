#![warn(missing_docs)]

//! `cachegrab` implements a demanded computation graph ([`Dcg`]) used in incremental computation (IC).
//!
//! # Usage
//!
//! A [`Dcg`] can be used as a dependency-aware caching mechanism within structs:
//!
//! ```
//! use dcg::{Dcg, Incremental, Cell, Memo, memo};
//! # use std::f64::consts::PI;
//!
//! struct Circle {
//!     radius: Cell<f64>, // `Cell`s hold data
//!     area: Memo<f64>, // `Memo`s store functions and caches their results
//! }
//!
//! impl Circle {
//!     fn from_radius(radius: f64) -> Self {
//!         let dcg = Dcg::new();
//!         let radius = dcg.cell(radius);
//!         let area = memo!(dcg, {
//!             println!("Calculating area...");
//!             PI * radius * radius
//!         }, radius);
//!         Self {
//!             radius,
//!             area,
//!         }
//!     }
//! }
//! ```
//!
//! All [`Dcg`] nodes' ([`Cell`], [`Thunk`], [`Memo`]) values can be retrieved with [`read`](Incremental::read):
//!
//! ```
//! # use dcg::{Dcg, Incremental, Cell, Memo, memo};
//! # use std::f64::consts::PI;
//! #
//! # struct Circle {
//! #     radius: Cell<f64>, // `Cell`s hold data
//! #     area: Memo<f64>, // `Memo`s store functions and caches their results
//! # }
//! #
//! # impl Circle {
//! #     fn from_radius(radius: f64) -> Self {
//! #         let dcg = Dcg::new();
//! #         let radius = dcg.cell(radius);
//! #         let area = memo!(dcg, {
//! #             println!("Calculating area...");
//! #             PI * radius * radius
//! #         }, radius);
//! #         Self {
//! #             radius,
//! #             area,
//! #         }
//! #     }
//! # }
//! let circle = Circle::from_radius(1.);
//! assert_eq!(circle.radius.read(), 1.);
//! assert_eq!(circle.area.read(), PI); // "Calculating area..."
//! assert_eq!(circle.area.read(), PI); // Nothing prints: we just used a cached value!
//! ```
//!
//! Use [`write`](RawCell::write) and [`modify`](RawCell::modify) to change [`Cell`] values:
//!
//! ```
//! # use dcg::{Dcg, Incremental, Cell, Memo, memo};
//! # use std::f64::consts::PI;
//! #
//! # struct Circle {
//! #     radius: Cell<f64>, // `Cell`s hold data
//! #     area: Memo<f64>, // `Memo`s store functions and caches their results
//! # }
//! #
//! # impl Circle {
//! #     fn from_radius(radius: f64) -> Self {
//! #         let dcg = Dcg::new();
//! #         let radius = dcg.cell(radius);
//! #         let area = memo!(dcg, {
//! #             println!("Calculating area...");
//! #             PI * radius * radius
//! #         }, radius);
//! #         Self {
//! #             radius,
//! #             area,
//! #         }
//! #     }
//! # }
//! # let circle = Circle::from_radius(1.);
//! // Let's change `radius`...
//! circle.radius.write(2.);
//! assert_eq!(circle.radius.modify(|r| *r + 1.), 2.); // "change" methods yield the last [`Cell`] value
//! assert_eq!(circle.radius.read(), 3.); // New radius is indeed 2 + 1 = 3
//!
//! // DCG saw `radius` change, so `area` is recomputed and cached
//! assert_eq!(circle.area.read(), 9. * PI); // "Calculating area..."
//! ```
//!
//! [`Dcg`] nodes can be shared between computations...
//!
//! ```
//! # use dcg::{Dcg, Incremental, Cell, Memo, memo};
//! # use std::f64::consts::PI;
//! struct Circle {
//!     # radius: Cell<f64>,
//!     # area: Memo<f64>,
//!     // ...
//!     circumference: Memo<f64>,
//! }
//!
//! impl Circle {
//!     fn from_radius(radius: f64) -> Self {
//!         let dcg = Dcg::new();
//!         let radius = dcg.cell(radius);
//!         let area = memo!(dcg, PI * radius * radius, radius);
//!         let circumference = memo!(dcg, 2. * PI * radius, radius);
//!         Self {
//!             radius,
//!             area,
//!             circumference,
//!         }
//!     }
//! }
//! ```
//!
//! And can operate on heterogeneous types:
//!
//! ```
//! # use dcg::{Dcg, Incremental, Cell, Memo, memo};
//! # use std::f64::consts::PI;
//! type Point = (f64, f64);
//!
//! struct Circle {
//!     # radius: Cell<f64>,
//!     // ..
//!     pos: Cell<Point>,
//!     bounding_box: Memo<(Point, f64, f64)>,
//! }
//!
//! impl Circle {
//!     fn from_radius(radius: f64) -> Self {
//!         # let dcg = Dcg::new();
//!         # let radius = dcg.cell(radius);
//!         // ...
//!         let pos = dcg.cell((0., 0.));
//!         let bounding_box = memo!(dcg, {
//!             let (x, y) = pos;
//!             let half_radius = radius / 2.;
//!             ((x - half_radius, y - half_radius), radius, radius)
//!         }, radius, pos);
//!         Self {
//!             radius,
//!             pos,
//!             bounding_box,
//!         }
//!     }
//! }
//! ```

use petgraph::{
    dot::Dot,
    graph::{DiGraph, NodeIndex},
    visit::{depth_first_search, Control, DfsEvent},
};

use std::{cell::RefCell, fmt, rc::Rc};

type Graph = DiGraph<bool, ()>;

/// Creates- and stores dependencies between- data and compute nodes in an incremental computation.
#[derive(Default)]
pub struct Dcg {
    graph: Rc<RefCell<Graph>>,
}

/// Refines the concept of a shared [`RawCell`].
pub type Cell<T> = Rc<RawCell<T>>;

/// Refines the concept of a shared [`RawThunk`].
pub type Thunk<T> = Rc<RawThunk<T>>;

/// Refines the concept of a shared [`RawMemo`].
pub type Memo<T> = Rc<RawMemo<T>>;

impl Dcg {
    /// Creates a new, empty DCG.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::Dcg;
    ///
    /// # #[allow(unused)]
    /// let dcg = Dcg::new();
    /// ```
    pub fn new() -> Self {
        Default::default()
    }

    /// Creates a dirty [`Cell`], containing `value`.
    ///
    /// The [`Cell`] starts dirty as it has never been read.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    /// let cell = dcg.cell(1);
    ///
    /// assert!(cell.is_dirty());
    /// assert_eq!(cell.read(), 1);
    /// assert!(cell.is_clean());
    /// ```
    pub fn cell<T>(&self, value: T) -> Cell<T> {
        Rc::new(RawCell {
            value: RefCell::new(value),
            node: Node::from_dcg(self, true),
        })
    }

    /// Creates a dirty [`Thunk`] storing `f` and registers `deps` as its dependencies in the [`Dcg`].
    ///
    /// The [`Thunk`] starts dirty as it has never been read.
    ///
    /// If caching behaviour is desired, use [`Dcg::memo`] or [`memo!`] instead.
    ///
    /// # Warning ⚠
    ///
    /// It is always preferable to use [`thunk!`] instead of this method; [`thunk!`] is strictly as
    /// expressive and doesn't require you to jump through the following hoops:
    ///
    /// - Clone any dependencies.
    /// - `move` them into the closure.
    /// - List their indices in `dep`.
    ///
    /// # Examples
    ///
    /// Creating [`Thunk`]s using [`Dcg::thunk`] is _highly_ discouraged (see [`thunk!`]s documentation). That said, this is how to create
    /// [`Thunk`]s using [`Dcg::thunk`]:
    ///
    /// ```
    /// use dcg::{Dcg, Incremental, thunk};
    ///
    /// let dcg = Dcg::new();
    /// let numerator = dcg.cell(1);
    /// let denominator = dcg.cell(1);
    /// let numerator_inc = numerator.clone();
    /// let denominator_inc = denominator.clone();
    /// let safe_div = dcg.thunk(
    ///     move || {
    ///         let denominator = denominator_inc.read();
    ///         if denominator == 0 {
    ///             None
    ///         } else {
    ///             Some(numerator_inc.read() / denominator)
    ///         }
    ///     },
    ///     &[numerator.idx(), denominator.idx()],
    /// );
    ///
    /// assert_eq!(safe_div.read(), Some(1));
    /// denominator.write(0);
    /// // numerator doesn't have to be- and isn't- executed!
    /// assert_eq!(safe_div.read(), None);
    /// ```
    pub fn thunk<T, F>(&self, f: F, deps: &[NodeIndex]) -> Thunk<T>
    where
        F: Fn() -> T + 'static,
    {
        let node = Node::from_dcg(self, true);
        self.add_dependencies(&node, deps);
        Rc::new(RawThunk {
            f: Rc::new(f),
            node,
        })
    }

    /// Creates a clean [`Memo`] storing `f`, cleans its transitive dependencies and registers `deps` as its dependencies in the [`Dcg`].
    ///
    /// The [`Memo`] and its dependencies will be cleaned because `f` is called to provide an initial cache value.
    ///
    /// If non-caching behaviour is desired, use [`Dcg::thunk`] or [`thunk!`] instead.
    ///
    /// # Warning ⚠
    ///
    /// It is always preferable to use [`memo!`] instead of this method; [`memo!`] is strictly as powerful and doesn't require you to jump through the following hoops:
    ///
    /// - Clone any dependencies.
    /// - `move` them into the closure.
    /// - List their indices in `dep`.
    ///
    /// # Examples
    ///
    /// Creating [`Memo`]s using [`Dcg::memo`] is _highly_ discouraged (see [`memo!`]s documentation). That said, this is how to create
    /// [`Memo`]s using [`Dcg::memo`]:
    ///
    /// ```
    /// use dcg::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    /// let numerator = dcg.cell(1);
    /// let denominator = dcg.cell(1);
    /// let numerator_inc = numerator.clone();
    /// let denominator_inc = denominator.clone();
    /// let safe_div = dcg.memo(
    ///     move || {
    ///         let denominator = denominator_inc.read();
    ///         if denominator == 0 {
    ///             None
    ///         } else {
    ///             Some(numerator_inc.read() / denominator)
    ///         }
    ///     },
    ///     &[numerator.idx(), denominator.idx()],
    /// );
    ///
    /// assert_eq!(safe_div.read(), Some(1));
    /// denominator.write(0);
    /// // numerator doesn't have to be- and isn't- executed!
    /// assert_eq!(safe_div.read(), None);
    /// ```
    pub fn memo<T, F>(&self, f: F, deps: &[NodeIndex]) -> Memo<T>
    where
        F: Fn() -> T + 'static,
    {
        let node = Node::from_dcg(self, false);
        self.add_dependencies(&node, deps);
        let cached = RefCell::new(f());
        let memo = Rc::new(RawMemo {
            f: Rc::new(f),
            cached,
            node,
        });
        memo
    }

    fn add_dependencies(&self, node: &Node, deps: &[NodeIndex]) {
        let mut graph = self.graph.borrow_mut();
        deps.iter().for_each(|&x| {
            graph.add_edge(x, node.idx, ());
        });
    }
}

impl fmt::Debug for Dcg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", Dot::new(&*self.graph.borrow()))
    }
}

struct Node {
    graph: Rc<RefCell<Graph>>,
    idx: NodeIndex,
}

impl Node {
    fn from_dcg(dcg: &Dcg, dirty: bool) -> Self {
        Self {
            graph: dcg.graph.clone(),
            idx: dcg.graph.borrow_mut().add_node(dirty),
        }
    }

    fn is_dirty(&self) -> bool {
        self.graph.borrow_mut()[self.idx]
    }

    /// Dirties the node's transitive dependents.
    /// A DFS from the node gathers clean edges, pruning already dirty ones, and dirties them.
    fn dirty_dependents(&self) {
        let mut dependents = Vec::new();
        {
            let graph = self.graph.borrow();
            depth_first_search(&*graph, Some(self.idx), |event| {
                if let DfsEvent::Discover(n, _) = event {
                    if graph[n] {
                        return Control::Prune::<()>;
                    }
                    dependents.push(n);
                }
                Control::Continue::<()>
            });
        }

        let mut graph = self.graph.borrow_mut();
        for node in dependents {
            graph[node] = true;
        }
    }
}

/// Queryable, writable incremental data-store.
pub struct RawCell<T> {
    value: RefCell<T>,
    node: Node,
}

impl<T> RawCell<T> {
    /// Retrieves the index of the corresponding DCG node.
    ///
    /// This used to indicate a dependency when creating a [`Thunk`] or [`Memo`]
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental, thunk};
    /// use petgraph::graph::NodeIndex;
    ///
    /// let dcg = Dcg::new();
    /// let cell = dcg.cell(1);
    /// let thunk = thunk!(dcg, cell, cell);
    ///
    /// assert_eq!(cell.idx(), NodeIndex::new(0));
    /// ```
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }
}

impl<T: PartialEq> RawCell<T> {
    /// Writes a value into the cell and dirties its dependents if necessary.
    ///
    /// If `new` is equal to its current value, `new` is simply returned.
    /// Otherwise, `new` will be swapped with the current value, the cell and its dependents will be
    /// dirtied and the old value will be returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    /// let cell = dcg.cell(1);
    ///
    /// // Ensures `cell` is clean
    /// cell.read();
    ///
    /// // `cell` remains clean due to writing the same value
    /// assert_eq!(cell.write(1), 1);
    /// assert!(cell.is_clean());
    /// assert_eq!(cell.read(), 1);
    ///
    /// // `cell` dirtied due to writing a different value
    /// assert_eq!(cell.write(2), 1);
    /// assert!(cell.is_dirty());
    /// assert_eq!(cell.read(), 2);
    /// ```
    pub fn write(&self, new: T) -> T {
        if *self.value.borrow() == new {
            new
        } else {
            self.node.dirty_dependents();
            self.value.replace(new)
        }
    }

    /// Modifies the value in the cell, returning the value before modification and dirtying the node and
    /// its transitive dependents if the new value differs from the old value.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    /// let cell = dcg.cell(1);
    ///
    /// // Ensure `cell` is clean
    /// cell.read();
    ///
    /// // `cell` remains clean due to modify producing same value
    /// assert_eq!(cell.modify(|x| *x), 1);
    /// assert!(cell.is_clean());
    /// assert_eq!(cell.read(), 1);
    ///
    /// // `cell` dirtied due to modify producing different value
    /// assert_eq!(cell.modify(|x| *x + 1), 1);
    /// assert!(cell.is_dirty());
    /// assert_eq!(cell.read(), 2);
    /// ```
    pub fn modify<F>(&self, f: F) -> T
    where
        F: FnOnce(&mut T) -> T,
    {
        let old_value = self.value.replace_with(f);
        if old_value != *self.value.borrow() {
            self.node.dirty_dependents();
        }
        old_value
    }
}

/// Queryable incremental compute node.
pub struct RawThunk<T> {
    f: Rc<dyn Fn() -> T + 'static>,
    node: Node,
}

impl<T> RawThunk<T> {
    /// Retrieves the index of the corresponding DCG node.
    ///
    /// This used to indicate a dependency when creating a [`Thunk`] or [`Memo`]
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental, thunk};
    /// use petgraph::graph::NodeIndex;
    ///
    /// let dcg = Dcg::new();
    /// let thunk = thunk!(dcg, 1);
    ///
    /// assert_eq!(thunk.idx(), NodeIndex::new(0));
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }
}

/// Queryable incremental caching compute node.
pub struct RawMemo<T> {
    f: Rc<dyn Fn() -> T + 'static>,
    cached: RefCell<T>,
    node: Node,
}

impl<T> RawMemo<T> {
    /// Retrieves the index of the corresponding DCG node.
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }
}

/// Allows a type to adopt incremental "clean" and "dirty" semantics in a dependency tracking environment.
///
/// Fundamentally an [`Incremental`] type `I` produces [`Output`](Incremental::Output)s that change
/// over time due to changes in `I`'s dependencies.
///
/// `I` is either:
///
/// - **clean**: no dependencies have changed, reading `I` yields the same [`Output`](Incremental::Output) as last read.
/// - **dirty**: a dependency has changed, reading `I` computes a different [`Output`](Incremental::Output).
///
/// `I`'s state can be interrogated using [`is_dirty`](Incremental::is_dirty) and [`is_clean`](Incremental::is_clean).
///
/// [`Incremental`] provides separate methods for producing [`Output`](Incremental::Output)s:
///
/// - [`read`](Incremental::read): simply returns the most up-to-date [`Output`](Incremental::Output).
/// - [`read`](Incremental::read): does the same thing as [`read`](Incremental::read) but also cleans
/// `I`'s dependencies once the [`Output`](Incremental::Output) has been retrieved.
pub trait Incremental {
    /// The type returned when reading or reading a node.
    type Output;

    /// Cleans a node and returns its most up-to-date value.
    fn read(&self) -> Self::Output;

    /// Returns `true` if the node is dirty.
    fn is_dirty(&self) -> bool;

    /// Returns `true` if the node is clean.
    ///
    /// The default implementation requires that [`Incremental::is_dirty`] is implemented and negates the result.
    fn is_clean(&self) -> bool {
        !self.is_dirty()
    }
}

impl<T: Clone> Incremental for RawCell<T> {
    type Output = T;

    fn read(&self) -> Self::Output {
        self.node.graph.borrow_mut()[self.idx()] = false;
        self.value.borrow().clone()
    }

    fn is_dirty(&self) -> bool {
        self.node.is_dirty()
    }
}

impl<T> Incremental for RawThunk<T> {
    type Output = T;

    fn read(&self) -> Self::Output {
        self.node.graph.borrow_mut()[self.idx()] = false;
        (self.f)()
    }

    fn is_dirty(&self) -> bool {
        self.node.is_dirty()
    }
}

impl<T: Clone> Incremental for RawMemo<T> {
    type Output = T;

    fn read(&self) -> Self::Output {
        if self.is_dirty() {
            self.cached.replace((self.f)());
            self.node.graph.borrow_mut()[self.idx()] = false;
        }
        self.cached.borrow().clone()
    }

    fn is_dirty(&self) -> bool {
        self.node.is_dirty()
    }
}

/// Creates a dirty [`Thunk`].
///
/// [`thunk!`] takes two or more arguments.
///
/// The first argument is an expression that evaluates to the [`Dcg`] in which the thunk will be created.
///
/// The second argument is an expression that the [`Thunk`] will use to generate values.
/// The identifier of any [`Cell`], [`Thunk`] or [`Memo`] in scope can be used in the expression as:
///
/// - An "unwrapped" value. These are used as if the node had been [`read`](Incremental::read). These precede `;` in the remaining arguments.
/// - A "wrapped" value. These are used as if the node had not been [`read`](Incremental::read). These follow `;` in the remaining arguments.
///
/// # Warning ⚠
///
/// Dependencies referenced will always be read.
/// If more control is required, e.g. conditionally reading a node, use
/// [`thunk!`] instead.
///
/// # Examples
///
/// ```
/// use dcg::{Dcg, Incremental, thunk};
///
/// let dcg = Dcg::new();
/// let numerator = dcg.cell(1);
/// let denominator = dcg.cell(1);
/// let safe_div = thunk!(dcg, {
///     if denominator == 0 {
///         None
///     } else {
///         Some(numerator.read() / denominator)
///     }
/// }, denominator; numerator);
///
/// assert_eq!(safe_div.read(), Some(1));
/// denominator.write(0);
/// // numerator doesn't have to be- and isn't- executed!
/// assert_eq!(safe_div.read(), None);
/// ```
#[macro_export]
macro_rules! thunk {
    ($dcg:expr, $thunk:expr, $($unwrapped:ident),*; $($wrapped:ident),*) => {{
        ::paste::paste! {
            $(
                let [<$unwrapped _inc>] = $unwrapped.clone();
            )*
            $(
                let $wrapped = $wrapped.clone();
            )*
            $(
                let [<$wrapped _idx>] = $wrapped.idx();
            )*
        }
        $dcg.thunk(move || {
            ::paste::paste! {
                $(
                    let $unwrapped = [<$unwrapped _inc>].read();
                )*
            }
            $thunk
        }, &[$($unwrapped.idx()),*, ::paste::paste! { $([<$wrapped _idx>]),* }])
    }};
    ($dcg:expr, $thunk: expr) => {
        thunk!($dcg, $thunk, )
    };
    ($dcg:expr, $thunk:expr, $($unwrapped:ident),*) => {{
        ::paste::paste! {
            $(
                let [<$unwrapped _inc>] = $unwrapped.clone();
            )*
        }
        $dcg.thunk(move || {
            ::paste::paste! {
                $(
                    let $unwrapped= [<$unwrapped _inc>].read();
                )*
            }
            $thunk
        }, &[$($unwrapped.idx()),*])
    }};
    ($dcg:expr, $thunk:expr; $($wrapped:ident),*) => {{
        ::paste::paste! {
            $(
                let $wrapped = $wrapped.clone();
            )*
            $(
                let [<$wrapped _idx>] = $wrapped.idx();
            )*
        }
        $dcg.thunk(move || {
            $thunk
        }, &[::paste::paste! { $([<$wrapped _idx>]),* }])
    }}
}

/// Creates a clean [`Memo`] and cleans its transitive dependencies.
///
/// [`memo!`] takes two or more arguments.
///
/// The first argument is an expression that evaluates to the [`Dcg`] in which the memo will be created.
///
/// The second argument is an expression that the [`Memo`] will use to generate values.
/// The identifier of any [`Cell`], [`Thunk`] or [`Memo`] in scope can be used in the expression as:
///
/// - An "unwrapped" value. These are used as if the node had been [`read`](Incremental::read). These precede `;` in the remaining arguments.
/// - A "wrapped" value. These are used as if the node had not been [`read`](Incremental::read). These follow `;` in the remaining arguments.
///
/// # Warning ⚠
///
/// Dependencies referenced will always be read.
/// If more control is required, e.g. conditionally reading a node, use
/// [`Dcg::memo`] instead.
///
/// # Examples
///
/// ```
/// use dcg::{Dcg, Incremental, memo};
///
/// let dcg = Dcg::new();
/// let numerator = dcg.cell(1);
/// let denominator = dcg.cell(1);
/// let safe_div = memo!(dcg, {
///     if denominator == 0 {
///         None
///     } else {
///         Some(numerator.read() / denominator)
///     }
/// }, denominator; numerator);
///
/// assert_eq!(safe_div.read(), Some(1));
/// denominator.write(0);
/// // numerator doesn't have to be- and isn't- executed!
/// assert_eq!(safe_div.read(), None);
/// ```
#[macro_export]
macro_rules! memo {
    ($dcg:expr, $memo:expr, $($unwrapped:ident),*; $($wrapped:ident),*) => {{
        ::paste::paste! {
            $(
                let [<$unwrapped _inc>] = $unwrapped.clone();
            )*
            $(
                let $wrapped = $wrapped.clone();
            )*
            $(
                let [<$wrapped _idx>] = $wrapped.idx();
            )*
        }
        $dcg.memo(move || {
            ::paste::paste! {
                $(
                    let $unwrapped = [<$unwrapped _inc>].read();
                )*
            }
            $memo
        }, &[$($unwrapped.idx()),*, ::paste::paste! { $([<$wrapped _idx>]),* }])
    }};
    ($dcg:expr, $memo:expr) => {
        memo!($dcg, $memo, )
    };
    ($dcg:expr, $memo:expr, $($unwrapped:ident),*) => {{
        ::paste::paste! {
            $(
                let [<$unwrapped _inc>] = $unwrapped.clone();
            )*
        }
        $dcg.memo(move || {
            ::paste::paste! {
                $(
                    let $unwrapped = [<$unwrapped _inc>].read();
                )*
            }
            $memo
        }, &[$($unwrapped.idx()),*])
    }};
    ($dcg:expr, $memo:expr; $($wrapped:ident),*) => {{
        ::paste::paste! {
            $(
                let $wrapped = $wrapped.clone();
            )*
            $(
                let [<$wrapped _idx>] = $wrapped.idx();
            )*
        }
        $dcg.memo(move || {
            $memo
        }, &[::paste::paste! { $([<$wrapped _idx>]),* }])
    }};
}

#[cfg(test)]
mod tests {
    use std::cell;

    use super::*;

    #[test]
    fn create_cell() {
        let dcg = Dcg::new();

        let c = dcg.cell(1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(c.is_dirty());
    }

    #[test]
    fn create_thunk() {
        let dcg = Dcg::new();

        let t = dcg.thunk(|| (), &[]);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(t.is_dirty());
    }

    #[test]
    fn create_memo() {
        let dcg = Dcg::new();

        let m = dcg.memo(|| (), &[]);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(m.is_clean());
    }

    #[test]
    fn create_thunk_macro() {
        let dcg = Dcg::new();

        let t = thunk!(dcg, 1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(t.is_dirty());
    }

    #[test]
    fn create_memo_macro() {
        let dcg = Dcg::new();

        let m = memo!(dcg, 1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(m.is_clean());
    }

    #[test]
    fn cell_read() {
        let dcg = Dcg::new();
        let c = dcg.cell(1);

        assert_eq!(c.read(), 1);
    }

    #[test]
    fn thunk_read() {
        let dcg = Dcg::new();
        let t = thunk!(dcg, 1);

        assert_eq!(t.read(), 1);
    }

    #[test]
    fn memo_read() {
        let dcg = Dcg::new();
        let m = memo!(dcg, 1);

        assert_eq!(m.read(), 1);
    }

    #[test]
    fn cell_write() {
        let dcg = Dcg::new();
        let c = dcg.cell(1);
        let m1 = memo!(dcg, c, c);
        let m2 = memo!(dcg, m1, m1);
        let m3 = memo!(dcg, c, c);
        let m4 = memo!(dcg, m3, m3);
        dcg.graph.borrow_mut()[m3.idx()] = true;

        //   m1 --> m2
        //  /
        // c --> (m3) --> m4
        assert_eq!(c.write(2), 1);

        //   (m1) --> (m2)
        //   /
        // (c) --> (m3) --> m4
        assert!(c.is_dirty());
        assert!(m1.is_dirty());
        assert!(m2.is_dirty());
        assert!(m3.is_dirty());
        assert!(m4.is_clean());
        assert_eq!(c.read(), 2);
    }

    #[test]
    fn cell_modify() {
        let dcg = Dcg::new();
        let c = dcg.cell(1);
        let m1 = memo!(dcg, c, c);
        let m2 = memo!(dcg, m1, m1);
        let m3 = memo!(dcg, c, c);
        let m4 = memo!(dcg, m3, m3);
        dcg.graph.borrow_mut()[m3.idx()] = true;

        //   m1 --> m2
        //  /
        // c --> (m3) --> m4
        assert_eq!(c.modify(|x| *x + 1), 1);

        //   (m1) --> (m2)
        //  /
        // (c) --> (m3) --> m4
        assert!(c.is_dirty());
        assert!(m1.is_dirty());
        assert!(m2.is_dirty());
        assert!(m3.is_dirty());
        assert!(m4.is_clean());
        assert_eq!(c.read(), 2);
    }

    #[test]
    fn thunk_read_cleans() {
        let dcg = Dcg::new();
        let c1 = dcg.cell(1);
        let c2 = dcg.cell(1);
        let t1 = thunk!(dcg, c1, c1);
        let t2 = thunk!(dcg, c2, c2);
        let t3 = thunk!(dcg, t1 + t2, t1, t2);
        dcg.graph.borrow_mut()[t1.idx()] = false;

        //        (c1) --> t1
        //                   \
        // (c2) --> (t2) --> (t3)
        t3.read();

        //     c1 --> t1
        //              \
        // c2 --> t2 --> t3
        assert!(c1.is_clean());
        assert!(c2.is_clean());
        assert!(t1.is_clean());
        assert!(t2.is_clean());
        assert!(t3.is_clean());
    }

    #[test]
    fn memo_read_cleans() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let memo = memo!(dcg, cell, cell);

        cell.write(2);

        assert_eq!(memo.read(), 2);
        assert!(cell.is_clean());
        assert!(memo.is_clean());
    }

    #[test]
    fn memo_read_cleans_deep() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let memo1 = memo!(dcg, cell, cell);
        let memo2 = memo!(dcg, memo1, memo1);

        // (cell) -- (memo1) -- (memo2)

        cell.write(2);
        memo2.read();

        assert!(cell.is_clean());
        assert!(memo1.is_clean());
        assert!(memo2.is_clean());
    }

    #[test]
    fn memo_read_cleans_wide() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let memo1 = memo!(dcg, cell, cell);
        let memo2 = memo!(dcg, cell, cell);

        //      (memo1)
        //     /
        // (cell)
        //     \
        //      (memo2)

        cell.write(2);
        memo2.read();

        assert!(cell.is_clean());
        assert!(memo1.is_dirty());
        assert!(memo2.is_clean());
    }

    #[test]
    fn conditional_execution() {
        let dcg = Dcg::new();
        let numerator = dcg.cell(1);
        let denominator = dcg.cell(1);
        let num_was_read = Rc::new(cell::Cell::new(false));
        let numerator_inc = numerator.clone();
        let denominator_inc = denominator.clone();
        let num_was_read_inc = num_was_read.clone();
        let safe_div = dcg.memo(
            move || {
                let denominator = denominator_inc.read();
                if denominator == 0 {
                    None
                } else {
                    num_was_read_inc.set(true);
                    Some(numerator_inc.read() / denominator)
                }
            },
            &[numerator.idx(), denominator.idx()],
        );

        // memo created
        assert!(num_was_read.get());

        num_was_read.set(false);

        // cached value fetched
        assert_eq!(safe_div.read(), Some(1));
        assert!(!num_was_read.get());

        // affected by change
        denominator.write(2);
        assert_eq!(safe_div.read(), Some(0));
        assert!(num_was_read.get());

        num_was_read.set(false);

        // not affected by change
        denominator.write(0);
        assert_eq!(safe_div.read(), None);
        assert!(!num_was_read.get());
    }

    #[test]
    fn geometry() {
        use std::f64::consts::PI;

        let circle = Dcg::new();

        let radius = circle.cell(1.);
        let pos = circle.cell((0., 0.));

        let area = memo!(circle, PI * radius * radius, radius);

        let circum = memo!(circle, 2. * PI * radius, radius);

        let left_bound = memo!(
            circle,
            {
                let (x, y) = pos;
                (x - radius, y)
            },
            pos,
            radius
        );

        assert!(radius.is_clean());
        assert!(area.is_clean());
        assert!(circum.is_clean());
        assert!(left_bound.is_clean());

        assert_eq!(area.read(), PI);
        assert_eq!(circum.read(), 2. * PI);
        assert_eq!(left_bound.read(), (-1., 0.));

        assert!(radius.is_clean());
        assert!(area.is_clean());
        assert!(circum.is_clean());
        assert!(left_bound.is_clean());

        assert_eq!(radius.write(2.), 1.);

        assert!(radius.is_dirty());
        assert!(area.is_dirty());
        assert!(circum.is_dirty());
        assert!(left_bound.is_dirty());

        assert_eq!(area.read(), 4. * PI);
        assert_eq!(circum.read(), 4. * PI);
        assert_eq!(left_bound.read(), (-2., 0.));
    }

    #[test]
    fn multiplication_table() {
        let dcg = Dcg::new();

        let n: usize = 100;

        let nums = (1..=n).map(|i| dcg.cell(i)).collect::<Vec<_>>();

        let mut thunk_table = Vec::with_capacity(n.pow(2) / 2 + n);
        for i in 0..n {
            for j in i..n {
                let x = &nums[i];
                let y = &nums[j];
                thunk_table.push(thunk!(dcg, x * y, x, y));
            }
        }

        assert_eq!(thunk_table[0].read(), 1);
        assert_eq!(thunk_table[1].read(), 2);

        assert_eq!(nums[0].write(5), 1);

        assert_eq!(thunk_table[0].read(), 25);
        assert_eq!(thunk_table[1].read(), 10);
    }
}
