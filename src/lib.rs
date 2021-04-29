#![warn(missing_docs)]

//! `cachegrab` implements a demanded computation graph ([`Dcg`]) used in incremental computation (IC).
//!
//! # Usage
//!
//! A [`Dcg`] can be used as a dependency-aware caching mechanism within structs:
//!
//! ```
//! use cachegrab::{Dcg, Incremental, Var, Memo, memo};
//! # use std::f64::consts::PI;
//!
//! struct Circle {
//!     radius: Var<f64>, // `Var`s hold data
//!     area: Memo<f64>, // `Memo`s store functions and cache their results
//! }
//!
//! impl Circle {
//!     fn from_radius(radius: f64) -> Self {
//!         let dcg = Dcg::new();
//!         let radius = dcg.var(radius);
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
//! All [`Dcg`] nodes' ([`Var`], [`Thunk`], [`Memo`]) values can be retrieved with [`read`](Incremental::read):
//!
//! ```
//! # use cachegrab::{Dcg, Incremental, Var, Memo, memo};
//! # use std::f64::consts::PI;
//! #
//! # struct Circle {
//! #     radius: Var<f64>, // `Var`s hold data
//! #     area: Memo<f64>, // `Memo`s store functions and caches their results
//! # }
//! #
//! # impl Circle {
//! #     fn from_radius(radius: f64) -> Self {
//! #         let dcg = Dcg::new();
//! #         let radius = dcg.var(radius);
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
//! Use [`write`](RawVar::write) and [`modify`](RawVar::modify) to change [`Var`] values:
//!
//! ```
//! # use cachegrab::{Dcg, Incremental, Var, Memo, memo};
//! # use std::f64::consts::PI;
//! #
//! # struct Circle {
//! #     radius: Var<f64>, // `Var`s hold data
//! #     area: Memo<f64>, // `Memo`s store functions and caches their results
//! # }
//! #
//! # impl Circle {
//! #     fn from_radius(radius: f64) -> Self {
//! #         let dcg = Dcg::new();
//! #         let radius = dcg.var(radius);
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
//! assert_eq!(circle.radius.modify(|r| *r + 1.), 2.); // "change" methods yield the last [`Var`] value
//! assert_eq!(circle.radius.read(), 3.); // New radius is indeed 2 + 1 = 3
//!
//! // DCG saw `radius` change, so `area` is recomputed and cached
//! assert_eq!(circle.area.read(), 9. * PI); // "Calculating area..."
//! ```
//!
//! [`Dcg`] nodes can be shared between computations...
//!
//! ```
//! # use cachegrab::{Dcg, Incremental, Var, Memo, memo};
//! # use std::f64::consts::PI;
//! struct Circle {
//!     # radius: Var<f64>,
//!     # area: Memo<f64>,
//!     // ...
//!     circumference: Memo<f64>,
//! }
//!
//! impl Circle {
//!     fn from_radius(radius: f64) -> Self {
//!         let dcg = Dcg::new();
//!         let radius = dcg.var(radius);
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
//! # use cachegrab::{Dcg, Incremental, Var, Memo, memo};
//! # use std::f64::consts::PI;
//! type Point = (f64, f64);
//!
//! struct Circle {
//!     # radius: Var<f64>,
//!     // ..
//!     pos: Var<Point>,
//!     bounding_box: Memo<(Point, f64, f64)>,
//! }
//!
//! impl Circle {
//!     fn from_radius(radius: f64) -> Self {
//!         # let dcg = Dcg::new();
//!         # let radius = dcg.var(radius);
//!         // ...
//!         let pos = dcg.var((0., 0.));
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

/// Refines the concept of a shared [`RawVar`].
pub type Var<T> = Rc<RawVar<T>>;

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
    /// use cachegrab::Dcg;
    ///
    /// # #[allow(unused)]
    /// let dcg = Dcg::new();
    /// ```
    pub fn new() -> Self {
        Default::default()
    }

    /// Creates a dirty [`Var`], containing `value`.
    ///
    /// The [`Var`] starts dirty as it has never been read.
    ///
    /// # Examples
    ///
    /// ```
    /// use cachegrab::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    /// let a = dcg.var(1);
    ///
    /// assert!(a.is_dirty());
    /// assert_eq!(a.read(), 1);
    /// assert!(a.is_clean());
    /// ```
    pub fn var<T>(&self, value: T) -> Var<T> {
        Rc::new(RawVar {
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
    /// use cachegrab::{Dcg, Incremental, thunk};
    ///
    /// let dcg = Dcg::new();
    /// let numerator = dcg.var(1);
    /// let denominator = dcg.var(1);
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
    /// use cachegrab::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    /// let numerator = dcg.var(1);
    /// let denominator = dcg.var(1);
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
        for &dep in deps {
            graph.add_edge(dep, node.idx, ());
        }
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
pub struct RawVar<T> {
    value: RefCell<T>,
    node: Node,
}

impl<T> RawVar<T> {
    /// Retrieves the index of the corresponding DCG node.
    ///
    /// This used to indicate a dependency when creating a [`Thunk`] or [`Memo`]
    ///
    /// # Examples
    ///
    /// ```
    /// use cachegrab::{Dcg, Incremental, thunk};
    /// use petgraph::graph::NodeIndex;
    ///
    /// let dcg = Dcg::new();
    /// let a = dcg.var(1);
    /// let thunk = thunk!(dcg, a, a);
    ///
    /// assert_eq!(a.idx(), NodeIndex::new(0));
    /// ```
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }
}

impl<T: PartialEq> RawVar<T> {
    /// Writes a value into the [`Var`] and dirties its dependents if necessary.
    ///
    /// If `new` is equal to its current value, `new` is simply returned.
    /// Otherwise, `new` will be swapped with the current value, the [`Var`] and its dependents will be
    /// dirtied and the old value will be returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use cachegrab::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    /// let a = dcg.var(1);
    ///
    /// // Ensures `a` is clean
    /// a.read();
    ///
    /// // `a` remains clean due to writing the same value
    /// assert_eq!(a.write(1), 1);
    /// assert!(a.is_clean());
    /// assert_eq!(a.read(), 1);
    ///
    /// // `a` dirtied due to writing a different value
    /// assert_eq!(a.write(2), 1);
    /// assert!(a.is_dirty());
    /// assert_eq!(a.read(), 2);
    /// ```
    pub fn write(&self, new: T) -> T {
        if *self.value.borrow() == new {
            new
        } else {
            self.node.dirty_dependents();
            self.value.replace(new)
        }
    }

    /// Modifies the value in the [`Var`], returning the value before modification and dirtying the node and
    /// its transitive dependents if the new value differs from the old value.
    ///
    /// # Examples
    ///
    /// ```
    /// use cachegrab::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    /// let a = dcg.var(1);
    ///
    /// // Ensure `a` is clean
    /// a.read();
    ///
    /// // `a` remains clean due to modify producing same value
    /// assert_eq!(a.modify(|x| *x), 1);
    /// assert!(a.is_clean());
    /// assert_eq!(a.read(), 1);
    ///
    /// // `a` dirtied due to modify producing different value
    /// assert_eq!(a.modify(|x| *x + 1), 1);
    /// assert!(a.is_dirty());
    /// assert_eq!(a.read(), 2);
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
    /// use cachegrab::{Dcg, Incremental, thunk};
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

impl<T: Clone> Incremental for RawVar<T> {
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
/// The identifier of any [`Var`], [`Thunk`] or [`Memo`] in scope can be used in the expression as:
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
/// use cachegrab::{Dcg, Incremental, thunk};
///
/// let dcg = Dcg::new();
/// let numerator = dcg.var(1);
/// let denominator = dcg.var(1);
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
/// The identifier of any [`Var`], [`Thunk`] or [`Memo`] in scope can be used in the expression as:
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
/// use cachegrab::{Dcg, Incremental, memo};
///
/// let dcg = Dcg::new();
/// let numerator = dcg.var(1);
/// let denominator = dcg.var(1);
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

    use cell::Cell;

    use super::*;

    #[test]
    fn create_var() {
        let dcg = Dcg::new();

        let a = dcg.var(1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(a.is_dirty());
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
    fn var_read() {
        let dcg = Dcg::new();
        let a = dcg.var(1);

        assert_eq!(a.read(), 1);
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
    fn var_write() {
        let dcg = Dcg::new();
        let a = dcg.var(1);
        let m1 = memo!(dcg, a, a);
        let m2 = memo!(dcg, m1, m1);
        let m3 = memo!(dcg, a, a);
        let m4 = memo!(dcg, m3, m3);
        dcg.graph.borrow_mut()[m3.idx()] = true;

        //   m1 --> m2
        //  /
        // a --> (m3) --> m4
        assert_eq!(a.write(2), 1);

        //   (m1) --> (m2)
        //   /
        // (a) --> (m3) --> m4
        assert!(a.is_dirty());
        assert!(m1.is_dirty());
        assert!(m2.is_dirty());
        assert!(m3.is_dirty());
        assert!(m4.is_clean());
        assert_eq!(a.read(), 2);
    }

    #[test]
    fn var_modify() {
        let dcg = Dcg::new();
        let a = dcg.var(1);
        let m1 = memo!(dcg, a, a);
        let m2 = memo!(dcg, m1, m1);
        let m3 = memo!(dcg, a, a);
        let m4 = memo!(dcg, m3, m3);
        dcg.graph.borrow_mut()[m3.idx()] = true;

        //   m1 --> m2
        //  /
        // a --> (m3) --> m4
        assert_eq!(a.modify(|x| *x + 1), 1);

        //   (m1) --> (m2)
        //  /
        // (a) --> (m3) --> m4
        assert!(a.is_dirty());
        assert!(m1.is_dirty());
        assert!(m2.is_dirty());
        assert!(m3.is_dirty());
        assert!(m4.is_clean());
        assert_eq!(a.read(), 2);
    }

    #[test]
    fn thunk_read_cleans() {
        let dcg = Dcg::new();
        let a = dcg.var(1);
        let b = dcg.var(1);
        let t1 = thunk!(dcg, a, a);
        let t2 = thunk!(dcg, b, b);
        let t3 = thunk!(dcg, t1 + t2, t1, t2);
        dcg.graph.borrow_mut()[t1.idx()] = false;

        //        (a) --> t1
        //                   \
        // (b) --> (t2) --> (t3)
        t3.read();

        //     a --> t1
        //              \
        // b --> t2 --> t3
        assert!(a.is_clean());
        assert!(b.is_clean());
        assert!(t1.is_clean());
        assert!(t2.is_clean());
        assert!(t3.is_clean());
    }

    #[test]
    fn memo_read_cleans() {
        let dcg = Dcg::new();
        let a = dcg.var(1);
        let b = dcg.var(1);
        let m1 = memo!(dcg, a, a);
        let m2 = memo!(dcg, b, b);
        let m3 = memo!(dcg, m1 + m2, m1, m2);
        a.write(2);
        b.write(2);
        dcg.graph.borrow_mut()[m1.idx()] = false;

        //        (a) --> m1
        //                   \
        // (b) --> (m2) --> (m3)
        m3.read();

        //     (a) --> m1
        //              \
        // b --> m2 --> m3
        assert!(a.is_dirty());
        assert!(b.is_clean());
        assert!(m1.is_clean());
        assert!(m2.is_clean());
        assert!(m3.is_clean());
    }

    #[test]
    fn conditional_execution() {
        let dcg = Dcg::new();
        let a = dcg.var(1);
        let b = dcg.var(1);
        let a_read = Rc::new(Cell::new(false));
        let a_read_clone = a_read.clone();
        let safe_div = memo!(dcg, {
            if b == 0 {
                None
            } else {
                a_read_clone.set(true);
                Some(a.read() / b)
            }
        }, b; a);

        // memo created
        assert!(a_read.get());

        a_read.set(false);

        // cached value fetched
        assert_eq!(safe_div.read(), Some(1));
        assert!(!a_read.get());

        // affected by change
        b.write(2);
        assert_eq!(safe_div.read(), Some(0));
        assert!(a_read.get());

        a_read.set(false);

        // not affected by change
        b.write(0);
        assert_eq!(safe_div.read(), None);
        assert!(!a_read.get());
    }
}
