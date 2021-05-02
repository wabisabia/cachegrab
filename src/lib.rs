#![warn(missing_docs)]

//! `cachegrab` provides [`Dcg`] (Demanded Computation Graph).
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
//!         // `Memo`s cache most recent value
//!         let area = memo!(dcg, radius => {
//!             println!("Calculating area...");
//!             PI * radius * radius
//!         });
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
//! #         let area = memo!(dcg, (radius) => {
//! #             println!("Calculating area...");
//! #             PI * radius * radius
//! #         });
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
//! #         let area = memo!(dcg, radius => {
//! #             println!("Calculating area...");
//! #             PI * radius * radius
//! #         });
//! #         Self {
//! #             radius,
//! #             area,
//! #         }
//! #     }
//! # }
//! # let circle = Circle::from_radius(1.);
//! // Let's change `radius`...
//! circle.radius.write(2.);
//! assert_eq!(circle.radius.modify(|r| *r + 1.), 2.);  // "Change" methods return the last value
//! assert_eq!(circle.radius.read(), 3.);               // New radius is indeed 3
//!
//! // We changed `radius`, so `area` is re-computed and cached
//! assert_eq!(circle.area.read(), 9. * PI);            // "Calculating area..."
//! ```
//!
//! [`Dcg`] nodes can be shared between computations...
//!
//! ```
//! # use cachegrab::{Dcg, memo};
//! # use std::f64::consts::PI;
//! # let dcg = Dcg::new();
//! # let radius = dcg.var(1.);
//! let area = memo!(dcg, radius => PI * radius * radius);        // radius used here
//! let circumference = memo!(dcg, radius => 2. * PI * radius);   // ... and here
//! ```

use petgraph::{
    dot::Dot,
    graph::{DiGraph, NodeIndex},
    visit::{depth_first_search, Control, DfsEvent},
};

#[doc(hidden)]
pub use paste::paste;
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
        Rc::new(RawVar::new(self, value))
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
        Rc::new(RawThunk::new(self, f, deps))
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
        Rc::new(RawMemo::new(self, f, deps))
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
    fn new(dcg: &Dcg) -> Self {
        Self {
            graph: dcg.graph.clone(),
            idx: dcg.graph.borrow_mut().add_node(true),
        }
    }

    fn add_dependencies(&self, deps: &[NodeIndex]) {
        let mut graph = self.graph.borrow_mut();
        for &dep in deps {
            graph.add_edge(dep, self.idx, ());
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

/// Data-storing [`Dcg`] node.
pub struct RawVar<T> {
    value: RefCell<T>,
    node: Node,
}

impl<T> RawVar<T> {
    fn new(dcg: &Dcg, value: T) -> Self {
        Self {
            value: RefCell::new(value),
            node: Node::new(dcg),
        }
    }

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
    /// let thunk = thunk!(dcg, a);
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

/// Naively re-computing [`Dcg`] node.
pub struct RawThunk<T> {
    f: Box<dyn Fn() -> T + 'static>,
    node: Node,
}

impl<T> RawThunk<T> {
    fn new<F>(dcg: &Dcg, f: F, deps: &[NodeIndex]) -> Self
    where
        F: Fn() -> T + 'static,
    {
        let node = Node::new(dcg);
        node.add_dependencies(deps);
        Self {
            f: Box::new(f),
            node,
        }
    }

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

/// Result-caching [`RawThunk`].
pub struct RawMemo<T> {
    thunk: RawThunk<T>,
    cached: RefCell<Option<T>>,
}

impl<T> RawMemo<T> {
    fn new<F>(dcg: &Dcg, f: F, deps: &[NodeIndex]) -> Self
    where
        F: Fn() -> T + 'static,
    {
        Self {
            thunk: RawThunk::new(dcg, f, deps),
            cached: RefCell::new(None),
        }
    }

    /// Retrieves the index of the corresponding DCG node.
    pub fn idx(&self) -> NodeIndex {
        self.thunk.idx()
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
            self.cached.replace(Some(self.thunk.read()));
        }
        self.cached.borrow().clone().unwrap()
    }

    fn is_dirty(&self) -> bool {
        self.thunk.is_dirty()
    }
}

/// Creates a dirty [`Thunk`].
///
/// The first argument is the [`Dcg`] in which the [`Thunk`] will be created.
///
/// The second argument specifies how the [`Thunk`] will generate values.
/// It can be:
///
/// - An [`Incremental`]'s `ident`.
/// - Of the form `params => expr` where
///     - `params` is either
///         - `(reads; unreads)` where `read` and `unread` are `,`-separated lists of [`Incremental`] `ident`s.
///         - `read` - an [`Incremental`]'s `ident`.
///         - `(reads)` where `reads` is a `,`-separated list of [`Incremental`] `ident`s.
///     - `expr` is an expression that treats:
///         - `read` params as if they were (`read`)[Incremental::read].
///         - `unread` params as normal.
/// - An `expr` (ideally not referencing an [`Incremental`]).
///
/// ```
/// use cachegrab::{Dcg, Incremental, thunk};
///
/// let dcg = Dcg::new();
/// let numerator = dcg.var(1);
/// let denominator = dcg.var(1);
/// let safe_div = thunk!(dcg, (denominator; numerator) => {
///     if denominator == 0 {
///         None
///     } else {
///         Some(numerator.read() / denominator)
///     }
/// });
///
/// assert_eq!(safe_div.read(), Some(1));
/// denominator.write(0);
/// // numerator doesn't have to be- and isn't- executed!
/// assert_eq!(safe_div.read(), None);
/// ```
#[macro_export]
macro_rules! thunk {
    ($dcg:expr, $read:ident) => {
        thunk!($dcg, ($read) => $read)
    };
    ($dcg:expr, ($($read:ident),*; $($unread:ident),*) => $f:expr) => {{
        $crate::paste! {
            $(
                let [<$read _clone>] = $read.clone();
            )*
            $(
                let $unread = $unread.clone();
            )*
            $(
                let [<$unread _idx>] = $unread.idx();
            )*
        }
        $dcg.thunk(move || {
            $crate::paste! {
                $(
                    let $read = $crate::Incremental::read(&*[<$read _clone>]);
                )*
            }
            $f
        }, &[$($read.idx(),)* $crate::paste! { $([<$unread _idx>]),* }])
    }};
    ($dcg:expr, $read:ident => $f:expr) => {
        thunk!($dcg, ($read) => $f)
    };
    ($dcg:expr, ($($read:ident),*) => $f:expr) => {
        thunk!($dcg, ($($read),*;) => $f)
    };
    ($dcg:expr, $f:expr) => {
        thunk!($dcg, (;) => $f)
    };
}

/// Creates a clean [`Memo`] and cleans its transitive dependencies.
///
/// The first argument is the [`Dcg`] in which the [`Memo`] will be created.
///
/// The second argument specifies how the [`Memo`] will generate values.
/// It can be:
///
/// - An [`Incremental`]'s `ident`; the [`Incremental`] is simply (`read`)[Incremental::read].
/// - Of the form `params => expr` where
///     - `params` can be
///         - `(reads; unreads)` where `read` and `unread` are `,`-separated lists of [`Incremental`] `ident`s.
///         - `read` - an [`Incremental`]'s `ident`.
///         - `(reads)` where `reads` is a `,`-separated list of [`Incremental`] `ident`s.
///     - `expr` is an expression that treats:
///         - `read` params as if they were (`read`)[Incremental::read].
///         - `unread` params as normal.
/// - An `expr` (ideally not referencing an [`Incremental`]).
///
/// # Examples
///
/// ```
/// use cachegrab::{Dcg, Incremental, memo};
///
/// let dcg = Dcg::new();
/// let numerator = dcg.var(1);
/// let denominator = dcg.var(1);
/// let safe_div = memo!(dcg, (denominator; numerator) => {
///     if denominator == 0 {
///         None
///     } else {
///         Some(numerator.read() / denominator)
///     }
/// });
///
/// assert_eq!(safe_div.read(), Some(1));
/// denominator.write(0);
/// // numerator doesn't have to be- and isn't- executed!
/// assert_eq!(safe_div.read(), None);
/// ```
#[macro_export]
macro_rules! memo {
    ($dcg:expr, $read:ident) => {
        memo!($dcg, ($read) => $read)
    };
    ($dcg:expr, ($($read:ident),*; $($unread:ident),*) => $f:expr) => {{
        $crate::paste! {
            $(
                let [<$read _inc>] = $read.clone();
            )*
            $(
                let $unread = $unread.clone();
            )*
            $(
                let [<$unread _idx>] = $unread.idx();
            )*
        }
        $dcg.memo(move || {
            $crate::paste! {
                $(
                    let $read = $crate::Incremental::read(&*[<$read _inc>]);
                )*
            }
            $f
        }, &[$($read.idx(),)* $crate::paste! { $([<$unread _idx>]),* }])
    }};
    ($dcg:expr, $read:ident => $f:expr) => {
        memo!($dcg, ($read) => $f)
    };
    ($dcg:expr, ($($read:ident),*) => $f:expr) => {
        memo!($dcg, ($($read),*;) => $f)
    };
    ($dcg:expr, $f:expr) => {
        memo!($dcg, (;) => $f)
    };
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

        let t = thunk!(dcg, 1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(t.is_dirty());
    }

    #[test]
    fn create_memo() {
        let dcg = Dcg::new();

        let m = memo!(dcg, 1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(m.is_dirty());
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
        let m1 = memo!(dcg, a);
        let m2 = memo!(dcg, m1);
        let m3 = memo!(dcg, a);
        let m4 = memo!(dcg, m3);
        m2.read();
        m4.read();
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
        let m1 = memo!(dcg, a);
        let m2 = memo!(dcg, m1);
        let m3 = memo!(dcg, a);
        let m4 = memo!(dcg, m3);
        m2.read();
        m4.read();
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
        let t1 = thunk!(dcg, a);
        let t2 = thunk!(dcg, b);
        let t3 = thunk!(dcg, (t1, t2) => t1 + t2);
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
        let m1 = memo!(dcg, a);
        let m2 = memo!(dcg, b);
        let m3 = memo!(dcg, (m1, m2) => m1 + m2);
        // we ensure m1 contains Some(value) to avoid unwrapping a None
        m1.read();
        a.write(2);
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
        let safe_div = memo!(dcg, (b; a) => {
            if b == 0 {
                None
            } else {
                a_read_clone.set(true);
                Some(a.read() / b)
            }
        });

        // lazy memo created
        assert!(!a_read.get());

        a_read.set(false);

        // computes and caches value
        assert_eq!(safe_div.read(), Some(1));
        assert!(a_read.get());

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
