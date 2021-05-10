#![warn(missing_docs)]

//! `cachegrab` provides [`Dcg`] (Demanded Computation Graph).
//!
//! # Usage
//!
//! A [`Dcg`] can be used as a dependency-aware caching mechanism within structs:
//!
//! ```
//! use cachegrab::{Dcg, incremental::Incremental, Var, Memo, memo};
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
//! # use cachegrab::{Dcg, incremental::Incremental, Var, Memo, memo};
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
//! # use cachegrab::{Dcg, incremental::Incremental, Var, Memo, memo};
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
pub mod incremental;
use incremental::Incremental;

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
    /// use cachegrab::{Dcg, incremental::Incremental};
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
            node: Node::new(self),
        })
    }

    /// Creates a dirty [`Thunk`], adding incoming dependency edges from `params` and storing `f`.
    ///
    /// The [`Thunk`] starts dirty as it has never been read.
    ///
    /// If caching behaviour is desired, use [`Dcg::memo`] or [`memo!`] instead.
    ///
    /// # Warning ⚠
    ///
    /// It is preferable to use [`thunk!`] instead; [`thunk!`] is as expressive and doesn't require
    /// the following steps:
    ///
    /// - Clone and pass dependencies as `params`.
    /// - `move` further clones into `f`.
    ///
    /// # Usage
    ///
    /// ```
    /// use cachegrab::{Dcg, incremental::Incremental, thunk};
    ///
    /// let dcg = Dcg::new();
    /// let a = dcg.var(1);
    /// let b = dcg.var(1);
    /// let a_inc = a.clone();
    /// let b_inc = b.clone();
    /// let safe_div = dcg.thunk((a.clone(), b.clone()),
    ///     move || {
    ///         let b = b_inc.read();
    ///         if b == 0 {
    ///             None
    ///         } else {
    ///             Some(a.read() / b)
    ///         }
    ///     });
    ///
    /// assert_eq!(safe_div.read(), Some(1));
    /// b.write(0);
    /// // a doesn't have to be- and isn't- read!
    /// assert_eq!(safe_div.read(), None);
    /// ```
    pub fn thunk<P, T, F>(&self, params: P, f: F) -> Thunk<T>
    where
        P: Incremental,
        F: Fn() -> T + 'static,
    {
        Rc::new(RawThunk::new(self, params, f))
    }

    /// Creates a clean [`Memo`], adding incoming dependency edges from `params` and storing `f`.
    ///
    /// The [`Memo`] starts dirty as it has never been read.
    ///
    /// If non-caching behaviour is desired, use [`Dcg::thunk`] or [`thunk!`] instead.
    ///
    /// # Warning ⚠
    ///
    /// It is preferable to use [`memo!`] instead; [`memo!`] is as powerful and doesn't require the
    /// following steps:
    ///
    /// - Clone and pass dependencies as `params`.
    /// - `move` further clones into `f`.
    ///
    /// # Usage
    ///
    /// ```
    /// use cachegrab::{Dcg, incremental::Incremental, thunk};
    ///
    /// let dcg = Dcg::new();
    /// let a = dcg.var(1);
    /// let b = dcg.var(1);
    /// let a_inc = a.clone();
    /// let b_inc = b.clone();
    /// let safe_div = dcg.memo((a.clone(), b.clone()),
    ///     move || {
    ///         let b = b_inc.read();
    ///         if b == 0 {
    ///             None
    ///         } else {
    ///             Some(a.read() / b)
    ///         }
    ///     });
    ///
    /// assert_eq!(safe_div.read(), Some(1));
    /// b.write(0);
    /// // a doesn't have to be- and isn't- read!
    /// assert_eq!(safe_div.read(), None);
    /// ```
    pub fn memo<P, T, F>(&self, params: P, f: F) -> Memo<T>
    where
        P: Incremental,
        F: Fn() -> T + 'static,
    {
        Rc::new(RawMemo {
            thunk: RawThunk::new(self, params, f),
            cached: RefCell::new(None),
        })
    }
}

impl fmt::Debug for Dcg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", Dot::new(&*self.graph.borrow()))
    }
}

/// A handle for a node in a [`Dcg`].
pub struct Node {
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

    fn add_dependencies<P>(&self, params: P)
    where
        P: Incremental,
    {
        let mut graph = self.graph.borrow_mut();
        for node in params.nodes() {
            graph.add_edge(node.idx, self.idx, ());
        }
    }

    fn clean(&self) {
        self.graph.borrow_mut()[self.idx] = false;
    }

    fn is_dirty(&self) -> bool {
        self.graph.borrow()[self.idx]
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
    /// use cachegrab::{Dcg, incremental::Incremental};
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
    /// use cachegrab::{Dcg, incremental::Incremental};
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
    fn new<P, F>(dcg: &Dcg, params: P, f: F) -> Self
    where
        P: Incremental,
        F: Fn() -> T + 'static,
    {
        let node = Node::new(dcg);
        node.add_dependencies(params);
        Self {
            f: Box::new(f),
            node,
        }
    }
}

/// Result-caching [`RawThunk`].
pub struct RawMemo<T> {
    thunk: RawThunk<T>,
    cached: RefCell<Option<T>>,
}

impl<T: Clone> Incremental for RawVar<T> {
    type Output = T;

    fn latest(&self) -> Self::Output {
        self.value.borrow().clone()
    }

    fn is_dirty(&self) -> bool {
        self.node.is_dirty()
    }

    fn nodes(&self) -> Vec<&Node> {
        vec![&self.node]
    }
}

impl<T> Incremental for RawThunk<T> {
    type Output = T;

    fn latest(&self) -> Self::Output {
        (self.f)()
    }

    fn is_dirty(&self) -> bool {
        self.node.is_dirty()
    }

    fn nodes(&self) -> Vec<&Node> {
        vec![&self.node]
    }
}

impl<T: Clone> Incremental for RawMemo<T> {
    type Output = T;

    fn latest(&self) -> Self::Output {
        if self.is_dirty() {
            self.cached.replace(Some(self.thunk.read()));
        }
        self.cached.borrow().clone().unwrap()
    }

    fn is_dirty(&self) -> bool {
        self.thunk.is_dirty()
    }

    fn nodes(&self) -> Vec<&Node> {
        vec![&self.thunk.node]
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
/// use cachegrab::{Dcg, incremental::Incremental, thunk};
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
                let [<$read _param>] = $read.clone();
            )*
            $(
                let $unread = $unread.clone();
                let [<$unread _param>] = $unread.clone();
            )*
            $dcg.thunk(($([<$read _param>],)* $([<$unread _param>]),*), move || {
                $(
                    let $read = $crate::incremental::Incremental::read(&*[<$read _clone>]);
                )*
                $f
            })
        }
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
/// use cachegrab::{Dcg, incremental::Incremental, memo};
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
                let [<$read _param>] = $read.clone();
            )*
            $(
                let $unread = $unread.clone();
                let [<$unread _param>] = $unread.clone();
            )*
            $dcg.memo(($([<$read _param>],)* $([<$unread _param>]),*), move || {
                $(
                    let $read = $crate::incremental::Incremental::read(&*[<$read _inc>]);
                )*
                $f
            })
        }
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
        dcg.graph.borrow_mut()[m3.thunk.node.idx] = true;

        //   m1 --> m2           (m1) --> (m2)
        //  /               -->  /
        // a --> (m3) --> m4   (a) --> (m3) --> m4
        assert_eq!(a.write(2), 1);

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
        dcg.graph.borrow_mut()[m3.thunk.node.idx] = true;
        println!("{:?}", dcg);

        //   m1 --> m2           (m1) --> (m2)
        //  /               --> /
        // a --> (m3) --> m4   (a) --> (m3) --> m4
        assert_eq!(a.modify(|x| *x + 1), 1);

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
        dcg.graph.borrow_mut()[t1.node.idx] = false;

        //        (a) --> t1            a --> t1
        //                   \  -->             \
        // (b) --> (t2) --> (t3)    b --> t2 --> t3
        t3.read();

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
        dcg.graph.borrow_mut()[m1.thunk.node.idx] = false;

        //        (a) --> m1           (a) --> m1
        //                   \  -->             \
        // (b) --> (m2) --> (m3)   b --> m2 --> m3
        m3.read();

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
