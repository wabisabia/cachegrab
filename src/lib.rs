#![warn(missing_docs)]

//! `dcg` implements a demanded computation graph (DCG) used in incremental computation (IC).
//!
//! # Usage
//!
//! ```
//! use dcg::Dcg;
//! ```

use petgraph::{
    dot::Dot,
    graph::{DiGraph, NodeIndex},
    visit::{depth_first_search, Control, DfsEvent, Reversed},
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
    /// The [`Cell`] starts dirty as it has never been queried.
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
    /// assert_eq!(cell.query(), 1);
    /// assert!(cell.is_clean());
    /// ```
    pub fn cell<T>(&self, value: T) -> Cell<T> {
        Rc::new(RawCell {
            value: RefCell::new(value),
            node: Node::from_dcg(self),
        })
    }

    /// Creates a dirty [`Thunk`] storing `f` and registers `deps` as its dependencies in the
    /// [`Dcg`].
    ///
    /// The [`Thunk`] starts dirty as it has never been queried.
    ///
    /// # **NOTE ⚠**
    ///
    /// It is almost always preferable to use [`thunk!`].
    /// [`Dcg::thunk`] is error-prone as it requires the user to remember to manually clone dependencies, move
    /// them into the defined closure and read from them.
    ///
    /// If caching behaviour is desired, use [`memo!`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental, thunk};
    ///
    /// let dcg = Dcg::new();
    /// let cell = dcg.cell(1);
    /// let thunk = thunk!(dcg, cell + 1, cell);
    ///
    /// assert!(thunk.is_dirty());
    /// assert_eq!(thunk.query(), 2);
    /// assert!(thunk.is_clean());
    /// ```
    pub fn thunk<T, F>(&self, f: F, deps: &[NodeIndex]) -> Thunk<T>
    where
        F: Fn() -> T + 'static,
    {
        let node = Node::from_dcg(self);
        self.add_dependencies(&node, deps);
        Rc::new(RawThunk {
            f: Rc::new(f),
            node,
        })
    }

    /// Creates a dirty [`Memo`] storing f and registers `deps` as its dependencies in the [`Dcg`].
    ///
    /// This method queries the newly created node to generate an initial cached value, so the node
    /// and its dependencies will be cleaned.
    ///
    /// # **NOTE ⚠**
    ///
    /// It is almost always preferable to use [`memo!`].
    /// [`Dcg::memo`] is error-prone as it requires the user to remember to manually clone dependencies, move
    /// them into the defined closure and read from them.
    ///
    /// If a non-caching behaviour is desired, use [`thunk!`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental, memo};
    ///
    /// let dcg = Dcg::new();
    /// let cell = dcg.cell(1);
    /// let memo = memo!(dcg, cell + 1, cell);
    ///
    /// assert!(memo.is_dirty());
    /// assert_eq!(memo.query(), 2);
    /// assert!(memo.is_clean());
    /// assert_eq!(cell.write(2), 1);
    /// assert!(memo.is_dirty());
    /// ```
    pub fn memo<T: Clone, F>(&self, f: F, deps: &[NodeIndex]) -> Memo<T>
    where
        F: Fn() -> T + 'static,
    {
        let node = Node::from_dcg(self);
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
    fn from_dcg(dcg: &Dcg) -> Self {
        Self {
            graph: dcg.graph.clone(),
            idx: dcg.graph.borrow_mut().add_node(true),
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
        dependents.iter().for_each(|&node| {
            graph[node] = true;
        });
    }

    /// Cleans the node's transitive dependencies.
    /// A DFS over the reversed graph from the node gathers dirty edges, pruning already clean ones, and cleans them.
    fn clean_dependencies(&self) {
        let mut dependencies = Vec::new();
        {
            let graph = self.graph.borrow();
            let rev_graph = Reversed(&*graph);
            depth_first_search(rev_graph, Some(self.idx), |event| {
                if let DfsEvent::Discover(n, _) = event {
                    if !graph[n] {
                        return Control::Prune::<()>;
                    }
                    dependencies.push(n);
                }
                Control::Continue::<()>
            });
        }

        let mut graph = self.graph.borrow_mut();
        dependencies.iter().for_each(|&node| {
            graph[node] = false;
        });
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
    /// cell.query();
    ///
    /// // `cell` remains clean due to writing the same value
    /// assert_eq!(cell.write(1), 1);
    /// assert!(cell.is_clean());
    /// assert_eq!(cell.query(), 1);
    ///
    /// // `cell` dirtied due to writing a different value
    /// assert_eq!(cell.write(2), 1);
    /// assert!(cell.is_dirty());
    /// assert_eq!(cell.query(), 2);
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
    /// cell.query();
    ///
    /// // `cell` remains clean due to modify producing same value
    /// assert_eq!(cell.modify(|x| *x), 1);
    /// assert!(cell.is_clean());
    /// assert_eq!(cell.query(), 1);
    ///
    /// // `cell` dirtied due to modify producing different value
    /// assert_eq!(cell.modify(|x| *x + 1), 1);
    /// assert!(cell.is_dirty());
    /// assert_eq!(cell.query(), 2);
    /// ```
    pub fn modify<F>(&self, f: F) -> T
    where
        F: FnOnce(&mut T) -> T,
    {
        let mut value = self.value.borrow_mut();
        let new = f(&mut value);

        if *value == new {
            return new;
        }

        drop(value);

        self.node.dirty_dependents();
        self.value.replace(new)
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
/// - **clean**: no dependencies have changed, querying `I` yields the same [`Output`](Incremental::Output) as last query.
/// - **dirty**: a dependency has changed, querying `I` computes a different [`Output`](Incremental::Output).
///
/// `I`'s state can be interrogated using [`is_dirty`](Incremental::is_dirty) and [`is_clean`](Incremental::is_clean).
///
/// [`Incremental`] provides separate methods for producing [`Output`](Incremental::Output)s:
///
/// - [`read`](Incremental::read): simply returns the most up-to-date [`Output`](Incremental::Output).
/// - [`query`](Incremental::query): does the same thing as [`read`](Incremental::read) but also cleans
/// `I`'s dependencies once the [`Output`](Incremental::Output) has been retrieved.
pub trait Incremental {
    /// The type returned when reading or querying a node.
    type Output;

    /// Returns the node's most up-to-date value.
    ///
    /// This method is **only** used when trying to retrieve a value _within an incremental computation_.
    /// Otherwise, if a node's value is needed, use [`Incremental::query`] instead.
    fn read(&self) -> Self::Output;

    /// Cleans dependencies and returns the node's most up-to-date value.
    ///
    /// This method is used to interrogate incremental nodes to force their values to update.
    /// If no dependency cleaning is required, consider using [`Incremental::read`].
    fn query(&self) -> Self::Output;

    /// Returns `true` if the node is dirty.
    ///
    /// The default implementation requires that [`Incremental::is_clean`] is implemented and negates the result.
    fn is_dirty(&self) -> bool {
        !self.is_clean()
    }

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
        self.value.borrow().clone()
    }

    fn query(&self) -> Self::Output {
        // does not require DFS clean because we know cell has no dependencies
        self.node.graph.borrow_mut()[self.idx()] = false;
        self.read()
    }

    fn is_dirty(&self) -> bool {
        self.node.is_dirty()
    }
}

impl<T> Incremental for RawThunk<T> {
    type Output = T;

    fn read(&self) -> Self::Output {
        (self.f)()
    }

    fn query(&self) -> Self::Output {
        let value = self.read();
        if self.is_dirty() {
            self.node.clean_dependencies();
        }
        value
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
        }
        self.cached.borrow().clone()
    }

    fn query(&self) -> Self::Output {
        if self.is_dirty() {
            self.cached.replace((self.f)());
            self.node.clean_dependencies();
        }
        self.cached.borrow().clone()
    }

    fn is_dirty(&self) -> bool {
        self.node.is_dirty()
    }
}

/// Creates a [`Thunk`].
///
/// [`thunk!`] takes two or more arguments:
/// - The first argument is an expression that evaluates to the [`Dcg`] in which the thunk will be created.
/// - The second argument is an expression that the [`Thunk`] will use to generate values.
/// - The remaining arguments should be the identifiers of any node referenced in the expression, a.k.a. the [`Thunk`]'s dependencies.
#[macro_export]
macro_rules! thunk {
    ($dcg:expr, $thunk:expr, $($node:ident),*) => {{
        ::paste::paste! {
            $(
                let [<$node _inc>] = $node.clone();
            )*
        }
        $dcg.thunk(move || {
            ::paste::paste! {
                $(
                    let $node = [<$node _inc>].read();
                )*
            }
            $thunk
        }, &[$($node.idx()),*])
    }};
    ($dcg:expr, $thunk: expr) => {
        thunk!($dcg, $thunk, )
    };
}

/// Creates a [`Memo`].
///
/// [`memo!`] takes two or more arguments:
/// - The first argument is an expression that evaluates to the [`Dcg`] in which the memo will be created.
/// - The second argument is an expression that the [`Memo`] will use to generate values.
/// - The remaining arguments should be the identifiers of any node referenced in the expression, a.k.a. the [`Memo`]'s dependencies.
#[macro_export]
macro_rules! memo {
    ($dcg:expr, $memo:expr, $($node:ident),*) => {{
        ::paste::paste! {
            $(
                let [<$node _inc>] = $node.clone();
            )*
        }
        $dcg.memo(move || {
            ::paste::paste! {
                $(
                    let $node = [<$node _inc>].read();
                )*
            }
            $memo
        }, &[$($node.idx()),*])
    }};
    ($dcg:expr, $memo:expr) => {
        memo!($dcg, $memo, )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_cell() {
        let dcg = Dcg::new();

        let cell = dcg.cell(1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(cell.is_dirty());
    }

    #[test]
    fn create_thunk() {
        let dcg = Dcg::new();
        let thunk = dcg.thunk(|| (), &[]);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(thunk.is_dirty());
    }

    #[test]
    fn create_memo() {
        let dcg = Dcg::new();
        let memo = dcg.memo(|| (), &[]);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(memo.is_dirty());
    }

    #[test]
    fn create_thunk_macro() {
        let dcg = Dcg::new();
        let thunk = thunk!(dcg, 1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(thunk.is_dirty());
    }

    #[test]
    fn create_memo_macro() {
        let dcg = Dcg::new();
        let memo = memo!(dcg, 1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert!(memo.is_dirty());
    }

    #[test]
    fn cell_query() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);

        assert_eq!(cell.query(), 1);
    }

    #[test]
    fn thunk_query() {
        let dcg = Dcg::new();
        let thunk = thunk!(dcg, 1);

        assert_eq!(thunk.query(), 1);
    }

    #[test]
    fn memo_query() {
        let dcg = Dcg::new();
        let memo = memo!(dcg, 1);

        assert_eq!(memo.query(), 1);
    }

    #[test]
    fn cell_write() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);

        assert_eq!(cell.write(2), 1);
        assert_eq!(cell.query(), 2);
    }

    #[test]
    fn cell_write_dirties() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk = thunk!(dcg, cell, cell);
        thunk.query();

        cell.write(2);

        assert!(cell.is_dirty());
        assert!(thunk.is_dirty());
    }

    #[test]
    fn write_dirties_deep() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk1 = thunk!(dcg, cell, cell);
        let memo = memo!(dcg, thunk1, thunk1);
        thunk1.query();

        cell.write(2);

        assert!(cell.is_dirty());
        assert!(thunk1.is_dirty());
        assert!(memo.is_dirty());
    }

    #[test]
    fn write_dirties_wide() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk1 = thunk!(dcg, cell, cell);
        let thunk2 = thunk!(dcg, cell, cell);
        thunk1.query();
        thunk2.query();

        cell.write(2);

        assert!(thunk1.is_dirty());
        assert!(thunk2.is_dirty());
    }

    #[test]
    fn write_dirty_prunes() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk1 = thunk!(dcg, cell, cell);
        let thunk2 = thunk!(dcg, thunk1, thunk1);
        dcg.graph.borrow_mut()[thunk2.idx()] = false;

        // (cell) -- (thunk1) -- thunk2
        //
        // Modify cell
        // If thunk2 remains clean after dirtying, we know it was pruned
        cell.write(2);

        assert!(cell.is_dirty());
        assert!(thunk1.is_dirty());
        assert!(thunk2.is_clean());
    }

    #[test]
    fn write_dirty_doesnt_preemptively_prune() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk1 = thunk!(dcg, cell, cell);
        let thunk2 = thunk!(dcg, cell, cell);
        let thunk3 = thunk!(dcg, thunk2, thunk2);
        thunk1.query();
        dcg.graph.borrow_mut()[thunk3.idx()] = false;

        //      thunk1
        //     /
        // cell
        //     \
        //      (thunk2) -- thunk3
        //
        // Modify cell
        // DFS discovers thunk2 then thunk1
        // If thunk3 remains clean after dirtying and thunk1 was dirtied, we know thunk3 was pruned and thunk1 was still visited and dirtied
        cell.write(2);

        assert!(cell.is_dirty());
        assert!(thunk1.is_dirty());
        assert!(thunk2.is_dirty());
        assert!(thunk3.is_clean());
    }

    #[test]
    fn modify() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);

        assert_eq!(cell.modify(|x| *x + 1), 1);
        assert_eq!(cell.query(), 2);
    }

    #[test]
    fn modify_dirties() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk = thunk!(dcg, cell, cell);
        thunk.query();

        assert_eq!(cell.modify(|x| *x + 1), 1);
        assert!(cell.is_dirty());
        assert_eq!(cell.query(), 2);
    }

    #[test]
    fn thunks_nest() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk1 = thunk!(dcg, cell, cell);
        let thunk2 = thunk!(dcg, cell, cell);
        let thunk3 = thunk!(dcg, thunk1 + thunk2, thunk1, thunk2);

        {
            let graph = dcg.graph.borrow();
            assert_eq!(graph.node_count(), 4);
            assert!(graph.contains_edge(cell.idx(), thunk1.idx()));
            assert!(graph.contains_edge(cell.idx(), thunk2.idx()));
            assert!(graph.contains_edge(thunk1.idx(), thunk3.idx()));
            assert!(graph.contains_edge(thunk2.idx(), thunk3.idx()));
        }

        assert!(cell.is_dirty());
        assert!(thunk1.is_dirty());
        assert!(thunk2.is_dirty());
        assert!(thunk3.is_dirty());

        assert_eq!(thunk1.query(), 1);
        assert_eq!(thunk2.query(), 1);
        assert_eq!(thunk3.query(), 2);
    }

    #[test]
    fn memo_query_cleans() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let memo = memo!(dcg, cell, cell);

        cell.write(2);

        assert_eq!(memo.query(), 2);
        assert!(cell.is_clean());
        assert!(memo.is_clean());
    }

    #[test]
    fn thunk_query_cleans() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk = thunk!(dcg, cell, cell);

        cell.write(2);

        assert_eq!(thunk.query(), 2);
        assert!(cell.is_clean());
        assert!(thunk.is_clean());
    }

    #[test]
    fn thunk_query_cleans_deep() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk1 = thunk!(dcg, cell, cell);
        let thunk2 = thunk!(dcg, thunk1, thunk1);

        cell.write(2);
        thunk2.query();

        assert!(cell.is_clean());
        assert!(thunk1.is_clean());
        assert!(thunk2.is_clean());
    }

    #[test]
    fn thunk_query_cleans_wide() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let thunk1 = thunk!(dcg, cell, cell);
        let thunk2 = thunk!(dcg, cell, cell);

        //      (thunk1)
        //     /
        // (cell)
        //     \
        //      (thunk2)

        cell.write(2);
        thunk2.query();

        assert!(cell.is_clean());
        assert!(thunk1.is_dirty());
        assert!(thunk2.is_clean());
    }

    #[test]
    fn memo_query_cleans_deep() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let memo1 = memo!(dcg, cell, cell);
        let memo2 = memo!(dcg, memo1, memo1);

        // (cell) -- (memo1) -- (memo2)

        cell.write(2);
        memo2.query();

        assert!(cell.is_clean());
        assert!(memo1.is_clean());
        assert!(memo2.is_clean());
    }

    #[test]
    fn memo_query_cleans_wide() {
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
        memo2.query();

        assert!(cell.is_clean());
        assert!(memo1.is_dirty());
        assert!(memo2.is_clean());
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

        assert!(radius.is_dirty());
        assert!(area.is_dirty());
        assert!(circum.is_dirty());
        assert!(left_bound.is_dirty());

        assert_eq!(area.query(), PI);
        assert_eq!(circum.query(), 2. * PI);
        assert_eq!(left_bound.query(), (-1., 0.));

        assert!(radius.is_clean());
        assert!(area.is_clean());
        assert!(circum.is_clean());
        assert!(left_bound.is_clean());

        assert_eq!(radius.write(2.), 1.);

        assert!(radius.is_dirty());
        assert!(area.is_dirty());
        assert!(circum.is_dirty());
        assert!(left_bound.is_dirty());

        assert_eq!(area.query(), 4. * PI);
        assert_eq!(circum.query(), 4. * PI);
        assert_eq!(left_bound.query(), (-2., 0.));
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

        assert_eq!(thunk_table[0].query(), 1);
        assert_eq!(thunk_table[1].query(), 2);

        assert_eq!(nums[0].write(5), 1);

        assert_eq!(thunk_table[0].query(), 25);
        assert_eq!(thunk_table[1].query(), 10);
    }
}
