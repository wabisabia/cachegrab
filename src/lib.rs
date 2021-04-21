#![warn(missing_docs)]

//! `dcg` implements a demanded computation graph (DCG) used in incremental computation (IC).

use petgraph::{
    dot::Dot,
    graph::{DiGraph, NodeIndex},
    visit::{depth_first_search, Control, DfsEvent, Reversed},
    EdgeDirection::Incoming,
};

use std::{cell, cell::RefCell, fmt, rc::Rc};

type Graph = DiGraph<(), bool>;

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

    /// Creates a dirty data node, or _cell_, initialised with the given value.
    ///
    /// The constructed data node will initially be dirty, as it has never been queried.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    ///
    /// # #[allow(unused)]
    /// let cell = dcg.cell(1);
    ///
    /// assert!(cell.is_dirty());
    /// assert_eq!(cell.query(), 1);
    /// assert!(cell.is_clean());
    /// ```
    pub fn cell<T>(&self, value: T) -> Cell<T> {
        Rc::new(RawCell {
            value: RefCell::new(value),
            dirty: cell::Cell::new(true),
            node: Node::from_dcg(self),
        })
    }

    /// Creates a dirty compute node from the given closure and registers the given dependencies with the
    /// DCG for tracking.
    ///
    /// The constructed compute node will initially be dirty, as it has never been queried.
    ///
    /// If a caching compute node is needed, use [`Dcg::memo`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let cell_inc = cell.clone();
    /// # #[allow(unused)]
    /// let thunk = dcg.thunk(move || cell_inc.read() + 1, &[cell.idx()]);
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
        self.add_dependencies(&node, deps, true);
        Rc::new(RawThunk {
            f: Rc::new(f),
            node,
        })
    }

    /// Creates a clean caching compute node from the given closure and registers the given dependencies
    /// with the DCG for tracking.
    ///
    /// This method queries the newly created node to generate an initial cached value, so the node
    /// and its dependencies will be cleaned.
    ///
    /// If a non-caching compute node is needed, use [`Dcg::thunk`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let cell_inc = cell.clone();
    /// # #[allow(unused)]
    /// let memo = dcg.memo(move || cell_inc.read() + 1, &[cell.idx()]);
    ///
    /// assert!(memo.is_clean());
    /// assert_eq!(memo.query(), 2);
    /// assert_eq!(cell.write(2), 1);
    /// assert!(memo.is_dirty());
    /// ```
    pub fn memo<T: Clone, F>(&self, f: F, deps: &[NodeIndex]) -> Memo<T>
    where
        F: Fn() -> T + 'static,
    {
        let node = Node::from_dcg(self);
        self.add_dependencies(&node, deps, false);
        let cached = f();
        let memo = Rc::new(RawMemo {
            f: Rc::new(f),
            cached,
            node,
        });
        memo.query();
        memo
    }

    fn add_dependencies(&self, node: &Node, deps: &[NodeIndex], dirty: bool) {
        let mut graph = self.graph.borrow_mut();
        deps.iter().for_each(|&x| {
            graph.add_edge(x, node.idx, dirty);
        });
    }
}

impl fmt::Debug for Dcg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", Dot::new(&*self.graph.borrow()))
    }
}

// Dirties the indexed node's transitive dependents.
// A DFS originating from the node gathers clean edges, pruning those that are already dirty, and dirties them.
fn dirty(graph: &Rc<RefCell<Graph>>, from: NodeIndex) {
    let mut edges_to_update = Vec::new();
    {
        let borrowed = graph.borrow();
        depth_first_search(&*borrowed, Some(from), |event| {
            match event {
                DfsEvent::TreeEdge(u, v) | DfsEvent::CrossForwardEdge(u, v) => {
                    let edge = borrowed.find_edge(u, v).unwrap();
                    if borrowed[edge] {
                        return Control::Prune::<()>;
                    }
                    edges_to_update.push(edge);
                }
                _ => (),
            }
            Control::Continue::<()>
        });
    }

    let mut graph = graph.borrow_mut();
    edges_to_update.iter().for_each(|&edge| {
        graph[edge] = true;
    });
}

// Cleans the indexed node's transitive dependencies.
// A DFS in the reversed graph originating from the node gathers dirty edges, pruning those that are already clean, and cleans them.
fn clean(graph: &Rc<RefCell<Graph>>, from: NodeIndex) {
    let mut edges_to_clean = Vec::new();
    {
        let graph = graph.borrow();
        let rev_graph = Reversed(&*graph);
        depth_first_search(rev_graph, Some(from), |event| {
            match event {
                DfsEvent::TreeEdge(u, v) | DfsEvent::CrossForwardEdge(u, v) => {
                    let edge = graph.find_edge(v, u).unwrap();
                    if !graph[edge] {
                        return Control::Prune::<()>;
                    }
                    edges_to_clean.push(edge);
                }
                _ => (),
            };
            Control::Continue::<()>
        });
    }

    let mut graph = graph.borrow_mut();
    edges_to_clean.iter().for_each(|&edge| {
        graph[edge] = false;
    });
}

struct Node {
    graph: Rc<RefCell<Graph>>,
    idx: NodeIndex,
}

impl Node {
    fn from_dcg(dcg: &Dcg) -> Self {
        Self {
            graph: dcg.graph.clone(),
            idx: dcg.graph.borrow_mut().add_node(()),
        }
    }
}

/// Queryable, writable incremental data-storing node.
pub struct RawCell<T> {
    value: RefCell<T>,
    dirty: cell::Cell<bool>,
    node: Node,
}

impl<T> RawCell<T>
where
    T: PartialEq + Clone,
{
    /// Retrieves the index of the corresponding DCG node.
    ///
    /// This used to indicate a dependency when creating a [`Thunk`] or [`Memo`]
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental};
    /// use petgraph::graph::NodeIndex;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let cell_inc = cell.clone();
    /// let thunk = dcg.thunk(move || cell_inc.read(), &[cell.idx()]);
    ///
    /// assert_eq!(cell.idx(), NodeIndex::new(0));
    /// ```
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }

    /// Writes a value into the cell, returning the previous stored value and dirtying the node and
    /// its transitive dependents if the new value differs from the old value.
    ///
    /// # Examples
    ///
    /// ```
    /// use dcg::{Dcg, Incremental};
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// cell.query();
    ///
    /// assert_eq!(cell.write(1), 1);
    /// assert!(cell.is_clean());
    /// assert_eq!(cell.query(), 1);
    ///
    /// assert_eq!(cell.write(2), 1);
    /// assert!(cell.is_dirty());
    /// assert_eq!(cell.query(), 2);
    /// ```
    pub fn write(&self, new: T) -> T {
        if *self.value.borrow_mut() == new {
            return new;
        }

        self.dirty.set(true);

        dirty(&self.node.graph, self.idx());

        self.value.replace(new)
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
    ///
    /// let cell = dcg.cell(1);
    ///
    /// cell.query();
    ///
    /// assert_eq!(cell.modify(|x| *x), 1);
    /// assert!(cell.is_clean());
    /// assert_eq!(cell.query(), 1);
    ///
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

        self.dirty.set(true);

        dirty(&self.node.graph, self.idx());

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
    /// use dcg::{Dcg, Incremental};
    /// use petgraph::graph::NodeIndex;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let thunk = dcg.thunk(|| 1, &[]);
    ///
    /// let thunk_inc = thunk.clone();
    /// let memo = dcg.memo(move || thunk_inc.read(), &[thunk.idx()]);
    ///
    /// assert_eq!(thunk.idx(), NodeIndex::new(0));
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }
}

/// Queryable incremental caching compute node.
pub struct RawMemo<T> {
    f: Rc<dyn Fn() -> T + 'static>,
    cached: T,
    node: Node,
}

impl<T> RawMemo<T> {
    /// Retrieves the index of the corresponding DCG node.
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }
}

/// Allows a type to adopt incremental "clean" and "dirty" semantics in a dependency tracking environment.
pub trait Incremental {
    /// The type returned when reading or querying a node.
    type Output;

    /// Returns the node's most up-to-date value.
    ///
    /// This method is **only** used when trying to retrieve a value _within an incremental computation_.
    /// Otherwise, if a node's value is needed, use [`Incremental::query`] instead.
    fn read(&self) -> Self::Output;

    /// Cleans transitive dependencies and returns the node's most up-to-date value.
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
        self.dirty.set(false);
        self.value.borrow().clone()
    }

    fn query(&self) -> Self::Output {
        self.read()
    }

    fn is_dirty(&self) -> bool {
        self.dirty.get()
    }
}

impl<T> Incremental for RawThunk<T> {
    type Output = T;

    fn read(&self) -> Self::Output {
        (self.f)()
    }

    fn query(&self) -> Self::Output {
        if self.is_dirty() {
            clean(&self.node.graph, self.idx());
        }

        (self.f)()
    }

    fn is_dirty(&self) -> bool {
        self.node
            .graph
            .borrow()
            .edges_directed(self.idx(), Incoming)
            .any(|edge| *edge.weight())
    }
}

impl<T> Incremental for RawMemo<T>
where
    T: Clone,
{
    type Output = T;

    fn read(&self) -> Self::Output {
        if self.is_clean() {
            self.cached.clone()
        } else {
            (self.f)()
        }
    }

    fn query(&self) -> Self::Output {
        if self.is_clean() {
            return self.cached.clone();
        }

        clean(&self.node.graph, self.idx());

        (self.f)()
    }

    fn is_dirty(&self) -> bool {
        self.node
            .graph
            .borrow()
            .edges_directed(self.idx(), Incoming)
            .any(|edge| *edge.weight())
    }
}

/// Convenience macro for creating [`Dcg::thunk`]s.
#[macro_export]
macro_rules! thunk {
    ($dcg:expr, $thunk:expr, $($node:expr),+) => {
        $dcg.thunk(move || $thunk, &[$($node.idx()),+])
    }
}

/// Convenience macro for creating [`Dcg::memo`]s.
#[macro_export]
macro_rules! memo {
    ($dcg:expr, $memo:expr, $($node:expr),+) => {
        $dcg.memo(move || $memo, &[$($node.idx()),+])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_cell() {
        let dcg = Dcg::new();

        let cell = dcg.cell(1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert_eq!(cell.query(), 1);
    }

    #[test]
    fn create_thunk() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk = dcg.thunk(move || cell_inc.read(), &[cell.idx()]);

        {
            let graph = dcg.graph.borrow();
            assert_eq!(graph.node_count(), 2);
            assert!(graph.contains_edge(cell.idx(), thunk.idx()));
            assert!(graph[graph.find_edge(cell.idx(), thunk.idx()).unwrap()]);
        }

        assert!(cell.is_dirty());
        assert!(thunk.is_dirty());
        assert_eq!(thunk.query(), 1);
    }

    #[test]
    fn thunk_macro() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk = thunk!(dcg, cell_inc.read(), cell);

        assert_eq!(thunk.query(), 1);
    }

    #[test]
    fn create_memo() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);

        let cell_inc = cell.clone();
        let memo = dcg.memo(move || cell_inc.read(), &[cell.idx()]);

        {
            let graph = dcg.graph.borrow();
            assert_eq!(graph.node_count(), 2);
            assert!(graph.contains_edge(cell.idx(), memo.idx()));
            assert!(!graph[graph.find_edge(cell.idx(), memo.idx()).unwrap()]);
        }

        assert!(cell.is_clean());
        assert!(memo.is_clean());
        assert_eq!(memo.query(), 1);
    }

    #[test]
    fn memo_macro() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let memo = memo!(dcg, cell_inc.read(), cell);

        assert_eq!(memo.query(), 1);
    }

    #[test]
    fn write() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);

        assert_eq!(cell.write(2), 1);
        assert_eq!(cell.query(), 2);
    }

    #[test]
    fn write_dirties() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk = thunk!(dcg, cell_inc.read(), cell);
        thunk.query();

        cell.write(2);

        assert!(cell.is_dirty());
        assert!(thunk.is_dirty());
        let graph = dcg.graph.borrow();
        assert!(graph[graph.find_edge(cell.idx(), thunk.idx()).unwrap()]);
    }

    #[test]
    fn write_dirties_deep() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk1 = thunk!(dcg, cell_inc.read(), cell);
        let thunk1_inc = thunk1.clone();
        let memo = memo!(dcg, thunk1_inc.read(), thunk1);
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
        let cell_inc = cell.clone();
        let thunk1 = thunk!(dcg, cell_inc.read(), cell);
        let cell_inc = cell.clone();
        let thunk2 = thunk!(dcg, cell_inc.read(), cell);
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
        let cell_inc = cell.clone();
        let thunk1 = thunk!(dcg, cell_inc.read(), cell);
        let thunk1_inc = thunk1.clone();
        let thunk2 = thunk!(dcg, thunk1_inc.read(), thunk1);
        // let e1 = cell -> thunk1, e2 = thunk1 -> thunk2
        // force e1 = dirty, e2 = clean
        // if e2 remains clean after dirtying, we know e1 was pruned
        let graph = dcg.graph.borrow();
        let e1 = graph.find_edge(cell.idx(), thunk1.idx()).unwrap();
        let e2 = graph.find_edge(thunk1.idx(), thunk2.idx()).unwrap();
        drop(graph);
        {
            dcg.graph.borrow_mut()[e2] = false;
        }

        cell.write(2);

        let graph = dcg.graph.borrow();
        assert!(graph[e1]);
        assert!(!graph[e2]);
    }

    #[test]
    fn write_dirty_doesnt_overprune() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk1 = thunk!(dcg, cell_inc.read(), cell);
        let cell_inc = cell.clone();
        let thunk2 = thunk!(dcg, cell_inc.read(), cell);
        let thunk2_inc = thunk2.clone();
        let thunk3 = thunk!(dcg, thunk2_inc.read(), thunk2);
        //      thunk1
        //  e1 /
        // cell
        //  e2 \\      e3
        //      thunk2 -- thunk3
        // DFS visits e2 then e1
        // if e3 remains clean after dirtying and e1 was dirtied, we know e2 was pruned and e1 was
        // still visited and dirtied
        let graph = dcg.graph.borrow();
        let e1 = graph.find_edge(cell.idx(), thunk1.idx()).unwrap();
        let e2 = graph.find_edge(cell.idx(), thunk2.idx()).unwrap();
        let e3 = graph.find_edge(thunk2.idx(), thunk3.idx()).unwrap();
        drop(graph);
        {
            let mut graph = dcg.graph.borrow_mut();
            graph[e1] = false;
            graph[e3] = false;
        }

        cell.write(2);

        let graph = dcg.graph.borrow();
        assert!(graph[e1]);
        assert!(graph[e2]);
        assert!(!graph[e3]);
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
        let cell_inc = cell.clone();
        let thunk = thunk!(dcg, cell_inc.read(), cell);
        thunk.query();

        assert_eq!(cell.modify(|x| *x + 1), 1);
        assert!(cell.is_dirty());
        let graph = dcg.graph.borrow();
        assert!(graph[graph.find_edge(cell.idx(), thunk.idx()).unwrap()]);
        assert_eq!(cell.query(), 2);
    }

    #[test]
    fn thunks_nest() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk1 = thunk!(dcg, cell_inc.read(), cell);
        let thunk1_inc = thunk1.clone();
        let cell_inc = cell.clone();
        let thunk2 = thunk!(dcg, cell_inc.read(), cell);
        let thunk2_inc = thunk2.clone();
        let thunk3 = thunk!(dcg, thunk1_inc.read() + thunk2_inc.read(), thunk1, thunk2);

        {
            let graph = dcg.graph.borrow();
            assert_eq!(graph.node_count(), 4);
            assert!(graph.contains_edge(cell.idx(), thunk1.idx()));
            assert!(graph.contains_edge(cell.idx(), thunk2.idx()));
            assert!(graph.contains_edge(thunk1.idx(), thunk3.idx()));
            assert!(graph.contains_edge(thunk2.idx(), thunk3.idx()));
            assert!(graph[graph.find_edge(cell.idx(), thunk1.idx()).unwrap()]);
            assert!(graph[graph.find_edge(cell.idx(), thunk2.idx()).unwrap()]);
            assert!(graph[graph.find_edge(thunk1.idx(), thunk3.idx()).unwrap()]);
            assert!(graph[graph.find_edge(thunk2.idx(), thunk3.idx()).unwrap()]);
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
        let cell_inc = cell.clone();
        let memo = memo!(dcg, cell_inc.read(), cell);

        cell.write(2);

        assert_eq!(memo.query(), 2);
        assert!(cell.is_clean());
        assert!(memo.is_clean());
        let graph = dcg.graph.borrow();
        assert!(!graph[graph.find_edge(cell.idx(), memo.idx()).unwrap()]);
    }

    #[test]
    fn thunk_query_cleans() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk = thunk!(dcg, cell_inc.read(), cell);

        cell.write(2);

        assert_eq!(thunk.query(), 2);
        assert!(cell.is_clean());
        assert!(thunk.is_clean());
        let graph = dcg.graph.borrow();
        assert!(!graph[graph.find_edge(cell.idx(), thunk.idx()).unwrap()]);
    }

    #[test]
    fn thunk_query_cleans_deep() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk1 = thunk!(dcg, cell_inc.read(), cell);
        let thunk1_inc = thunk1.clone();
        let thunk2 = thunk!(dcg, thunk1_inc.read(), thunk1);

        cell.write(2);
        thunk2.query();

        let graph = dcg.graph.borrow();
        assert!(!graph[graph.find_edge(thunk1.idx(), thunk2.idx()).unwrap()]);
        assert!(!graph[graph.find_edge(cell.idx(), thunk1.idx()).unwrap()]);
    }

    #[test]
    fn thunk_query_cleans_wide() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk1 = thunk!(dcg, cell_inc.read(), cell);
        let thunk1_inc = cell.clone();
        let thunk2 = thunk!(dcg, thunk1_inc.read(), cell);

        cell.write(2);
        thunk2.query();

        let graph = dcg.graph.borrow();
        assert!(graph[graph.find_edge(cell.idx(), thunk1.idx()).unwrap()]);
        assert!(!graph[graph.find_edge(cell.idx(), thunk2.idx()).unwrap()]);
    }

    #[test]
    fn memo_query_cleans_deep() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let memo1 = memo!(dcg, cell_inc.read(), cell);
        let memo1_inc = memo1.clone();
        let memo2 = memo!(dcg, memo1_inc.read(), memo1);

        cell.write(2);
        memo2.query();

        let graph = dcg.graph.borrow();
        assert!(!graph[graph.find_edge(memo1.idx(), memo2.idx()).unwrap()]);
        assert!(!graph[graph.find_edge(cell.idx(), memo1.idx()).unwrap()]);
    }

    #[test]
    fn memo_query_cleans_wide() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let memo1 = memo!(dcg, cell_inc.read(), cell);
        let cell_inc = cell.clone();
        let memo2 = memo!(dcg, cell_inc.read(), cell);

        cell.write(2);
        memo2.query();

        let graph = dcg.graph.borrow();
        assert!(graph[graph.find_edge(cell.idx(), memo1.idx()).unwrap()]);
        assert!(!graph[graph.find_edge(cell.idx(), memo2.idx()).unwrap()]);
    }

    #[test]
    fn geometry() {
        use std::f64::consts::PI;

        let circle = Dcg::new();

        let radius = circle.cell(1.);
        let pos = circle.cell((0., 0.));

        let radius_inc = radius.clone();
        let area = memo!(
            circle,
            {
                let r = radius_inc.read();
                PI * r * r
            },
            radius
        );

        let radius_inc = radius.clone();
        let circum = memo!(circle, 2. * PI * radius_inc.read(), radius);

        let pos_inc = pos.clone();
        let radius_inc = radius.clone();
        let left_bound = memo!(
            circle,
            {
                let (x, y) = pos_inc.read();
                (x - radius_inc.read(), y)
            },
            pos,
            radius
        );

        assert!(radius.is_clean());
        assert!(area.is_clean());
        assert!(circum.is_clean());
        assert!(left_bound.is_clean());

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
                let x_inc = nums[i].clone();
                let y_inc = nums[j].clone();
                thunk_table.push(thunk!(dcg, x_inc.read() * y_inc.read(), nums[i], nums[j]));
            }
        }

        assert_eq!(thunk_table[0].query(), 1);
        assert_eq!(thunk_table[1].query(), 2);

        assert_eq!(nums[0].write(5), 1);

        assert_eq!(thunk_table[0].query(), 25);
        assert_eq!(thunk_table[1].query(), 10);
    }

    // #[test]
    // fn random_graph() {
    //     #[derive(Copy, Clone)]
    //     struct Node {
    //         idx: usize,
    //         visited: bool,
    //     }

    //     impl Node {
    //         fn new(idx: usize) -> Self {
    //             Self {
    //                 idx,
    //                 visited: false,
    //             }
    //         }
    //     }

    //     impl PartialEq for Node {
    //         fn eq(&self, other: &Self) -> bool {
    //             self.idx == other.idx
    //         }
    //     }

    //     impl Eq for Node {}

    //     impl Hash for Node {
    //         fn hash<H: Hasher>(&self, state: &mut H) {
    //             self.idx.hash(state);
    //         }
    //     }

    //     let v = 100;
    //     let mut rng = SmallRng::seed_from_u64(123);
    //     let precision = 2;
    //     let scale = 10u32.pow(precision);
    //     let dist = Uniform::from(0..scale);
    //     let density = 0.5;
    //     let mut graph = HashMap::<_, HashSet<_>>::with_capacity(v);
    //     for i in 0..v {
    //         for j in 0..v {
    //             if dist.sample(&mut rng) <= (density * scale as f32) as u32 {
    //                 graph
    //                     .entry(i)
    //                     .and_modify(|neighbours| {
    //                         neighbours.insert(Node::new(j));
    //                     })
    //                     .or_default();
    //             }
    //         }
    //     }
    //     let current = 0;
    //     let mut frontier = graph[&current].iter().copied().collect::<Vec<_>>();
    //     while !frontier.is_empty() {
    //         let current = frontier.pop();
    //     }
    // }
}
