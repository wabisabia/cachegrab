//! Provides a struct [`Dcg`], which can be used to create and compose Dynamic
//! Computation Graphs (DCGs).
//!
//! # Concepts
//!
//! ## Nodes
//!
//! Fundamentally, a [`Dcg`] handles the instantiation of [`DcgNode`]s.
//!
//! [`DcgNode`]s are one of three types:
//!
//! - [`Cell`] - A primitive type used to wrap a value of type `T`.
//! - [`Thunk`] - A generator of `T`s, essentially a reference to a closure of
//! type `Fn() -> T`.
//! - [`Memo`] - A [`Thunk`] that caches its results.
//!
//! The dependencies of [`Thunk`] and [`Memo`] nodes **must be provided fully
//! and explicitly** upon instantiation. A [`Dcg`] stores this dependency data
//! and manages it to optimise computations.
//!
//! ## "Dirtiness"
//!
//! It may be helpful to view dirtiness through the lens of cache validity.
//!
//! A node in the DCG is "dirty" (or invalid) if the next value it
//! generates may differ from the last time it was queried. [`DcgNode`]s are
//! consequently dirty if their underlying value is changed, or if any of their
//! dependencies are dirty; dirtiness is transitive.
//!
//! Edges may be considered dirty if they originate from a dirty node.
//!
//! We can now examine the consequence of dirtiness on nodes:
//!
//! - [`Cell`] is independent, i.e. have no incoming dependency edges, so can
//! only be dirtied by a change to their inner value.
//! - [`Thunk`] always eagerly executes its closure, so while it *can* be
//! dirty, this does not affect its behaviour.
//! - [`Memo`] is most affected by dirtiness. If dirty, its cache may be
//! invalid. When queried, to safely return an up-to-date value, it must
//! generate and cache a new value. Otherwise, its cached value may be used.
//!
//! ## Generation
//!
//! Now that dirtiness has been covered, the method of generating values is
//! clear:
//!
//! - [`Cell`] yields a copy of its inner `T`.
//! - [`Thunk`] yields a copy of the result of executing its thunk.
//! - [`Memo`] yields a copy of its cache if clean; otherwise it behaves like
//! a thunk, also caching the yielded value.
//!
//! # Example Usage
//!
//! A [`Dcg`] is instantiated without referring to its generic type `T`.
//! Instead, `T` is typically inferred by the compiler via further usage and
//! hence cannot be instantiated without use:
//! ```compile_fail
//! use dcg::Dcg;
//!
//! let dcg = Dcg::new();
//!
//! // This snippet compiles successfully if the line below is uncommented
//! // dcg.cell(1);
//! ```
//! [`Cell`]s can be created with [`Dcg::cell`].
//! ```
//! # use dcg::Dcg;
//! # let dcg = Dcg::new();
//! let a = dcg.cell(1);
//! let b = dcg.cell(2);
//! ```
//! [`Thunk`]s and [`Memo`]s can also be created with [`Dcg::thunk`] and
//! [`Dcg::memo`] respectively, by passing a closure reference.
//!
//! The result of a closure may depend on other nodes. These values are
//! retrieved using [`DcgNode::get`]. A closure's dependencies must be **fully
//! specified** within a [`DcgNode`] slice:
//! ```
//! # use dcg::Dcg;
//! # let dcg = Dcg::new();
//! # let a = dcg.cell(1);
//! # let b = dcg.cell(2);
//! let add_ab = || a.get() + b.get();
//! let c = dcg.thunk(&add_ab, &[a, b]);
//! ```
//! Closure-based nodes may not depend on other nodes. This case is supported
//! through [`Dcg::lone_thunk`] and [`Dcg::lone_memo`]:
//! ```
//! # use dcg::Dcg;
//! # let dcg = Dcg::new();
//! # let a = dcg.cell(1);
//! # let b = dcg.cell(2);
//! # let add_ab = || a.get() + b.get();
//! # let c = dcg.thunk(&add_ab, &[a, b]);
//! let three = || 3;
//! let constant = dcg.lone_thunk(&three);
//! ```
//! When the contents of a node must be retrieved, it is done through
//! [`DcgNode::query`] (**NOT [`DcgNode::get`]**):
//! ```
//! # use dcg::Dcg;
//! # let dcg = Dcg::new();
//! # let a = dcg.cell(1);
//! # let b = dcg.cell(2);
//! # let add_ab = || a.get() + b.get();
//! # let c = dcg.thunk(&add_ab, &[a, b]);
//! # let three = || 3;
//! # let constant = dcg.lone_thunk(&three);
//! let add_c_constant = || c.get() + constant.get();
//! let d = dcg.thunk(&add_c_constant, &[c, constant]);
//! assert_eq!(d.query(), 6);
//! ```
use std::{cell::RefCell, fmt::Debug, marker::PhantomData, ops::Deref, ops::DerefMut};

use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::{depth_first_search, DfsEvent, EdgeRef},
};
use petgraph::{visit::Dfs, visit::Reversed, Direction};
use Direction::Incoming;

/// Internal graph node type. Stores the type and data of a [`Dcg`] graph node.
#[derive(Clone)]
pub enum Node<'a, T>
where
    T: Clone + PartialEq + Debug,
{
    /// Contains a value of type [`Result<T ,T>`].
    ///
    /// The underlying value may be retrieved or replaced by calling
    /// [`DcgNode::get`] or [`DcgNode::set`] on the corresponding
    /// [`DcgNode<Cell>`].
    Cell(Result<T, T>),

    /// Contains a thunk which produces a value of type `T`.
    ///
    /// The result of the thunk may be retrieved by calling [`DcgNode::get`] on
    /// the corresponding [`DcgNode<Thunk>`].
    Thunk(&'a dyn Fn() -> T),

    /// Contains a thunk which produces a value of type `T` and a cached value
    /// of type `T` which holds the result of the previous evaluation of the
    /// thunk.
    ///
    /// When [`DcgNode::get`] is called on the corresponding [`DcgNode<Memo>`] the
    /// value returned depends on the cleanliness of the node's dependencies.
    /// - If any dependency is dirty, the [`DcgNode<Memo>`] is dirty and thunk
    /// is re-evaluated, cached and returned.
    /// - Otherwise, the cached value is returned.
    Memo(&'a dyn Fn() -> T, T),
}

/// [`DcgNode`] marker denoting a [`Dcg::cell`].
pub struct Cell {}

/// [`DcgNode`] marker denoting a [`Dcg::thunk`] or [`Dcg::lone_thunk`].
pub struct Thunk {}

/// [`DcgNode`] marker denoting a [`Dcg::memo`] or [`Dcg::lone_memo`].
pub struct Memo {}

/// Tracks the graph a node belongs to, its index and its type.
///
/// The `Ty` marker provides compile-time information about a [`DcgNode`]'s
/// type. Valid types are:
///
/// - [`Cell`]
/// - [`Thunk`]
/// - [`Memo`]
///
/// Conversion to the underlying [`NodeIndex`] is provided via a
/// [`From<DcgNode>`] implementation.
pub struct DcgNode<'a, T, Ty>
where
    T: Clone + PartialEq + Debug,
{
    graph: &'a GraphRepr<'a, T>,
    idx: NodeIndex,
    phantom: PhantomData<Ty>,
}

impl<T, Ty> DcgNode<'_, T, Ty>
where
    T: Clone + PartialEq + Debug,
{
    fn inner_node(&self) -> Node<'_, T> {
        self.graph.borrow()[self.idx].clone()
    }

    pub fn is_dirty(&self) -> bool {
        let dcg = self.graph.borrow();
        match &dcg[self.idx] {
            Node::Cell(result) => result.is_err(),
            _ => dcg
                .edges_directed(self.idx, Direction::Incoming)
                .any(|edge| *edge.weight()),
        }
    }

    pub fn is_clean(&self) -> bool {
        !self.is_dirty()
    }

    pub fn get(&self) -> T {
        let inner_node = self.inner_node();
        if self.is_clean() {
            match inner_node {
                Node::Cell(result) => result.unwrap(),
                Node::Thunk(thunk) => thunk(),
                Node::Memo(_, value) => value,
            }
        } else {
            let value = match inner_node {
                Node::Cell(result) => result.unwrap_err(),
                Node::Thunk(thunk) => thunk(),
                Node::Memo(thunk, _) => thunk(),
            };
            match &mut self.graph.borrow_mut()[self.idx] {
                Node::Cell(result) => *result = Ok(value.clone()),
                Node::Thunk(_) => (),
                Node::Memo(_, cached) => *cached = value.clone(),
            };
            value
        }
    }

    pub fn query(&self) -> T {
        let mut dirty_edges = Vec::new();
        {
            let dcg = self.graph.borrow();
            let rev_dcg = Reversed(&*dcg);
            let mut dfs = Dfs::new(rev_dcg, self.idx);
            while let Some(node) = dfs.next(rev_dcg) {
                dirty_edges.extend(dcg.edges_directed(node, Incoming).filter_map(|edge| {
                    if *edge.weight() {
                        Some(edge.id())
                    } else {
                        None
                    }
                }));
            }
        }
        let value = self.get();
        dirty_edges.iter().for_each(|edge| {
            self.graph.borrow_mut()[*edge] = false;
        });
        value
    }
}

impl<T> DcgNode<'_, T, Cell>
where
    T: Clone + PartialEq + Debug,
{
    /// Sets the [`DcgNode`}'s value to `new_value`, "dirtying" all dependent
    /// nodes.
    ///
    /// This method dirties all nodes that are transitively dependent on
    /// `node` and returns the previous cell value.
    ///
    /// This function only accepts [`DcgNode<Cell>`]s, i.e. [`Node`]s
    /// generated by [`Dcg::cell`]:
    ///
    /// ```compile_fail
    /// use dcg::Dcg;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let x = || 42;
    /// let thunk = dcg.lone_thunk(&x);
    ///
    /// // Compile error! set can only be called on cells
    /// thunk.set(1);
    /// ```
    ///
    /// The dirtying phase performs a Depth-First-Search from `node` and sets
    /// the weight of each tree/cross edge encountered to [`true`]
    /// # Examples
    /// ```
    /// use dcg::Dcg;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let get_cell = || cell.get();
    /// let thunk1 = dcg.thunk(&get_cell, &[cell]);
    /// let thunk2 = dcg.thunk(&get_cell, &[cell]);
    /// let get_thunk1 = || thunk1.get();
    /// let thunk3 = dcg.thunk(&get_thunk1, &[thunk1]);
    ///
    /// /* BEFORE: no dirty edges
    /// *
    /// *     thunk1 -- thunk3
    /// *    /
    /// *   a
    /// *    \
    /// *     thunk2
    /// */
    ///
    /// assert_eq!(cell.set(42), 1);
    ///
    /// /* AFTER: all edges dirtied
    /// *
    /// *      thunk1 == thunk3
    /// *    //
    /// *   a
    /// *    \\
    /// *      thunk2
    /// */
    ///
    /// assert_eq!(cell.get(), 42);
    ///
    /// assert!(dcg.borrow().raw_edges().iter().all(|edge| edge.weight));
    /// ```
    pub fn set(&self, new_value: T) -> T {
        let value = match &mut self.graph.borrow_mut()[self.idx] {
            Node::Cell(ref mut result) => {
                let value = match result {
                    Ok(value) => {
                        if *value == new_value {
                            return new_value;
                        }
                        value
                    }
                    Err(value) => value,
                };
                let tmp = value.clone();
                *result = Err(new_value);
                tmp
            }
            _ => unreachable!(),
        };

        let mut transitive_edges = Vec::new();
        {
            let dcg = self.graph.borrow();
            depth_first_search(&*dcg, Some(self.idx), |event| {
                let uv = match event {
                    DfsEvent::TreeEdge(u, v) => Some((u, v)),
                    DfsEvent::CrossForwardEdge(u, v) => Some((u, v)),
                    _ => None,
                };
                match uv {
                    Some((u, v)) => transitive_edges.push(dcg.find_edge(u, v).unwrap()),
                    None => (),
                }
            });
        }

        let mut dcg = self.graph.borrow_mut();
        transitive_edges.iter().for_each(|&edge| {
            dcg[edge] = true;
        });
        value
    }
}

/// Workaround for a [bug](https://github.com/rust-lang/rust/issues/26925) in
/// [`PhantomData`]
impl<T, Ty> Clone for DcgNode<'_, T, Ty>
where
    T: Clone + PartialEq + Debug,
{
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

/// Workaround for a [bug](https://github.com/rust-lang/rust/issues/26925) in
/// [`PhantomData`]
impl<T, Ty> Copy for DcgNode<'_, T, Ty> where T: Clone + PartialEq + Debug {}

impl<T, Ty> From<DcgNode<'_, T, Ty>> for NodeIndex
where
    T: Clone + PartialEq + Debug,
{
    fn from(node: DcgNode<'_, T, Ty>) -> Self {
        node.idx
    }
}

impl<T> Debug for Node<'_, T>
where
    T: Clone + PartialEq + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Node::Cell(value) => write!(f, "{:?}", value),
            Node::Thunk(_) => f.debug_tuple("Thunk").finish(),
            Node::Memo(_, last_value) => f.debug_tuple("Memo").field(&last_value).finish(),
        }
    }
}

type GraphRepr<'a, T> = RefCell<DiGraph<Node<'a, T>, bool>>;

/// The central data structure responsible for vending [`DcgNode`]s and
/// maintaining dependency information.
///
/// [`Dcg`]s are generic over their stored type `T` and can only contain nodes
/// that "generate" values of type `T`.
///
/// This struct is a thin wrapper around a [`DiGraph`] with nodes of type
/// [`Node`] and edges of type [`bool`].
///
pub struct Dcg<'a, T>(pub GraphRepr<'a, T>)
where
    T: Clone + PartialEq + Debug;

impl<'a, T> Deref for Dcg<'a, T>
where
    T: Clone + PartialEq + Debug,
{
    type Target = GraphRepr<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> DerefMut for Dcg<'a, T>
where
    T: Clone + PartialEq + Debug,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T> Dcg<'a, T>
where
    T: Clone + PartialEq + Debug,
{
    /// Creates an empty DCG.
    /// # Examples
    /// ```
    /// use dcg::Dcg;
    ///
    /// let dcg: Dcg<i64> = Dcg::new();
    ///
    /// assert_eq!(dcg.borrow().node_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self(RefCell::new(DiGraph::new()))
    }

    fn add_dependencies<N, D>(&self, node: DcgNode<T, N>, dependencies: &[DcgNode<T, D>]) {
        let idx = node.into();
        let dep_states: Vec<_>;
        {
            dep_states = dependencies
                .iter()
                .map(|&dep| (dep.into(), dep.is_dirty()))
                .collect();
        }
        let mut dcg = self.borrow_mut();
        dep_states.iter().for_each(|(dep, dirty)| {
            dcg.add_edge(*dep, idx, *dirty);
        });
    }

    /// Creates and adds a [`Node::Cell`] to the dependency graph, returning
    /// a corresponding [`DcgNode<Cell>`].
    /// # Examples
    /// ```
    /// use dcg::Dcg;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// assert_eq!(dcg.borrow().node_count(), 1);
    /// assert_eq!(cell.get(), 1);
    /// ```
    pub fn cell(&'a self, value: T) -> DcgNode<'a, T, Cell> {
        DcgNode {
            graph: &self.0,
            idx: self.borrow_mut().add_node(Node::Cell(Ok(value))),
            phantom: PhantomData,
        }
    }

    /// Creates and adds a [`Node::Thunk`] and its [`DcgNode<Ty>`] dependencies
    /// to the dependency graph, returning a corresponding [`DcgNode<Thunk>`].
    ///
    /// To be used where the thunk is dependent on at least one DCG node. If
    /// this is not the case, instead use [`Dcg::lone_thunk`].
    ///
    /// # Examples
    /// ```
    /// use dcg::Dcg;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let get_cell = || cell.get();
    /// let thunk = dcg.thunk(&get_cell, &[cell]);
    ///
    /// let graph = dcg.borrow();
    ///
    /// assert_eq!(graph.node_count(), 2);
    ///
    /// assert!(graph.contains_edge(cell.into(), thunk.into()));
    ///
    /// assert_eq!(thunk.get(), cell.get());
    /// ```
    pub fn thunk<F, Ty>(
        &'a self,
        thunk: &'a F,
        dependencies: &[DcgNode<T, Ty>],
    ) -> DcgNode<'a, T, Thunk>
    where
        F: Fn() -> T,
    {
        let node = self.lone_thunk(thunk);
        self.add_dependencies(node, dependencies);
        node
    }

    /// Creates and adds a memo'd thunk and its dependencies to the dependency
    /// graph, returning a corresponding [`DcgNode<Memo>`].
    ///
    /// To be used where the memo is dependent on at least one DCG node. If
    /// this is not the case, instead use [`Dcg::lone_memo`].
    ///
    /// # Examples
    /// ```
    /// use dcg::{Dcg, Node};
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let get_cell = || cell.get();
    /// let memo = dcg.memo(&get_cell, &[cell]);
    ///
    /// let graph = dcg.borrow();
    /// let memo_idx = memo.into();
    ///
    /// assert_eq!(graph.node_count(), 2);
    ///
    /// assert!(graph.contains_edge(cell.into(), memo_idx));
    ///
    /// assert_eq!(memo.get(), cell.get());
    ///
    /// match graph[memo_idx] {
    ///     Node::Memo(_, value) => assert_eq!(value, 1),
    ///     _ => (),
    /// };
    /// ```
    pub fn memo<F, Ty>(&'a self, thunk: &'a F, dependencies: &[DcgNode<T, Ty>]) -> DcgNode<T, Memo>
    where
        F: Fn() -> T,
    {
        let node = self.lone_memo(thunk);
        self.add_dependencies(node, dependencies);
        node
    }

    /// Creates and adds a thunk with no dependencies to the dependency graph,
    /// returning a corresponding [`DcgNode<Thunk>`].
    ///
    /// To be used where the thunk is not dependent on any DCG nodes. If this
    /// is not the case, instead use [`Dcg::thunk`].
    /// # Examples
    /// ```
    /// use dcg::Dcg;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let meaning_of_life = || 42;
    /// let thunk = dcg.lone_thunk(&meaning_of_life);
    ///
    /// assert_eq!(dcg.borrow().node_count(), 1);
    ///
    /// assert_eq!(thunk.get(), 42);
    /// ```
    pub fn lone_thunk<F>(&'a self, thunk: &'a F) -> DcgNode<'a, T, Thunk>
    where
        F: Fn() -> T,
    {
        DcgNode {
            graph: &self.0,
            idx: self.borrow_mut().add_node(Node::Thunk(thunk)),
            phantom: PhantomData,
        }
    }

    /// Creates and adds a memo'd thunk with no dependencies to the dependency
    /// graph, returning a corresponding [`DcgNode<Memo>`].
    ///
    /// To be used where the thunk is not dependent on any DCG nodes. If this
    /// is not the case, instead use [`Dcg::memo`].
    /// # Examples
    /// ```
    /// use dcg::{Dcg, Node};
    /// use petgraph::graph::NodeIndex;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let meaning_of_life = || 42;
    /// let memo = dcg.lone_memo(&meaning_of_life);
    ///
    /// assert_eq!(dcg.borrow().node_count(), 1);
    ///
    /// assert_eq!(memo.query(), 42);
    ///
    /// let memo_idx: NodeIndex = memo.into();
    ///
    /// match dcg.borrow()[memo_idx] {
    ///     Node::Memo(_, value) => assert_eq!(value, 42),
    ///     _ => (),
    /// };
    /// ```
    pub fn lone_memo<F>(&'a self, thunk: &'a F) -> DcgNode<'a, T, Memo>
    where
        F: Fn() -> T,
    {
        let value = thunk();
        DcgNode {
            graph: &self.0,
            idx: self.borrow_mut().add_node(Node::Memo(thunk, value)),
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_cell() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        assert_eq!(dcg.borrow().node_count(), 1);

        assert_eq!(a.query(), 1);
    }

    #[test]
    fn create_thunk() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        let get_a = || a.get();
        let thunk = dcg.thunk(&get_a, &[a]);

        {
            let graph = dcg.borrow();

            assert_eq!(graph.node_count(), 2);

            assert!(graph.contains_edge(a.into(), thunk.into()));

            assert!(!graph.raw_edges().iter().any(|edge| edge.weight));
        }

        assert_eq!(thunk.query(), 1);
    }

    #[test]
    fn create_memo() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        let get_a = || a.get();
        let memo = dcg.memo(&get_a, &[a]);

        {
            let graph = dcg.borrow();

            assert_eq!(graph.node_count(), 2);

            assert!(graph.contains_edge(a.into(), memo.into()));

            assert!(!graph.raw_edges().iter().any(|edge| edge.weight));
        }

        assert_eq!(memo.query(), 1);
    }

    #[test]
    fn create_lone_thunk() {
        let dcg = Dcg::new();

        let thunk = dcg.lone_thunk(&|| 42);

        assert_eq!(dcg.borrow().node_count(), 1);

        assert_eq!(thunk.query(), 42);
    }

    #[test]
    fn create_lone_memo() {
        let dcg = Dcg::new();

        let memo = dcg.lone_memo(&|| 42);

        assert_eq!(dcg.borrow().node_count(), 1);

        assert_eq!(memo.query(), 42);
    }

    #[test]
    fn thunk_nested() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        let get_a = || a.get();

        let thunk1 = dcg.thunk(&get_a, &[a]);
        let thunk2 = dcg.thunk(&get_a, &[a]);

        let add = || thunk1.get() + thunk2.get();
        let thunk3 = dcg.thunk(&add, &[thunk1, thunk2]);

        {
            let graph = dcg.borrow();

            assert_eq!(graph.node_count(), 4);

            assert!(graph.contains_edge(a.into(), thunk1.into()));
            assert!(graph.contains_edge(a.into(), thunk2.into()));
            assert!(graph.contains_edge(thunk1.into(), thunk3.into()));
            assert!(graph.contains_edge(thunk2.into(), thunk3.into()));

            assert!(!graph.raw_edges().iter().any(|edge| edge.weight));
        }

        assert_eq!(thunk1.query(), 1);
        assert_eq!(thunk2.query(), 1);
        assert_eq!(thunk3.query(), 2);
    }

    #[test]
    fn dirtying_phase() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        let get_a = || a.get();
        let thunk = dcg.thunk(&get_a, &[a]);

        assert_eq!(thunk.query(), 1);

        assert_eq!(a.set(2), 1);

        let graph = dcg.borrow();

        assert!(graph[graph.find_edge(a.into(), thunk.into()).unwrap()]);
    }

    #[test]
    fn cleaning_phase() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        let get_a = || a.get();
        let thunk = dcg.thunk(&get_a, &[a]);

        a.set(2);

        assert_eq!(thunk.query(), 2);

        let graph = dcg.borrow();

        assert!(!graph[graph.find_edge(a.into(), thunk.into()).unwrap()]);
    }

    #[test]
    fn cleaning_phase_two_layers() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        let get_a = || a.get();
        let b = dcg.thunk(&get_a, &[a]);

        let get_b = || b.get();
        let c = dcg.thunk(&get_b, &[b]);

        a.set(2);

        c.query();

        let graph = dcg.borrow();

        assert!(!graph[graph.find_edge(b.into(), c.into()).unwrap()]);

        assert!(!graph[graph.find_edge(a.into(), b.into()).unwrap()]);
    }

    #[test]
    fn cleaning_phase_split() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        let get_a = || a.get();

        let b = dcg.thunk(&get_a, &[a]);
        let c = dcg.thunk(&get_a, &[a]);

        a.set(2);

        c.query();

        let graph = dcg.borrow();

        assert!(graph[graph.find_edge(a.into(), b.into()).unwrap()]);

        assert!(!graph[graph.find_edge(a.into(), c.into()).unwrap()]);
    }
}
