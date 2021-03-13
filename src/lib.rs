//! Provides tools for for building and editing Dynamic
//! Computation Graphs (DCGs).
//!
//! # Concepts
//!
//! ## DCG
//!
//! Fundamentally, a [`Dcg`] handles the instantiation of [`DcgNode`]s and
//! manages interdependencies.
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
//! ## "Dirty" nodes and edges
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
//! A [`Cell`]'s value can be changed with [`DcgNode::set`], yielding
//! the cell's previous value:
//! ```
//! # use dcg::Dcg;
//! # let dcg = Dcg::new();
//! # let a = dcg.cell(1);
//! # let b = dcg.cell(2);
//! # let add_ab = || a.get() + b.get();
//! # let c = dcg.thunk(&add_ab, &[a, b]);
//! # let three = || 3;
//! # let constant = dcg.lone_thunk(&three);
//! # let add_c_constant = || c.get() + constant.get();
//! # let d = dcg.thunk(&add_c_constant, &[c, constant]);
//! # assert_eq!(d.query(), 6);
//! assert_eq!(a.set(3), 1);
//! ```
use std::{cell::RefCell, fmt::Debug, marker::PhantomData, ops::Deref, ops::DerefMut};

use petgraph::{graph::DiGraph, graph::NodeIndex, visit::EdgeRef, EdgeDirection::Incoming};
use petgraph::{visit::Dfs, visit::Reversed};

/// Internal graph node type. Stores the type and data of a [`Dcg`] graph node.
#[derive(Clone)]
pub enum DcgData<'a, T>
where
    T: PartialEq + Clone + Debug,
{
    /// Contains a value of type [`Result<T ,T>`].
    ///
    /// The underlying value may be retrieved or replaced by calling
    /// [`DcgNode::get`] or [`DcgNode::set`] on the corresponding
    /// [`DcgNode<Cell>`].
    Cell(T, bool),

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
    T: PartialEq + Clone + Debug,
{
    graph: &'a GraphRepr<'a, T>,
    idx: NodeIndex,
    phantom: PhantomData<Ty>,
}

impl<T, Ty> DcgNode<'_, T, Ty>
where
    T: PartialEq + Clone + Debug,
{
    fn inner_node(&self) -> DcgData<'_, T> {
        self.graph.borrow()[self.idx].clone()
    }

    /// Returns [`true`] if the node is dirty and [`false`] otherwise.
    ///
    /// [`DcgNode<Cell>`] is dirty if the value contained is a [`Result::Err`].
    /// [`DcgNode<Thunk>`] and [`DcgNode<Memo>`] are dirty if at least one
    /// dirty edge terminates at the node.
    pub fn is_dirty(&self) -> bool {
        let dcg = self.graph.borrow();
        match &dcg[self.idx] {
            DcgData::Cell(_, dirty) => *dirty,
            _ => dcg
                .edges_directed(self.idx, Incoming)
                .any(|edge| *edge.weight()),
        }
    }

    /// Returns [`true`] if the node is clean and [`false`] otherwise.
    ///
    /// [`DcgNode<Cell>`] is clean if the value contained is a [`Result::Ok`].
    /// [`DcgNode<Thunk>`] and [`DcgNode<Memo>`] are clean if no dirty edges
    /// terminate at node.
    pub fn is_clean(&self) -> bool {
        !self.is_dirty()
    }

    /// Generates the node's value and cleans the node.
    ///
    /// This function should only be used within closures passed to
    /// [`Dcg::thunk`] or [`Dcg::memo`] or their `Dcg::lone_*` counterparts.
    ///
    /// If the node is a cell, this will return the value stored in its
    /// [`Result`] and ensure that the final inner result is wrapped in a
    /// [`Result::Ok`].
    ///
    /// If the node is a thunk, the closure stored will always be executed and
    /// returned.
    ///
    /// If the node is a memo'd thunk, the cached value is returned if the
    /// node is clean, otherwise the closure is executed and the result is
    /// cached and returned.
    pub fn get(&self) -> T {
        let inner_node = self.inner_node();
        if self.is_clean() {
            match inner_node {
                DcgData::Cell(value, _) => value,
                DcgData::Thunk(thunk) => thunk(),
                DcgData::Memo(_, value) => value,
            }
        } else {
            let value = match inner_node {
                DcgData::Cell(value, _) => value,
                DcgData::Thunk(thunk) => thunk(),
                DcgData::Memo(thunk, _) => thunk(),
            };
            match &mut self.graph.borrow_mut()[self.idx] {
                DcgData::Cell(_, ref mut dirty) => *dirty = false,
                DcgData::Thunk(_) => (),
                DcgData::Memo(_, cached) => *cached = value.clone(),
            };
            value
        }
    }

    /// Generates the node's value and cleans all transitive dependencies.
    ///
    /// This is the intended interface for retrieving DCG values.
    ///
    /// A Depth-First-Search is performed on the reversed graph edges. Dirty
    /// edges are collected. The desired value is then generated and returned
    /// after the dirty edges are cleaned.
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
    /// let thunk1 = dcg.thunk(&get_cell, &[cell]);
    /// let thunk2 = dcg.thunk(&get_cell, &[cell]);
    /// let get_thunk1 = || thunk1.get();
    /// let thunk3 = dcg.thunk(&get_thunk1, &[thunk1]);
    ///
    /// assert_eq!(cell.set(42), 1);
    ///
    /// /* BEFORE: all edges dirtied
    /// *
    /// *      thunk1 == thunk3
    /// *    //
    /// *   cell
    /// *    \\
    /// *      thunk2
    /// */
    ///
    /// assert_eq!(thunk3.query(), 42);
    ///
    /// /* AFTER: dependency edges cleaned
    /// *
    /// *     thunk1 -- thunk3
    /// *    /
    /// *   cell
    /// *    \\
    /// *      thunk2
    /// */
    ///
    /// let graph = dcg.borrow();
    ///
    /// assert!(!graph[graph.find_edge(cell.into(), thunk1.into()).unwrap()]);
    /// assert!(!graph[graph.find_edge(thunk1.into(), thunk3.into()).unwrap()]);
    /// assert!(graph[graph.find_edge(cell.into(), thunk2.into()).unwrap()]);
    ///
    /// ```
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
    T: PartialEq + Clone + Debug,
{
    /// Sets the [`DcgNode`]'s value to `new_value`, "dirtying" all dependent
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
    /// The dirtying phase performs a Depth-First-Search from `node` and sets
    /// the weight of each edge encountered to [`true`]
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
    /// let thunk1 = dcg.thunk(&get_cell, &[cell]);
    /// let thunk2 = dcg.thunk(&get_cell, &[cell]);
    /// let get_thunk1 = || thunk1.get();
    /// let thunk3 = dcg.thunk(&get_thunk1, &[thunk1]);
    ///
    /// /* BEFORE: no dirty edges
    /// *
    /// *     thunk1 -- thunk3
    /// *    /
    /// *   cell
    /// *    \
    /// *     thunk2
    /// */
    ///
    /// assert!(!dcg.borrow_mut().edge_weights_mut().all(|weight| *weight));
    ///
    /// assert_eq!(cell.set(42), 1);
    ///
    /// /* AFTER: all edges dirtied
    /// *
    /// *      thunk1 == thunk3
    /// *    //
    /// *   cell
    /// *    \\
    /// *      thunk2
    /// */
    ///
    /// assert!(dcg.borrow_mut().edge_weights_mut().all(|weight| *weight));
    /// ```
    pub fn set(&self, new_value: T) -> T {
        let value = match &mut self.graph.borrow_mut()[self.idx] {
            DcgData::Cell(ref mut value, ref mut dirty) => {
                if *value == new_value {
                    return new_value;
                }
                *dirty = true;
                let tmp = value.clone();
                *value = new_value;
                tmp
            }
            _ => unreachable!(),
        };

        let mut edges_to_dirty = Vec::new();
        {
            let dcg = self.graph.borrow();
            let mut dfs = Dfs::new(&*dcg, self.idx);
            while let Some(node) = dfs.next(&*dcg) {
                edges_to_dirty.extend(dcg.edges(node).filter_map(|edge| {
                    if *edge.weight() {
                        None
                    } else {
                        Some(edge.id())
                    }
                }));
            }
        }

        let mut dcg = self.graph.borrow_mut();
        edges_to_dirty.iter().for_each(|&edge| {
            dcg[edge] = true;
        });
        value
    }
}

/// Workaround for a [bug](https://github.com/rust-lang/rust/issues/26925) in
/// [`PhantomData`]
impl<T, Ty> Clone for DcgNode<'_, T, Ty>
where
    T: PartialEq + Clone + Debug,
{
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

/// Workaround for a [bug](https://github.com/rust-lang/rust/issues/26925) in
/// [`PhantomData`]
impl<T, Ty> Copy for DcgNode<'_, T, Ty> where T: PartialEq + Clone + Debug {}

impl<T, Ty> From<DcgNode<'_, T, Ty>> for NodeIndex
where
    T: PartialEq + Clone + Debug,
{
    fn from(node: DcgNode<'_, T, Ty>) -> Self {
        node.idx
    }
}

impl<T> Debug for DcgData<'_, T>
where
    T: PartialEq + Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DcgData::Cell(value, dirty) => {
                f.debug_tuple("Cell").field(&value).field(&dirty).finish()
            }
            DcgData::Thunk(_) => f.debug_tuple("Thunk").finish(),
            DcgData::Memo(_, last_value) => f.debug_tuple("Memo").field(&last_value).finish(),
        }
    }
}

type GraphRepr<'a, T> = RefCell<DiGraph<DcgData<'a, T>, bool>>;

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
    T: PartialEq + Clone + Debug;

impl<'a, T> Deref for Dcg<'a, T>
where
    T: PartialEq + Clone + Debug,
{
    type Target = GraphRepr<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> DerefMut for Dcg<'a, T>
where
    T: PartialEq + Clone + Debug,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T> Dcg<'a, T>
where
    T: PartialEq + Clone + Debug,
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
            idx: self.borrow_mut().add_node(DcgData::Cell(value, false)),
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
    /// use dcg::{Dcg, DcgData};
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
    ///     DcgData::Memo(_, value) => assert_eq!(value, 1),
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
            idx: self.borrow_mut().add_node(DcgData::Thunk(thunk)),
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
    /// use dcg::{Dcg, DcgData};
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
    ///     DcgData::Memo(_, value) => assert_eq!(value, 42),
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
            idx: self.borrow_mut().add_node(DcgData::Memo(thunk, value)),
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
        }

        assert!(!dcg.borrow_mut().edge_weights_mut().any(|weight| *weight));

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
        }

        assert!(!dcg.borrow_mut().edge_weights_mut().any(|weight| *weight));

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
        }

        assert!(!dcg.borrow_mut().edge_weights_mut().any(|weight| *weight));

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

        assert!(a.is_dirty());
        assert!(thunk.is_dirty());

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

        assert!(a.is_clean());
        assert!(thunk.is_clean());

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
