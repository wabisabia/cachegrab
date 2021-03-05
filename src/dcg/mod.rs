use std::{cell::RefCell, marker::PhantomData, ops::Deref};

use petgraph::Direction;
use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::{depth_first_search, DfsEvent},
};

pub enum NodeTy<'a, T>
where
    T: Clone,
{
    Cell(T),
    Thunk(&'a dyn Fn() -> T),
    Memo(&'a dyn Fn() -> T, Option<T>),
}

pub enum Cell {}
pub enum Thunk {}
pub enum Memo {}

pub struct Node<Ty>(NodeIndex, PhantomData<Ty>);

impl<Ty> Clone for Node<Ty> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<Ty> Copy for Node<Ty> {}

impl<Ty> From<Node<Ty>> for NodeIndex {
    fn from(node: Node<Ty>) -> Self {
        node.0
    }
}

impl<Ty> From<NodeIndex> for Node<Ty> {
    fn from(idx: NodeIndex) -> Self {
        Self(idx, PhantomData)
    }
}

impl<T> std::fmt::Debug for NodeTy<'_, T>
where
    T: Clone + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NodeTy::Cell(value) => write!(f, "{:?}", value),
            NodeTy::Thunk(_) => f.debug_tuple("Thunk").finish(),
            NodeTy::Memo(_, last_value) => f.debug_tuple("Memo").field(&last_value).finish(),
        }
    }
}

type GraphRepr<'a, T> = RefCell<DiGraph<NodeTy<'a, T>, bool>>;

pub struct Dcg<'a, T>(pub GraphRepr<'a, T>)
where
    T: Clone;

impl<'a, T> Deref for Dcg<'a, T>
where
    T: Clone,
{
    type Target = GraphRepr<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> Dcg<'a, T>
where
    T: Clone,
{
    /// Creates an empty DCG.
    /// # Examples
    /// ```
    /// use dcg::dcg::Dcg;
    /// let dcg: Dcg<i64> = Dcg::new();
    ///
    /// assert_eq!(dcg.borrow().node_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self(RefCell::new(DiGraph::new()))
    }

    fn is_dirty(&self, node: NodeIndex) -> bool {
        self.borrow()
            .edges_directed(node, Direction::Incoming)
            .any(|edge| *edge.weight())
    }

    fn add_dependencies(&self, node: NodeIndex, dependencies: &[NodeIndex]) {
        let dep_states: Vec<_>;
        {
            dep_states = dependencies
                .iter()
                .map(|&dep| (dep, self.is_dirty(dep)))
                .collect();
        }
        let mut dcg = self.borrow_mut();
        dep_states.iter().for_each(|(dep, dirty)| {
            dcg.add_edge(*dep, node, *dirty);
        });
    }

    /// Creates and adds a cell to the dependency graph.
    /// # Examples
    /// ```
    /// use dcg::dcg::Dcg;
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// assert_eq!(dcg.borrow().node_count(), 1);
    /// assert_eq!(dcg.get(cell), 1);
    /// ```
    pub fn cell(&self, value: T) -> Node<Cell> {
        Node(self.borrow_mut().add_node(NodeTy::Cell(value)), PhantomData)
    }

    /// Creates and adds a thunk and its dependencies to the dependency graph.
    /// # Examples
    /// ```
    /// use dcg::dcg::Dcg;
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let get_cell = || dcg.get(cell);
    /// let thunk = dcg.thunk(&get_cell, &[cell]);
    ///
    /// let borrowed = dcg.borrow();
    ///
    /// assert_eq!(borrowed.node_count(), 2);
    ///
    /// assert!(borrowed.contains_edge(cell.into(), thunk.into()));
    ///
    /// assert_eq!(dcg.get(thunk), dcg.get(cell));
    /// ```
    pub fn thunk<F, Ty>(&self, thunk: &'a F, dependencies: &[Node<Ty>]) -> Node<Thunk>
    where
        F: Fn() -> T,
    {
        let node = self.borrow_mut().add_node(NodeTy::Thunk(thunk));
        self.add_dependencies(
            node,
            dependencies
                .iter()
                .map(|node| (*node).into())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        Node(node, PhantomData)
    }

    /// Creates and adds a memo'd thunk and its dependencies to the dependency graph.
    /// # Examples
    /// ```
    /// use dcg::dcg::{Dcg, NodeTy};
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let get_cell = || dcg.get(cell);
    /// let memo = dcg.memo(&get_cell, &[cell]);
    ///
    /// let borrowed = dcg.borrow();
    ///
    /// assert_eq!(borrowed.node_count(), 2);
    ///
    /// assert!(borrowed.contains_edge(cell.into(), memo.into()));
    ///
    /// assert_eq!(dcg.get(memo), dcg.get(cell));
    ///
    /// match dcg.borrow().node_weight(memo.into()).unwrap() {
    ///     NodeTy::Memo(_, Some(value)) => assert_eq!(*value, 1),
    ///     _ => (),
    /// };
    /// ```
    pub fn memo<F, Ty>(&self, thunk: &'a F, dependencies: &[Node<Ty>]) -> Node<Memo>
    where
        F: Fn() -> T,
    {
        let node = self.borrow_mut().add_node(NodeTy::Memo(thunk, None));
        self.add_dependencies(
            node,
            dependencies
                .iter()
                .map(|node| (*node).into())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        Node(node, PhantomData)
    }

    /// Creates and adds a thunk with no dependencies to the dependency graph.
    ///
    /// To be used where the thunk is not dependent on any DCG nodes. If this is not the case, instead use [`Dcg::thunk`].
    /// # Examples
    /// ```
    /// use dcg::dcg::Dcg;
    /// let dcg = Dcg::new();
    ///
    /// let meaning_of_life = || 42;
    /// let thunk = dcg.lone_thunk(&meaning_of_life);
    ///
    /// assert_eq!(dcg.borrow().node_count(), 1);
    ///
    /// assert_eq!(dcg.get(thunk), 42);
    /// ```
    pub fn lone_thunk<F>(&self, thunk: &'a F) -> Node<Thunk>
    where
        F: Fn() -> T,
    {
        Node(
            self.borrow_mut().add_node(NodeTy::Thunk(thunk)),
            PhantomData,
        )
    }

    /// Creates and adds a memo'd thunk with no dependencies to the dependency graph.
    ///
    /// To be used where the thunk is not dependent on any DCG nodes. If this is not the case, instead use [`Dcg::memo`].
    /// # Examples
    /// ```
    /// use dcg::dcg::{Dcg, NodeTy};
    /// let dcg = Dcg::new();
    ///
    /// let meaning_of_life = || 42;
    /// let memo = dcg.lone_memo(&meaning_of_life);
    ///
    /// assert_eq!(dcg.borrow().node_count(), 1);
    ///
    /// assert_eq!(dcg.get(memo), 42);
    ///
    /// match dcg.borrow().node_weight(memo.into()).unwrap() {
    ///     NodeTy::Memo(_, Some(value)) => assert_eq!(*value, 42),
    ///     _ => (),
    /// };
    /// ```
    pub fn lone_memo<F>(&self, thunk: &'a F) -> Node<Memo>
    where
        F: Fn() -> T,
    {
        Node(
            self.borrow_mut()
                .add_node(NodeTy::Memo(thunk, Some(thunk()))),
            PhantomData,
        )
    }

    pub fn get<Ty>(&self, node: Node<Ty>) -> T {
        // TODO: The tricky bit...
        match self.borrow().node_weight(node.into()).unwrap() {
            NodeTy::Cell(value) => value.clone(),
            NodeTy::Thunk(thunk) => thunk().clone(),
            NodeTy::Memo(thunk, value) => match value {
                Some(value) => value.clone(),
                None => thunk().clone(),
            },
        }
    }

    /// Sets the value of `node` to `new_value`, "dirtying" all dependent
    /// nodes.
    ///
    /// Dirties all nodes that are transitively dependent on `node` and
    /// returns the previous cell value.
    ///
    /// This function only accepts nodes generates by [`Dcg::cell`]:
    /// ```compile_fail
    /// let dcg = Dcg::new();
    ///
    /// let x = || 42;
    /// let thunk = dcg.lone_thunk(&x);
    ///
    /// dcg.set(thunk, &x);
    /// ```
    ///
    /// The dirtying phase performs a Depth-First-Search from `node` and at
    /// each tree/cross edge, sets its weight to [`true`].
    ///
    /// # Examples
    /// ```
    /// use dcg::dcg::Dcg;
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let get_cell = || dcg.get(cell);
    /// let thunk1 = dcg.thunk(&get_cell, &[cell]);
    /// let thunk2 = dcg.thunk(&get_cell, &[cell]);
    /// let get_thunk1 = || dcg.get(thunk1);
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
    /// assert_eq!(dcg.set(cell, 42), 1);
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
    /// assert_eq!(dcg.get(cell), 42);
    ///
    /// assert!(dcg.borrow_mut().edge_weights_mut().all(|weight| *weight));
    /// ```
    pub fn set(&self, node: Node<Cell>, new_value: T) -> T {
        let idx = node.into();
        let value = match self.borrow_mut().node_weight_mut(idx).unwrap() {
            NodeTy::Cell(ref mut value) => {
                let tmp = value.clone();
                *value = new_value;
                tmp
            }
            _ => unreachable!(),
        };

        let mut transitive_edges = Vec::new();
        {
            let dcg = self.borrow();
            depth_first_search(&*dcg, Some(idx), |event| {
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

        let mut dcg = self.borrow_mut();
        transitive_edges.iter().for_each(|&edge| {
            *dcg.edge_weight_mut(edge).unwrap() = true;
        });
        value
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

        assert_eq!(dcg.get(a), 1);
    }

    #[test]
    fn create_thunk() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);

        let get_a = || dcg.get(a);
        let thunk = dcg.thunk(&get_a, &[a]);

        {
            let graph = dcg.borrow();

            assert_eq!(graph.node_count(), 2);

            assert!(graph.contains_edge(a.into(), thunk.into()));
        }

        assert!(dcg.borrow_mut().edge_weights_mut().all(|weight| !*weight));

        assert_eq!(dcg.get(thunk), 1);
    }

    #[test]
    fn create_memo() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);

        let get_a = || dcg.get(a);
        let memo = dcg.memo(&get_a, &[a]);

        {
            let graph = dcg.borrow();

            assert_eq!(graph.node_count(), 2);

            assert!(graph.contains_edge(a.into(), memo.into()));
        }

        assert!(dcg.borrow_mut().edge_weights_mut().all(|weight| !*weight));

        assert_eq!(dcg.get(memo), 1);
    }

    #[test]
    fn create_lone_thunk() {
        let dcg = Dcg::new();
        let thunk = dcg.lone_thunk(&|| 42);

        assert_eq!(dcg.borrow().node_count(), 1);

        assert_eq!(dcg.get(thunk), 42);
    }

    #[test]
    fn create_lone_memo() {
        let dcg = Dcg::new();
        let memo = dcg.lone_memo(&|| 42);

        assert_eq!(dcg.borrow().node_count(), 1);

        assert_eq!(dcg.get(memo), 42);
    }

    #[test]
    fn thunk_nested() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        let get_a = || dcg.get(a);

        let thunk1 = dcg.thunk(&get_a, &[a]);
        let thunk2 = dcg.thunk(&get_a, &[a]);

        let add = || dcg.get(thunk1) + dcg.get(thunk2);
        let thunk3 = dcg.thunk(&add, &[thunk1, thunk2]);

        {
            let graph = dcg.borrow();

            assert_eq!(graph.node_count(), 4);

            assert!(graph.contains_edge(a.into(), thunk1.into()));
            assert!(graph.contains_edge(a.into(), thunk2.into()));
            assert!(graph.contains_edge(thunk1.into(), thunk3.into()));
            assert!(graph.contains_edge(thunk2.into(), thunk3.into()));
        }

        assert!(dcg.borrow_mut().edge_weights_mut().all(|weight| !*weight));

        assert_eq!(dcg.get(thunk1), 1);
        assert_eq!(dcg.get(thunk2), 1);
        assert_eq!(dcg.get(thunk3), 2);
    }

    #[test]
    fn dirtying_phase() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        let get_a = || dcg.get(a);
        let thunk = dcg.thunk(&get_a, &[a]);

        assert_eq!(dcg.get(thunk), 1);

        assert_eq!(dcg.set(a, 2), 1);

        let graph = dcg.borrow();

        assert_eq!(
            *graph
                .edge_weight(graph.find_edge(a.into(), thunk.into()).unwrap())
                .unwrap(),
            true
        );

        assert_eq!(dcg.get(thunk), 2);
    }
}
