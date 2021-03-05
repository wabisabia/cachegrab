#![allow(dead_code, unused)]

use std::cell::RefCell;

use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::depth_first_search,
    visit::Control,
    visit::ControlFlow,
    visit::IntoNeighbors,
};
use petgraph::{visit::DfsEvent, Direction};

pub enum Node<'a, T>
where
    T: Clone,
{
    Cell(T),
    Thunk(&'a dyn Fn() -> T),
    Memo(&'a dyn Fn() -> T, Option<T>),
}

impl<T> std::fmt::Debug for Node<'_, T>
where
    T: Clone + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Node::Cell(value) => write!(f, "{:?}", value),
            Node::Thunk(_) => f.debug_tuple("Thunk").finish(),
            Node::Memo(_, last_value) => f.debug_tuple("Memo").field(&last_value).finish(),
        }
    }
}

pub struct Dcg<'a, T>
where
    T: Clone,
{
    pub graph: RefCell<DiGraph<Node<'a, T>, bool>>,
}

impl<'a, T> Dcg<'a, T>
where
    T: Clone,
{
    pub fn new() -> Self {
        Self {
            graph: RefCell::new(DiGraph::new()),
        }
    }

    fn is_dirty(&self, node: NodeIndex) -> bool {
        self.graph
            .borrow()
            .edges_directed(node, Direction::Incoming)
            .any(|edge| *edge.weight())
    }

    fn is_clean(&self, node: NodeIndex) -> bool {
        self.is_dirty(node)
    }

    fn is_independent(&self, node: NodeIndex) -> bool {
        self.graph
            .borrow()
            .edges_directed(node, Direction::Incoming)
            .next()
            .is_none()
    }

    fn add_dependencies(&self, node: NodeIndex, dependencies: &[NodeIndex]) {
        let dep_states: Vec<_>;
        {
            dep_states = dependencies
                .iter()
                .map(|&dep| (dep, self.is_dirty(dep)))
                .collect();
        }
        let mut dcg = self.graph.borrow_mut();
        dep_states.iter().for_each(|(dep, dirty)| {
            dcg.add_edge(*dep, node, *dirty);
        });
    }

    /// Creates and adds a cell to the dependency graph.
    pub fn cell(&self, value: T) -> NodeIndex {
        self.graph.borrow_mut().add_node(Node::Cell(value))
    }

    /// Creates and adds a thunk and its dependencies to the dependency graph.
    pub fn thunk<F>(&self, thunk: &'a F, dependencies: &[NodeIndex]) -> NodeIndex
    where
        F: Fn() -> T,
    {
        let node = self.graph.borrow_mut().add_node(Node::Thunk(thunk));
        self.add_dependencies(node, dependencies);
        node
    }

    /// Creates and adds a memo'd thunk and its dependencies to the dependency graph.
    pub fn memo<F>(&self, thunk: &'a F, dependencies: &[NodeIndex]) -> NodeIndex
    where
        F: Fn() -> T,
    {
        let node = self.graph.borrow_mut().add_node(Node::Memo(thunk, None));
        self.add_dependencies(node, dependencies);
        node
    }

    /// Creates and adds a thunk to the dependency graph.
    ///
    /// To be used where the thunk is not dependent on any DCG calls.
    ///
    /// If this is not the case, instead use [`Dcg::thunk`].
    pub fn lone_thunk<F>(&self, thunk: &'a F) -> NodeIndex
    where
        F: Fn() -> T,
    {
        self.graph.borrow_mut().add_node(Node::Thunk(thunk))
    }

    /// Creates and adds a memo'd thunk to the dependency graph.
    ///
    /// To be used where the thunk is not dependent on any DCG calls.
    ///
    /// If this is not the case, instead use [`Dcg::thunk`].
    pub fn lone_memo<F>(&self, thunk: &'a F) -> NodeIndex
    where
        F: Fn() -> T,
    {
        self.graph
            .borrow_mut()
            .add_node(Node::Memo(thunk, Some(thunk())))
    }

    pub fn get(&self, node: NodeIndex) -> T {
        match self.graph.borrow().node_weight(node).unwrap() {
            Node::Cell(value) => value.clone(),
            Node::Thunk(thunk) => thunk().clone(),
            Node::Memo(thunk, value) => thunk().clone(),
        }
    }

    /// Dirties all transitively dependent nodes
    pub fn set(&self, node: NodeIndex, value: T) -> Result<DirtyDiagnostics, ()> {
        let mut diagnostics = DirtyDiagnostics {};
        let mut transitive_edges = Vec::new();
        match self.graph.borrow().node_weight(node).unwrap() {
            Node::Cell(value) => {
                let dcg = self.graph.borrow();
                depth_first_search(&*dcg, Some(node), |event| {
                    if let DfsEvent::TreeEdge(i, j) = event {
                        transitive_edges.push(dcg.find_edge(i, j).unwrap());
                    }
                    if let DfsEvent::CrossForwardEdge(i, j) = event {
                        transitive_edges.push(dcg.find_edge(i, j).unwrap());
                    }
                });
            }
            node => return Err(()),
        };
        for edge in transitive_edges {
            *self.graph.borrow_mut().edge_weight_mut(edge).unwrap() = true;
        }
        *self.graph.borrow_mut().node_weight_mut(node).unwrap() = Node::Cell(value);
        Ok(diagnostics)
    }
}

pub struct DirtyDiagnostics {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_cell() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);
        assert_eq!(dcg.get(a), 1);
    }

    #[test]
    fn create_thunk() {
        let dcg = Dcg::new();
        let thunk = dcg.lone_thunk(&|| 42);

        assert_eq!(dcg.get(thunk), 42);
    }

    #[test]
    fn create_thunk_using_cell() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);
        let get_a = || dcg.get(a);
        let thunk = dcg.thunk(&get_a, &[a]);
        assert_eq!(dcg.get(thunk), 1);
    }

    #[test]
    fn create_thunk_with_two_cells() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);
        let b = dcg.cell(2);
        let add_ab = || dcg.get(a) + dcg.get(b);
        let thunk = dcg.thunk(&add_ab, &[a, b]);

        /*      a
         *       \
         *        t: a + b
         *       /
         *      b
         */

        assert_eq!(dcg.get(thunk), 3);
    }

    #[test]
    fn create_thunk_nested() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);
        let b = dcg.cell(2);

        let add_ab = || dcg.get(a) + dcg.get(b);
        let thunk1 = dcg.thunk(&add_ab, &[a, b]);

        let add_one = || dcg.get(thunk1) + 1;
        let thunk2 = dcg.thunk(&add_one, &[thunk1]);

        let add_two = || dcg.get(thunk1) + 2;
        let thunk3 = dcg.thunk(&add_two, &[thunk1]);

        let two_times_thunk_plus_three = || dcg.get(thunk2) + dcg.get(thunk3);
        let thunk4 = dcg.thunk(&two_times_thunk_plus_three, &[thunk2, thunk3]);

        /*      a          t2: t + 1
         *       \        /         \
         *        t: a + b           t4: t2 + t3 = 2t + 3
         *       /        \         /
         *      b          t3: t + 2
         */

        assert_eq!(dcg.get(thunk1), 3);
        assert_eq!(dcg.get(thunk2), 4);
        assert_eq!(dcg.get(thunk3), 5);
        assert_eq!(dcg.get(thunk4), 9);
    }

    #[test]
    fn string_dcg() {
        let dcg = Dcg::new();
        let something = dcg.cell("something".to_string());

        let borrowed = || format!("{} borrowed", dcg.get(something));
        let thunk_borrowed = dcg.thunk(&borrowed, &[something]);

        let blue = || format!("{} blue", dcg.get(something));
        let thunk_blue = dcg.thunk(&blue, &[something]);

        assert_eq!(dcg.get(thunk_borrowed), "something borrowed".to_string());
        assert_eq!(dcg.get(thunk_blue), "something blue".to_string());
    }

    #[test]
    fn create_thunk_with() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);
        let b = dcg.cell(2);

        let add_ab = || if dcg.get(a) == 1 { 0 } else { dcg.get(b) };
        let thunk = dcg.thunk(&add_ab, &[a, b]);

        /* a
         *  \
         *  t: if a == 1 then 0 else b
         *  /
         * b
         */

        assert_eq!(dcg.graph.borrow().edge_count(), 2);
        assert_eq!(dcg.get(thunk), 0);

        dcg.set(a, 2);

        assert_eq!(dcg.get(thunk), 2);

        println!("{:?}", dcg.graph);
    }
}
