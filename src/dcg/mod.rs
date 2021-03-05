use std::cell::RefCell;

use petgraph::Direction;
use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::{depth_first_search, DfsEvent},
};

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
            Node::Memo(thunk, value) => match value {
                Some(value) => value.clone(),
                None => thunk().clone(),
            },
        }
    }

    /// Dirties all transitively dependent nodes
    pub fn set(&self, node: NodeIndex, new_value: T) -> Result<(), ()> {
        match self.graph.borrow_mut().node_weight_mut(node).unwrap() {
            Node::Cell(ref mut value) => {
                *value = new_value;
            }
            _ => return Err(()),
        };

        let mut transitive_edges = Vec::new();
        let dcg = self.graph.borrow();
        depth_first_search(&*dcg, Some(node), |event| {
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

        let mut dcg = self.graph.borrow_mut();
        transitive_edges.iter().for_each(|&edge| {
            *dcg.edge_weight_mut(edge).unwrap() = true;
        });
        Ok(())
    }
}

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
        let a = dcg.cell(1);

        let get_a = || dcg.get(a);
        let thunk = dcg.thunk(&get_a, &[a]);

        {
            let graph = dcg.graph.borrow();

            assert_eq!(graph.node_count(), 2);

            assert!(graph.contains_edge(a, thunk));
        }

        assert!(dcg
            .graph
            .borrow_mut()
            .edge_weights_mut()
            .all(|weight| !*weight));

        assert_eq!(dcg.get(thunk), 1);
    }

    #[test]
    fn create_memo() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);

        let get_a = || dcg.get(a);
        let memo = dcg.memo(&get_a, &[a]);

        {
            let graph = dcg.graph.borrow();

            assert_eq!(graph.node_count(), 2);

            assert!(graph.contains_edge(a, memo));
        }

        assert!(dcg
            .graph
            .borrow_mut()
            .edge_weights_mut()
            .all(|weight| !*weight));

        assert_eq!(dcg.get(memo), 1);
    }

    #[test]
    fn create_lone_thunk() {
        let dcg = Dcg::new();
        let thunk = dcg.lone_thunk(&|| 42);

        assert_eq!(dcg.graph.borrow().node_count(), 1);

        assert_eq!(dcg.get(thunk), 42);
    }

    #[test]
    fn create_lone_memo() {
        let dcg = Dcg::new();
        let memo = dcg.lone_memo(&|| 42);

        assert_eq!(dcg.graph.borrow().node_count(), 1);

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
            let graph = dcg.graph.borrow();

            assert_eq!(graph.node_count(), 4);

            assert!(graph.contains_edge(a, thunk1));
            assert!(graph.contains_edge(a, thunk2));
            assert!(graph.contains_edge(thunk1, thunk3));
            assert!(graph.contains_edge(thunk2, thunk3));
        }

        assert!(dcg
            .graph
            .borrow_mut()
            .edge_weights_mut()
            .all(|weight| !*weight));

        assert_eq!(dcg.get(thunk1), 1);
        assert_eq!(dcg.get(thunk2), 1);
        assert_eq!(dcg.get(thunk3), 2);
    }
}
