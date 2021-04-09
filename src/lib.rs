use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::Dfs,
    visit::EdgeRef,
    visit::IntoEdges,
    visit::Reversed,
    EdgeDirection::Incoming,
};

pub struct Dcg {
    pub graph: Rc<RefCell<GraphRepr>>,
}

type GraphRepr = DiGraph<(), bool>;

/// Internal graph node type. Stores the type and data of a [`Dcg`] graph node.
#[derive(Clone)]
pub enum DcgData<T> {
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
    Thunk(Rc<dyn Fn() -> T>),

    /// Contains a thunk which produces a value of type `T` and a cached value
    /// of type `T` which holds the result of the previous evaluation of the
    /// thunk.
    ///
    /// When [`DcgNode::get`] is called on the corresponding [`DcgNode<Memo>`] the
    /// value returned depends on the cleanliness of the node's dependencies.
    /// - If any dependency is dirty, the [`DcgNode<Memo>`] is dirty and thunk
    /// is re-evaluated, cached and returned.
    /// - Otherwise, the cached value is returned.
    Memo(Rc<dyn Fn() -> T>, T),
}

impl Dcg {
    pub fn new() -> Self {
        Dcg {
            graph: Rc::new(RefCell::new(DiGraph::new())),
        }
    }

    pub fn cell<T>(&self, value: T) -> Rc<DcgNode<T, CellTy>> {
        Rc::new(DcgNode {
            idx: self.graph.borrow_mut().add_node(()),
            graph: self.graph.clone(),
            data: RefCell::new(DcgData::Cell(value, true)),
            phantom: PhantomData,
        })
    }

    pub fn thunk<'a, T>(
        &'a self,
        thunk: Rc<dyn Fn() -> T>,
        dependencies: &[NodeIndex],
    ) -> DcgNode<T, ThunkTy> {
        let idx = self.graph.borrow_mut().add_node(());
        self.add_dependencies(idx, dependencies);
        DcgNode {
            idx,
            graph: self.graph.clone(),
            data: RefCell::new(DcgData::Thunk(thunk)),
            phantom: PhantomData,
        }
    }

    pub fn memo<'a, T>(
        &'a self,
        thunk: Rc<dyn Fn() -> T>,
        dependencies: &[NodeIndex],
    ) -> DcgNode<T, MemoTy> {
        let idx = self.graph.borrow_mut().add_node(());
        self.add_dependencies(idx, dependencies);
        let value = thunk();
        DcgNode {
            idx,
            graph: self.graph.clone(),
            data: RefCell::new(DcgData::Memo(thunk, value)),
            phantom: PhantomData,
        }
    }

    fn add_dependencies(&self, idx: NodeIndex, dependencies: &[NodeIndex]) {
        let mut dcg = self.graph.borrow_mut();
        dependencies.iter().for_each(|dep| {
            dcg.add_edge(*dep, idx, true);
        });
    }

    pub fn lone_thunk<'a, T>(&'a self, thunk: Rc<dyn Fn() -> T>) -> DcgNode<T, ThunkTy> {
        DcgNode {
            idx: self.graph.borrow_mut().add_node(()),
            graph: self.graph.clone(),
            data: RefCell::new(DcgData::Thunk(thunk)),
            phantom: PhantomData,
        }
    }

    pub fn lone_memo<'a, T>(&'a self, thunk: Rc<dyn Fn() -> T>) -> DcgNode<T, MemoTy> {
        let value = thunk();
        DcgNode {
            idx: self.graph.borrow_mut().add_node(()),
            graph: self.graph.clone(),
            data: RefCell::new(DcgData::Memo(thunk, value)),
            phantom: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct DcgNode<T, Ty> {
    pub idx: NodeIndex,
    graph: Rc<RefCell<GraphRepr>>,
    data: RefCell<DcgData<T>>,
    phantom: PhantomData<Ty>,
}

#[derive(Clone)]
pub struct CellTy;
#[derive(Clone)]
pub struct ThunkTy;
#[derive(Clone)]
pub struct MemoTy;

impl<T, Ty> DcgNode<T, Ty>
where
    T: Clone,
{
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
        match *self.data.borrow_mut() {
            DcgData::Cell(ref mut value, ref mut dirty) => {
                *dirty = false;
                value.clone()
            }
            DcgData::Thunk(ref thunk) => thunk(),
            DcgData::Memo(ref thunk, ref mut cached) => {
                if self
                    .graph
                    .borrow()
                    .edges_directed(self.idx, Incoming)
                    .any(|edge| *edge.weight())
                {
                    let tmp = thunk();
                    *cached = tmp.clone();
                    tmp
                } else {
                    cached.clone()
                }
            }
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
    /// use std::rc::Rc;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let cell1 = cell.clone();
    /// let get_cell = Rc::new(move || cell1.get());
    /// let thunk1 = dcg.thunk(get_cell.clone(), &[cell.idx]);
    /// let thunk2 = dcg.thunk(get_cell, &[cell.idx]);
    /// let thunk11 = thunk1.clone();
    /// let get_thunk1 = Rc::new(move || thunk11.get());
    /// let thunk3 = dcg.thunk(get_thunk1, &[thunk1.idx]);
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
    /// let graph = dcg.graph.borrow();
    ///
    /// assert!(!graph[graph.find_edge(cell.idx, thunk1.idx).unwrap()]);
    /// assert!(!graph[graph.find_edge(thunk1.idx, thunk3.idx).unwrap()]);
    /// assert!(graph[graph.find_edge(cell.idx, thunk2.idx).unwrap()]);
    ///
    /// ```
    pub fn query(&self) -> T {
        let mut data = self.data.borrow_mut();
        let thunk = match *data {
            DcgData::Cell(ref mut value, ref mut dirty) => {
                *dirty = false;
                return value.clone();
            }
            DcgData::Thunk(ref mut thunk) => thunk,
            DcgData::Memo(ref mut thunk, ref mut cached) => {
                if self
                    .graph
                    .borrow()
                    .edges_directed(self.idx, Incoming)
                    .any(|edge| *edge.weight())
                {
                    thunk
                } else {
                    return cached.clone();
                }
            }
        };
        let mut dirty_edges = Vec::new();
        {
            let dcg = self.graph.borrow();
            let rev_dcg = Reversed(&*dcg);
            let mut dfs = Dfs::new(rev_dcg, self.idx);
            while let Some(node) = dfs.next(rev_dcg) {
                dirty_edges.extend(rev_dcg.edges(node).filter_map(|edge| {
                    if *edge.weight() {
                        Some(edge.id())
                    } else {
                        None
                    }
                }));
            }
        }
        dirty_edges.iter().for_each(|edge| {
            self.graph.borrow_mut()[*edge] = false;
        });
        thunk()
    }

    pub fn dirty(&self) -> bool {
        use DcgData::*;
        match *self.data.borrow() {
            Cell(_, dirty) => dirty,
            _ => self
                .graph
                .borrow()
                .edges_directed(self.idx, Incoming)
                .any(|edge| *edge.weight()),
        }
    }

    pub fn clean(&self) -> bool {
        !self.dirty()
    }
}

impl<T> DcgNode<T, CellTy>
where
    T: PartialEq + Clone,
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
    /// use std::rc::Rc;
    ///
    /// let dcg = Dcg::new();
    ///
    /// let cell = dcg.cell(1);
    ///
    /// let cell1 = cell.clone();
    /// let get_cell = Rc::new(move || cell1.get());
    /// let thunk1 = dcg.thunk(get_cell.clone(), &[cell.idx]);
    /// let thunk2 = dcg.thunk(get_cell, &[cell.idx]);
    /// let thunk11 = thunk1.clone();
    /// let get_thunk1 = Rc::new(move || thunk11.get());
    /// let thunk3 = dcg.thunk(get_thunk1, &[thunk1.idx]);
    ///
    /// thunk2.query();
    /// thunk3.query();
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
    /// assert!(!dcg.graph.borrow_mut().edge_weights_mut().all(|weight| *weight));
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
    /// assert!(cell.dirty());
    /// assert!(thunk1.dirty());
    /// assert!(thunk2.dirty());
    /// assert!(thunk3.dirty());
    ///
    /// assert!(dcg.graph.borrow_mut().edge_weights_mut().all(|weight| *weight));
    /// ```
    pub fn set(&self, mut new_value: T) -> T {
        let value = match *self.data.borrow_mut() {
            DcgData::Cell(ref mut value, ref mut dirty) => {
                if new_value == *value {
                    return new_value;
                }
                std::mem::swap(value, &mut new_value);
                *dirty = true;
                new_value.clone()
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

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;

    #[test]
    fn create_cell() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);

        assert_eq!(dcg.graph.borrow().node_count(), 1);

        assert_eq!(a.query(), 1);
    }

    #[test]
    fn create_thunk() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);
        let a1 = a.clone();

        let get_a = Rc::new(move || a1.get());
        let thunk = dcg.thunk(get_a, &[a.idx]);

        {
            let graph = dcg.graph.borrow();

            assert_eq!(graph.node_count(), 2);

            assert!(graph.contains_edge(a.idx, thunk.idx));

            assert!(graph[graph.find_edge(a.idx, thunk.idx).unwrap()]);
        }

        assert!(a.dirty());
        assert!(thunk.dirty());

        assert_eq!(thunk.query(), 1);
    }

    #[test]
    fn create_memo() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);
        let a1 = a.clone();

        let get_a = Rc::new(move || a1.get());
        let memo = dcg.memo(get_a, &[a.idx]);

        {
            let graph = dcg.graph.borrow();

            assert_eq!(graph.node_count(), 2);

            assert!(graph.contains_edge(a.idx, memo.idx));

            assert!(graph[graph.find_edge(a.idx, memo.idx).unwrap()]);
        }

        assert!(a.clean());
        assert!(memo.dirty());

        assert_eq!(memo.query(), 1);
    }

    #[test]
    fn create_lone_thunk() {
        let dcg = Dcg::new();

        let thunk = dcg.lone_thunk(Rc::new(|| 42));

        assert_eq!(dcg.graph.borrow().node_count(), 1);

        assert_eq!(thunk.query(), 42);
    }

    #[test]
    fn create_lone_memo() {
        let dcg = Dcg::new();

        let memo = dcg.lone_memo(Rc::new(|| 42));

        assert_eq!(dcg.graph.borrow().node_count(), 1);

        assert_eq!(memo.query(), 42);
    }

    #[test]
    fn thunk_nested() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);
        let a1 = a.clone();

        let get_a = Rc::new(move || a1.get());

        let thunk1 = dcg.thunk(get_a.clone(), &[a.idx]);
        let thunk11 = thunk1.clone();
        let thunk2 = dcg.thunk(get_a.clone(), &[a.idx]);
        let thunk21 = thunk2.clone();

        let add = Rc::new(move || thunk11.get() + thunk21.get());
        let thunk3 = dcg.thunk(add, &[thunk1.idx, thunk2.idx]);

        {
            let graph = dcg.graph.borrow();

            assert_eq!(graph.node_count(), 4);

            assert!(graph.contains_edge(a.idx, thunk1.idx));
            assert!(graph.contains_edge(a.idx, thunk2.idx));
            assert!(graph.contains_edge(thunk1.idx, thunk3.idx));
            assert!(graph.contains_edge(thunk2.idx, thunk3.idx));

            assert!(graph[graph.find_edge(a.idx, thunk1.idx).unwrap()]);
            assert!(graph[graph.find_edge(a.idx, thunk2.idx).unwrap()]);
            assert!(graph[graph.find_edge(thunk1.idx, thunk3.idx).unwrap()]);
            assert!(graph[graph.find_edge(thunk2.idx, thunk3.idx).unwrap()]);
        }

        assert!(a.dirty());
        assert!(thunk1.dirty());
        assert!(thunk2.dirty());
        assert!(thunk3.dirty());

        assert_eq!(thunk1.query(), 1);
        assert_eq!(thunk2.query(), 1);
        assert_eq!(thunk3.query(), 2);
    }

    #[test]
    fn dirtying_phase() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);
        let a1 = a.clone();

        let get_a = Rc::new(move || a1.get());
        let thunk = dcg.thunk(get_a, &[a.idx]);

        assert_eq!(thunk.query(), 1);

        assert_eq!(a.set(2), 1);

        assert!(a.dirty());
        assert!(thunk.dirty());

        let graph = dcg.graph.borrow();

        assert!(graph[graph.find_edge(a.idx, thunk.idx).unwrap()]);
    }

    #[test]
    fn cleaning_phase() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);
        let a1 = a.clone();

        let get_a = Rc::new(move || a1.get());
        let thunk = dcg.thunk(get_a, &[a.idx]);

        a.set(2);

        assert_eq!(thunk.query(), 2);

        assert!(a.clean());
        assert!(thunk.clean());

        let graph = dcg.graph.borrow();

        assert!(!graph[graph.find_edge(a.idx, thunk.idx).unwrap()]);
    }

    #[test]
    fn cleaning_phase_two_layers() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);
        let a1 = a.clone();

        let get_a = Rc::new(move || a1.get());
        let b = dcg.thunk(get_a, &[a.idx]);
        let b1 = b.clone();

        let get_b = Rc::new(move || b1.get());
        let c = dcg.thunk(get_b, &[b.idx]);

        a.set(2);

        c.query();

        let graph = dcg.graph.borrow();

        assert!(!graph[graph.find_edge(b.idx, c.idx).unwrap()]);

        assert!(!graph[graph.find_edge(a.idx, b.idx).unwrap()]);
    }

    #[test]
    fn cleaning_phase_split() {
        let dcg = Dcg::new();

        let a = dcg.cell(1);
        let a1 = a.clone();

        let get_a = Rc::new(move || a1.get());

        let b = dcg.thunk(get_a.clone(), &[a.idx]);
        let c = dcg.thunk(get_a, &[a.idx]);

        a.set(2);

        c.query();

        let graph = dcg.graph.borrow();

        assert!(graph[graph.find_edge(a.idx, b.idx).unwrap()]);

        assert!(!graph[graph.find_edge(a.idx, c.idx).unwrap()]);
    }

    #[test]
    fn geometry() {
        let circle = Dcg::new();

        let radius = circle.cell(1.0);
        let pos = circle.cell((0.0, 0.0));

        let radius1 = radius.clone();
        let calc_area = Rc::new(move || {
            let r = radius1.get();
            std::f64::consts::PI * r * r
        });
        let area = circle.memo(calc_area, &[radius.idx]);

        let radius2 = radius.clone();
        let calc_circum = Rc::new(move || 2.0 * std::f64::consts::PI * radius2.get());
        let circum = circle.memo(calc_circum, &[radius.idx]);

        let pos1 = pos.clone();
        let radius3 = radius.clone();
        let calc_left_bound = Rc::new(move || {
            let (x, y) = pos1.get();
            (x - radius3.get(), y)
        });
        let left_bound = circle.memo(calc_left_bound, &[pos.idx, radius.idx]);

        assert!(radius.clean());
        assert!(area.dirty());
        assert!(circum.dirty());
        assert!(left_bound.dirty());

        assert_eq!(area.query(), std::f64::consts::PI);
        assert_eq!(circum.query(), 2.0 * std::f64::consts::PI);
        assert_eq!(left_bound.query(), (-1.0, 0.0));

        assert!(radius.clean());
        assert!(area.clean());
        assert!(circum.clean());
        assert!(left_bound.clean());

        assert_eq!(radius.set(2.0), 1.0);

        assert!(radius.dirty());
        assert!(area.dirty());
        assert!(circum.dirty());
        assert!(left_bound.dirty());

        assert_eq!(area.query(), 4.0 * std::f64::consts::PI);
        assert_eq!(circum.query(), 4.0 * std::f64::consts::PI);
        assert_eq!(left_bound.query(), (-2.0, 0.0));
    }

    #[test]
    fn multiplication_table() {
        let dcg = Dcg::new();

        let n: usize = 100;

        let nums = (1..=n).map(|i| Rc::new(dcg.cell(i))).collect::<Vec<_>>();

        let mut thunk_table = Vec::with_capacity(n.pow(2) / 2 + n);
        for i in 0..n {
            for j in i..n {
                let x = nums[i].clone();
                let x_idx = x.idx;
                let y = nums[j].clone();
                let y_idx = y.idx;
                thunk_table.push(dcg.thunk(Rc::new(move || x.get() * y.get()), &[x_idx, y_idx]));
            }
        }

        assert_eq!(thunk_table[0].query(), 1);
        assert_eq!(thunk_table[1].query(), 2);

        assert_eq!(nums[0].set(5), 1);

        assert_eq!(thunk_table[0].query(), 25);
        assert_eq!(thunk_table[1].query(), 10);
    }
}
