use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::Dfs,
    visit::EdgeRef,
    visit::IntoEdges,
    visit::Reversed,
    EdgeDirection::Incoming,
};

use std::{cell::RefCell, fmt::Debug, mem, ops::Deref, ops::DerefMut, rc::Rc};

type Graph = DiGraph<(), bool>;

#[derive(Debug, Clone, Default)]
pub struct Dcg {
    pub graph: Rc<RefCell<Graph>>,
}

impl Dcg {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn cell<T>(&self, value: T) -> IncCell<T> {
        IncCell::from_raw(RawIncCell {
            data: RefCell::new((value, true)),
            node: IcNode::from_dcg(self),
        })
    }

    pub fn thunk<T, F>(&self, f: F, deps: &[NodeIndex]) -> IncThunk<T>
    where
        F: Fn() -> T + 'static,
    {
        let node = IcNode::from_dcg(self);
        let mut graph = self.graph.borrow_mut();
        deps.iter().for_each(|&x| {
            graph.add_edge(x, node.idx, true);
        });
        IncThunk::from_raw(RawIncThunk {
            f: Rc::new(f),
            node,
        })
    }

    pub fn memo<T, F>(&self, f: F, deps: &[NodeIndex]) -> IncMemo<T>
    where
        F: Fn() -> T + 'static,
    {
        let node = IcNode::from_dcg(self);
        let cached = f();
        let mut graph = self.graph.borrow_mut();
        deps.iter().for_each(|&x| {
            // dependencies were cleaned when cached value was computed
            graph.add_edge(x, node.idx, false);
        });
        IncMemo::from_raw(RawIncMemo {
            f: Rc::new(f),
            cached,
            node,
        })
    }
}

#[derive(Debug, Clone)]
struct IcNode {
    graph: Rc<RefCell<Graph>>,
    idx: NodeIndex,
}

impl IcNode {
    fn from_dcg(dcg: &Dcg) -> Self {
        Self {
            graph: dcg.graph.clone(),
            idx: dcg.graph.borrow_mut().add_node(()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IncCell<T>(Rc<RawIncCell<T>>);

impl<T> IncCell<T> {
    fn from_raw(raw: RawIncCell<T>) -> Self {
        IncCell(Rc::new(raw))
    }
}

impl<T> Deref for IncCell<T> {
    type Target = Rc<RawIncCell<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for IncCell<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug)]
pub struct RawIncCell<T> {
    data: RefCell<(T, bool)>,
    node: IcNode,
}

impl<T> RawIncCell<T>
where
    T: PartialEq + Clone,
{
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }

    pub fn write(&self, new: T) -> T {
        let (ref mut value, ref mut dirty) = *self.data.borrow_mut();

        if *value == new {
            return new;
        }

        *dirty = true;

        {
            let mut graph = self.node.graph.borrow_mut();
            let mut dfs = Dfs::new(&*graph, self.idx());
            while let Some(node) = dfs.next(&*graph) {
                let edges = graph
                    .edges(node)
                    .filter_map(|edge| {
                        if *edge.weight() {
                            None
                        } else {
                            Some(edge.id())
                        }
                    })
                    .collect::<Vec<_>>();
                edges.iter().for_each(|edge| {
                    graph[*edge] = true;
                });
            }
        }

        mem::replace(value, new)
    }

    pub fn modify<F>(&self, f: F) -> T
    where
        F: FnOnce(&mut T) -> T,
    {
        let (ref mut value, ref mut dirty) = *self.data.borrow_mut();
        let new = f(value);

        if *value == new {
            return new;
        }

        *dirty = true;

        {
            let mut graph = self.node.graph.borrow_mut();
            let mut dfs = Dfs::new(&*graph, self.idx());
            while let Some(node) = dfs.next(&*graph) {
                let edges = graph
                    .edges(node)
                    .filter_map(|edge| {
                        if *edge.weight() {
                            None
                        } else {
                            Some(edge.id())
                        }
                    })
                    .collect::<Vec<_>>();
                edges.iter().for_each(|edge| {
                    graph[*edge] = true;
                });
            }
        }

        mem::replace(value, new)
    }
}

#[derive(Clone)]
pub struct IncThunk<T>(Rc<RawIncThunk<T>>);

impl<T> IncThunk<T> {
    fn from_raw(raw: RawIncThunk<T>) -> Self {
        Self(Rc::new(raw))
    }
}

impl<T> Deref for IncThunk<T> {
    type Target = RawIncThunk<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone)]
pub struct RawIncThunk<T> {
    f: Rc<dyn Fn() -> T + 'static>,
    node: IcNode,
}

impl<T> RawIncThunk<T> {
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }
}

#[derive(Clone)]
pub struct IncMemo<T>(Rc<RawIncMemo<T>>);

impl<T> IncMemo<T> {
    fn from_raw(raw: RawIncMemo<T>) -> Self {
        Self(Rc::new(raw))
    }
}

impl<T> Deref for IncMemo<T> {
    type Target = RawIncMemo<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone)]
pub struct RawIncMemo<T> {
    f: Rc<dyn Fn() -> T + 'static>,
    cached: T,
    node: IcNode,
}

impl<T> RawIncMemo<T> {
    pub fn idx(&self) -> NodeIndex {
        self.node.idx
    }
}

pub trait Incremental {
    type Output;

    /// Returns the node's most up-to-date value depending on its validity
    fn read(&self) -> Self::Output;

    /// If dirty, cleans transitive dependencies and returns the node's most up-to-date value
    fn query(&self) -> Self::Output;

    fn is_dirty(&self) -> bool {
        !self.is_clean()
    }

    fn is_clean(&self) -> bool {
        !self.is_dirty()
    }
}

impl<T: Clone> Incremental for RawIncCell<T> {
    type Output = T;

    fn read(&self) -> Self::Output {
        let (ref value, ref mut dirty) = *self.data.borrow_mut();
        *dirty = false;
        value.clone()
    }

    fn query(&self) -> Self::Output {
        self.read()
    }

    fn is_dirty(&self) -> bool {
        self.data.borrow().1
    }
}

impl<T> Incremental for RawIncThunk<T> {
    type Output = T;

    fn read(&self) -> Self::Output {
        (self.f)()
    }

    fn query(&self) -> Self::Output {
        if self.is_dirty() {
            let mut edges_to_clean = Vec::new();
            {
                let graph = self.node.graph.borrow();
                let rev_graph = Reversed(&*graph);
                let mut dfs = Dfs::new(rev_graph, self.idx());
                while let Some(node) = dfs.next(rev_graph) {
                    edges_to_clean.extend(rev_graph.edges(node).filter_map(|edge| {
                        if *edge.weight() {
                            Some(edge.id())
                        } else {
                            None
                        }
                    }));
                }
            }

            let mut graph = self.node.graph.borrow_mut();
            edges_to_clean.iter().for_each(|&edge| {
                graph[edge] = false;
            });
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

impl<T> Incremental for RawIncMemo<T>
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

        let mut edges_to_clean = Vec::new();
        {
            let graph = self.node.graph.borrow();
            let rev_graph = Reversed(&*graph);
            let mut dfs = Dfs::new(rev_graph, self.idx());
            while let Some(node) = dfs.next(rev_graph) {
                edges_to_clean.extend(rev_graph.edges(node).filter_map(|edge| {
                    if *edge.weight() {
                        Some(edge.id())
                    } else {
                        None
                    }
                }));
            }
        }

        {
            let mut graph = self.node.graph.borrow_mut();
            edges_to_clean.iter().for_each(|&edge| {
                graph[edge] = false;
            });
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

#[cfg(test)]
mod tests {
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

        let thunk = dcg.thunk(move || a1.read(), &[a.idx()]);

        {
            let graph = dcg.graph.borrow();
            assert_eq!(graph.node_count(), 2);
            assert!(graph.contains_edge(a.idx(), thunk.idx()));
            assert!(graph[graph.find_edge(a.idx(), thunk.idx()).unwrap()]);
        }

        assert!(a.is_dirty());
        assert!(thunk.is_dirty());
        assert_eq!(thunk.query(), 1);
    }

    #[test]
    fn create_memo() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);

        let a_inc = a.clone();
        let memo = dcg.memo(move || a_inc.read(), &[a.idx()]);

        {
            let graph = dcg.graph.borrow();
            assert_eq!(graph.node_count(), 2);
            assert!(graph.contains_edge(a.idx(), memo.idx()));
            assert!(!graph[graph.find_edge(a.idx(), memo.idx()).unwrap()]);
        }

        assert!(a.is_clean());
        assert!(memo.is_clean());
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
        let thunk = dcg.thunk(move || cell_inc.read(), &[cell.idx()]);

        thunk.query();
        cell.write(2);

        assert!(cell.is_dirty());
        let graph = dcg.graph.borrow();
        assert!(graph[graph.find_edge(cell.idx(), thunk.idx()).unwrap()]);
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
        let thunk = dcg.thunk(move || cell_inc.read(), &[cell.idx()]);

        assert_eq!(cell.modify(|x| *x + 1), 1);
        assert!(cell.is_dirty());
        let graph = dcg.graph.borrow();
        assert!(graph[graph.find_edge(cell.idx(), thunk.idx()).unwrap()]);
        assert_eq!(cell.query(), 2);
    }

    #[test]
    fn thunks_nest() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);

        let a_inc = a.clone();
        let thunk1 = dcg.thunk(move || a_inc.read(), &[a.idx()]);
        let thunk1_inc = thunk1.clone();
        let a_inc = a.clone();
        let thunk2 = dcg.thunk(move || a_inc.read(), &[a.idx()]);
        let thunk2_inc = thunk2.clone();
        let thunk3 = dcg.thunk(
            move || thunk1_inc.read() + thunk2_inc.read(),
            &[thunk1.idx(), thunk2.idx()],
        );

        {
            let graph = dcg.graph.borrow();
            assert_eq!(graph.node_count(), 4);
            assert!(graph.contains_edge(a.idx(), thunk1.idx()));
            assert!(graph.contains_edge(a.idx(), thunk2.idx()));
            assert!(graph.contains_edge(thunk1.idx(), thunk3.idx()));
            assert!(graph.contains_edge(thunk2.idx(), thunk3.idx()));
            assert!(graph[graph.find_edge(a.idx(), thunk1.idx()).unwrap()]);
            assert!(graph[graph.find_edge(a.idx(), thunk2.idx()).unwrap()]);
            assert!(graph[graph.find_edge(thunk1.idx(), thunk3.idx()).unwrap()]);
            assert!(graph[graph.find_edge(thunk2.idx(), thunk3.idx()).unwrap()]);
        }

        assert!(a.is_dirty());
        assert!(thunk1.is_dirty());
        assert!(thunk2.is_dirty());
        assert!(thunk3.is_dirty());

        assert_eq!(thunk1.query(), 1);
        assert_eq!(thunk2.query(), 1);
        assert_eq!(thunk3.query(), 2);
    }

    #[test]
    fn write_dirties_deep() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);
        let a_inc = a.clone();
        let thunk = dcg.thunk(move || a_inc.read(), &[a.idx()]);
        let thunk_inc = thunk.clone();
        let memo = dcg.memo(move || thunk_inc.read(), &[thunk.idx()]);

        assert_eq!(a.write(2), 1);

        assert!(a.is_dirty());
        assert!(thunk.is_dirty());
        assert!(memo.is_dirty());
        let graph = dcg.graph.borrow();
        assert!(graph[graph.find_edge(a.idx(), thunk.idx()).unwrap()]);
    }

    #[test]
    fn memo_query_cleans() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);
        let a_inc = a.clone();
        let memo = dcg.memo(move || a_inc.read(), &[a.idx()]);

        a.write(2);

        assert_eq!(memo.query(), 2);
        assert!(a.is_clean());
        assert!(memo.is_clean());
        let graph = dcg.graph.borrow();
        assert!(!graph[graph.find_edge(a.idx(), memo.idx()).unwrap()]);
    }

    #[test]
    fn thunk_query_cleans() {
        let dcg = Dcg::new();
        let a = dcg.cell(1);
        let a_inc = a.clone();
        let thunk = dcg.thunk(move || a_inc.read(), &[a.idx()]);

        a.write(2);

        assert_eq!(thunk.query(), 2);
        assert!(a.is_clean());
        assert!(thunk.is_clean());
        let graph = dcg.graph.borrow();
        assert!(!graph[graph.find_edge(a.idx(), thunk.idx()).unwrap()]);
    }

    #[test]
    fn thunk_query_cleans_deep() {
        let dcg = Dcg::new();
        let cell = dcg.cell(1);
        let cell_inc = cell.clone();
        let thunk1 = dcg.thunk(move || cell_inc.read(), &[cell.idx()]);
        let thunk1_inc = thunk1.clone();
        let thunk2 = dcg.thunk(move || thunk1_inc.read(), &[thunk1.idx()]);

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
        let thunk1 = dcg.thunk(move || cell_inc.read(), &[cell.idx()]);
        let thunk1_inc = cell.clone();
        let thunk2 = dcg.thunk(move || thunk1_inc.read(), &[cell.idx()]);

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
        let memo1 = dcg.memo(move || cell_inc.read(), &[cell.idx()]);
        let memo1_inc = memo1.clone();
        let memo2 = dcg.memo(move || memo1_inc.read(), &[memo1.idx()]);

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
        let memo1 = dcg.memo(move || cell_inc.read(), &[cell.idx()]);
        let memo1_inc = cell.clone();
        let memo2 = dcg.memo(move || memo1_inc.read(), &[cell.idx()]);

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

        let radius = circle.cell(1.0);
        let pos = circle.cell((0.0, 0.0));

        let radius_inc = radius.clone();
        let area = circle.memo(
            move || {
                let r = radius_inc.read();
                PI * r * r
            },
            &[radius.idx()],
        );

        let radius_inc = radius.clone();
        let circum = circle.memo(move || 2.0 * PI * radius_inc.read(), &[radius.idx()]);

        let pos_inc = pos.clone();
        let radius_inc = radius.clone();
        let left_bound = circle.memo(
            move || {
                let (x, y) = pos_inc.read();
                (x - radius_inc.read(), y)
            },
            &[pos.idx(), radius.idx()],
        );

        assert!(radius.is_clean());
        assert!(area.is_clean());
        assert!(circum.is_clean());
        assert!(left_bound.is_clean());

        assert_eq!(area.query(), PI);
        assert_eq!(circum.query(), 2.0 * PI);
        assert_eq!(left_bound.query(), (-1.0, 0.0));

        assert!(radius.is_clean());
        assert!(area.is_clean());
        assert!(circum.is_clean());
        assert!(left_bound.is_clean());

        assert_eq!(radius.write(2.0), 1.0);

        assert!(radius.is_dirty());
        assert!(area.is_dirty());
        assert!(circum.is_dirty());
        assert!(left_bound.is_dirty());

        assert_eq!(area.query(), 4.0 * PI);
        assert_eq!(circum.query(), 4.0 * PI);
        assert_eq!(left_bound.query(), (-2.0, 0.0));
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
                thunk_table.push(dcg.thunk(
                    move || x_inc.read() * y_inc.read(),
                    &[nums[i].idx(), nums[j].idx()],
                ));
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
