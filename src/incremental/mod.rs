//! Making types [`Incremental`].

use std::rc::Rc;

use crate::Node;

/// Allows a type to adopt incremental "clean" and "dirty" semantics in a dependency tracking
/// environment.
///
/// Fundamentally an [`Incremental`] type `I` produces [`Output`](Incremental::Output)s that change
/// over time due to changes in `I`'s dependencies.
///
/// `I` is either:
///
/// - **clean**: no dependencies have changed, reading `I` yields the same
/// [`Output`](Incremental::Output) as last read.
/// - **dirty**: a dependency has changed, reading `I` computes a different
/// [`Output`](Incremental::Output).
///
/// `I`'s state can be interrogated using [`is_dirty`](Incremental::is_dirty) and
/// [`is_clean`](Incremental::is_clean).
///
/// [`Incremental`] provides separate methods for producing [`Output`](Incremental::Output)s:
///
/// - [`read`](Incremental::read): cleans `I`'s [`Node`]s and returns `I`'s most up-to-date value.
/// - [`latest`](Incremental::read): returns `I`s most up-to-date value without affecting the
/// DCG.
pub trait Incremental {
    /// The type returned when reading or reading a node.
    type Output;

    /// Computes the [`Incremental`]'s most recent value, cleans its nodes and returns the computed
    /// value.
    ///
    /// The default implementation stores the result of [`latest`](Incremental::latest), cleans the
    /// [`Incremental`]'s [`Node`]'s and returns the result.
    ///
    /// The default implementation should not be overriden unless non-standard behaviour is
    /// required.
    fn read(&self) -> Self::Output {
        let value = self.latest();
        if self.is_dirty() {
            for node in self.nodes() {
                if node.is_dirty() {
                    node.clean();
                }
            }
        }
        value
    }

    /// Returns the [`Incremental`]'s most up-to-date value.
    fn latest(&self) -> Self::Output;

    /// Returns `true` if the node is dirty.
    fn is_dirty(&self) -> bool;

    /// Returns `true` if the node is clean.
    ///
    /// The default implementation requires that [`is_dirty`](Incremental::is_dirty) is implemented
    /// and negates the result.
    fn is_clean(&self) -> bool {
        !self.is_dirty()
    }

    /// Returns a vector containing references to the [`Incremental`]'s DCG [`Node`]s.
    fn nodes(&self) -> Vec<&Node>;
}

impl Incremental for () {
    type Output = ();

    fn latest(&self) -> Self::Output {}

    fn is_dirty(&self) -> bool {
        false
    }

    fn nodes(&self) -> Vec<&Node> {
        Vec::new()
    }
}

impl<A, T> Incremental for (A,)
where
    A: Incremental<Output = T>,
{
    type Output = (T,);

    fn latest(&self) -> Self::Output {
        (self.0.read(),)
    }

    fn is_dirty(&self) -> bool {
        self.0.is_dirty()
    }

    fn nodes(&self) -> Vec<&Node> {
        self.0.nodes()
    }
}

impl<A, B, AT, BT> Incremental for (A, B)
where
    A: Incremental<Output = AT>,
    B: Incremental<Output = BT>,
{
    type Output = (AT, BT);

    fn latest(&self) -> Self::Output {
        (self.0.read(), self.1.read())
    }

    fn is_dirty(&self) -> bool {
        self.0.is_dirty() || self.1.is_dirty()
    }

    fn nodes(&self) -> Vec<&Node> {
        vec![self.0.nodes(), self.1.nodes()]
            .into_iter()
            .flatten()
            .collect()
    }
}

impl<T, O> Incremental for Rc<T>
where
    T: Incremental<Output = O>,
{
    type Output = O;

    fn latest(&self) -> Self::Output {
        self.as_ref().read()
    }

    fn is_dirty(&self) -> bool {
        self.as_ref().is_dirty()
    }

    fn nodes(&self) -> Vec<&Node> {
        self.as_ref().nodes()
    }
}

impl<A, T> Incremental for Vec<A>
where
    A: Incremental<Output = T>,
{
    type Output = Vec<T>;

    fn latest(&self) -> Self::Output {
        self.iter().map(Incremental::read).collect()
    }

    fn is_dirty(&self) -> bool {
        self.iter().any(Incremental::is_dirty)
    }

    fn nodes(&self) -> Vec<&Node> {
        self.iter().flat_map(Incremental::nodes).collect()
    }
}
