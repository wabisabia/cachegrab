use std::rc::Rc;

use crate::Node;

/// Allows a type to adopt incremental "clean" and "dirty" semantics in a dependency tracking environment.
///
/// Fundamentally an [`Incremental`] type `I` produces [`Output`](Incremental::Output)s that change
/// over time due to changes in `I`'s dependencies.
///
/// `I` is either:
///
/// - **clean**: no dependencies have changed, reading `I` yields the same [`Output`](Incremental::Output) as last read.
/// - **dirty**: a dependency has changed, reading `I` computes a different [`Output`](Incremental::Output).
///
/// `I`'s state can be interrogated using [`is_dirty`](Incremental::is_dirty) and [`is_clean`](Incremental::is_clean).
///
/// [`Incremental`] provides separate methods for producing [`Output`](Incremental::Output)s:
///
/// - [`read`](Incremental::read): simply returns the most up-to-date [`Output`](Incremental::Output).
/// - [`read`](Incremental::read): does the same thing as [`read`](Incremental::read) but also cleans
/// `I`'s dependencies once the [`Output`](Incremental::Output) has been retrieved.
pub trait Incremental {
    /// The type returned when reading or reading a node.
    type Output;

    /// Cleans a node and returns its most up-to-date value.
    fn read(&self) -> Self::Output;

    /// Returns `true` if the node is dirty.
    fn is_dirty(&self) -> bool;

    /// Returns `true` if the node is clean.
    ///
    /// The default implementation requires that [`Incremental::is_dirty`] is implemented and negates the result.
    fn is_clean(&self) -> bool {
        !self.is_dirty()
    }

    /// Returns a vector containing references to the [`Incremental`]'s [`Dcg`] [`Node`]s.
    ///
    /// The default implementation returns an empty [`Vec`]
    fn nodes(&self) -> Vec<&Node>;
}

impl Incremental for () {
    type Output = ();

    fn read(&self) -> Self::Output {}

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

    fn read(&self) -> Self::Output {
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

    fn read(&self) -> Self::Output {
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

    fn read(&self) -> Self::Output {
        self.as_ref().read()
    }

    fn is_dirty(&self) -> bool {
        self.as_ref().is_dirty()
    }

    fn nodes(&self) -> Vec<&Node> {
        self.as_ref().nodes()
    }
}
