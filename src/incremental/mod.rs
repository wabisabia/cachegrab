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
}
