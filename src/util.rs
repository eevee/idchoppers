use std::hash::{Hash, Hasher};


/// Key wrapper type to convince HashMap and friends to hash on a reference's identity (i.e.
/// address), not on the underlying value
pub struct RefKey<'a, T: 'a>(pub &'a T);

impl<'a, T: 'a> PartialEq for RefKey<'a, T> {
    fn eq(&self, other: &RefKey<T>) -> bool {
        (self.0 as *const _) == (other.0 as *const _)
    }
}

impl<'a, T: 'a> Eq for RefKey<'a, T> {}

impl<'a, T: 'a> Hash for RefKey<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0 as *const T).hash(state)
    }
}
