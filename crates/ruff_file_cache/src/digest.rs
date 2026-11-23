//! BLAKE3 digests: of file contents, and of the values cache keys are made of.

use std::fmt;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

/// A BLAKE3 digest.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Digest([u8; 32]);

impl Digest {
    /// Hashes `bytes`.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    /// Hashes `value` by its [`Hash`] implementation.
    ///
    /// `Hash` is injective for the values keys are made of: strings end with a byte UTF-8 never
    /// contains, slices start with their length, and enums with their variant, so `("ab", "c")`
    /// and `("a", "bc")` hash different bytes. It isn't stable across Rust versions or platforms,
    /// which is fine for keys that also hold the [`BuildId`](crate::BuildId) of the binary that
    /// computed them.
    pub fn of(value: &impl Hash) -> Self {
        let mut hasher = DigestHasher(blake3::Hasher::new());
        value.hash(&mut hasher);
        Self(*hasher.0.finalize().as_bytes())
    }
}

impl AsRef<[u8]> for Digest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

/// Feeds a [`Hash`] implementation into BLAKE3.
struct DigestHasher(blake3::Hasher);

impl Hasher for DigestHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    /// Unused: a digest is read with [`blake3::Hasher::finalize`].
    fn finish(&self) -> u64 {
        let bytes = self.0.finalize();
        let mut first = [0; 8];
        first.copy_from_slice(&bytes.as_bytes()[..8]);
        u64::from_le_bytes(first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keys are made of several fields: no two different tuples share a digest, however their
    /// bytes line up.
    #[test]
    fn different_values_have_different_digests() {
        #[expect(clippy::needless_pass_by_value, reason = "quickcheck passes values")]
        fn property(a: (String, Vec<u8>, Option<u32>), b: (String, Vec<u8>, Option<u32>)) -> bool {
            (Digest::of(&a) == Digest::of(&b)) == (a == b)
        }
        quickcheck::QuickCheck::new()
            .tests(1000)
            .quickcheck(property as fn(_, _) -> bool);
        assert_ne!(Digest::of(&("ab", "c")), Digest::of(&("a", "bc")));
    }

    #[test]
    fn digests_display_as_hex() {
        let digest = Digest::of_bytes(b"");
        assert_eq!(
            digest.to_string(),
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }
}
