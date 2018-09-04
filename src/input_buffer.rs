use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use memmap::{Mmap, MmapOptions};

use errors::Result;

/// An in-memory byte buffer that can be used as a slice.
///
/// This can be used to read input files from the command line, or stdin,
/// indistinctly.  The `bytes()` method will give back the byte slice after
/// the `InputBuffer` is created.
pub enum InputBuffer {
    Stdin(Vec<u8>),
    File(Mmap),
}

impl InputBuffer {
    /// Creates an `InputBuffer` and consumes all of `stdin`
    ///
    /// This will allocate enough memory for `stdin`'s contents.
    /// Afterwards, the contents can be obtained with the `bytes()`
    /// method.
    pub fn new_from_stdin() -> Result<InputBuffer> {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;

        Ok(InputBuffer::Stdin(buf))
    }

    /// Creates an `InputBuffer` by memory-mapping a file
    ///
    /// This will map the specified file into read-only memory.
    /// Afterwards, the contents can be obtained with the `bytes()`
    /// method.
    pub fn new_from_file<P>(path: P) -> Result<InputBuffer>
    where
        P: AsRef<Path>,
    {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(InputBuffer::File(mmap))
    }

    /// Gets the stored buffer of bytes
    pub fn bytes(&self) -> &[u8] {
        match *self {
            InputBuffer::Stdin(ref v) => &*v,
            InputBuffer::File(ref m) => &*m,
        }
    }
}
