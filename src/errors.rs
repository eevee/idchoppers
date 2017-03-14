use std::io;

error_chain! {
    foreign_links {
        Io(io::Error);
    }

    errors {
        ParseError {
            description("nonspecific parse error")
            display("nonspecific parse error")
        }
        TruncatedData(whence: &'static str) {
            description("unexpected end of input")
            display("unexpected end of input while parsing {}", whence)
        }
        InvalidMagic {
            description("invalid magic")
            display("invalid magic")
        }
        MissingMapLump(lump: &'static str) {
            description("missing required map lump")
            display("missing required map lump: {}", lump)
        }
    }
}
