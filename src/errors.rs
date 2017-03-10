use nom;

error_chain! {
    errors {
        ParseError {
            description("nonspecific parse error")
            display("nonspecific parse error")
        }
        TruncatedData {
            description("oh no data")
            display("oh no data")
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
