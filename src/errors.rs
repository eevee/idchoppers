use std::io;

use nom;
use nom::IResult;
use nom::{generate_colors, prepare_errors, print_codes, print_offsets};

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
        NegativeOffset(lump: &'static str, index: usize, value: isize) {
            description("nonsensical negative offset")
            display("found nonsensical negative offset {} in position {} while reading {}", value, index, lump)
        }
    }
}


pub(crate) fn nom_to_result<O>(whence: &'static str, input: &[u8], result: IResult<&[u8], O>) -> Result<O> {
    /*
    if let IResult::Done(_, value) = result {
        return Ok(value);
    }
    else {
        display_error(input, result);
        bail!(ErrorKind::ParseError);
    }
    */
    match result {
        IResult::Done(_, value) => {
            Ok(value)
        }
        IResult::Incomplete(_) => {
            bail!(ErrorKind::TruncatedData(whence));
        }
        IResult::Error(err) => {
            println!("(error: {:?})", err);
            let error_kind = match err {
                nom::Err::Code(ref k) | nom::Err::Node(ref k, _) | nom::Err::Position(ref k, _) | nom::Err::NodePosition(ref k, _, _) => k
            };
            println!("(error code: {:?})", error_kind);
            if let nom::Err::Position(_, pos) = err {
                println!("input size is {}, error happened at {}", input.len(), (&pos[0] as *const u8 as usize) - (&input[0] as *const u8 as usize));
            }
            match *error_kind {
                nom::ErrorKind::Custom(1) => bail!(ErrorKind::InvalidMagic),
                _ => bail!(ErrorKind::ParseError),
            }
        }
    }
}



use std::collections::HashMap;
pub fn display_error<O>(input: &[u8], res: IResult<&[u8], O>) {
    let mut h: HashMap<u32, &str> = HashMap::new();

    if let Some(v) = prepare_errors(input, res) {
        let colors = generate_colors(&v);
        println!("parsers: {}", print_codes(colors, h));
        println!("{}", print_offsets(input, 0, &v));
    }
}
