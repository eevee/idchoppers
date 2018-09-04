use std::str;

use nom::{self, IResult, Needed};


pub fn fixed_length_ascii(input: &[u8], len: usize) -> IResult<&[u8], &str> {
    if input.len() < len {
        return Err(nom::Err::Incomplete(Needed::Size(len)));
    }

    for i in 0..len {
        match input[i] {
            0 => {
                // This is the end
                let s = unsafe { str::from_utf8_unchecked(&input[..i]) };
                return Ok((&input[len..], s));
            }
            32 ... 126 => {
                // OK
            }
            _ => {
                // Totally bogus character
                return Err(nom::Err::Error(nom::Context::Code(&input[i..], nom::ErrorKind::Custom(0))));
            }
        }
    }

    Ok((&input[len..], unsafe { str::from_utf8_unchecked(&input[..len]) }))
}
