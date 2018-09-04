use std::str;
use std::u8;

use nom::{le_i16, le_i32};

use super::util::fixed_length_ascii;
use ::errors::{ErrorKind, Result, nom_to_result};


pub struct TEXTURExEntry<'name> {
    pub name: &'name str,
    pub width: i16,
    pub height: i16,
}

named!(texturex_lump_header<Vec<i32>>, do_parse!(
    numtextures: le_i32 >>
    offsets: count!(le_i32, numtextures as usize) >>
    (offsets)
));

named!(texturex_lump_entry<TEXTURExEntry>, do_parse!(
    name: apply!(fixed_length_ascii, 8) >>
    le_i32 >>  // "masked", unused
    // TODO these should be positive
    width: le_i16 >>
    height: le_i16 >>
    le_i32 >>  // "columndirectory", unused
    patchcount: le_i16 >>
    // TODO patches
    (TEXTURExEntry{
        name,
        width,
        height,
    })
));

pub fn parse_texturex_names(buf: &[u8]) -> Result<Vec<TEXTURExEntry>> {
    let offsets = nom_to_result("TEXTUREx header", buf, texturex_lump_header(buf))?;
    let mut ret = Vec::with_capacity(offsets.len());
    for (i, &offset) in offsets.iter().enumerate() {
        if offset < 0 {
            bail!(ErrorKind::NegativeOffset("TEXTUREx", i, offset as isize));
        }
        // TODO check for too large offset too, instead of Incomplete
        ret.push(nom_to_result("TEXTUREx", buf, texturex_lump_entry(&buf[(offset as usize)..]))?);
    }
    Ok(ret)
}
