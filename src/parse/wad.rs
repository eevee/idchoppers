use std::u8;

use nom::{self, IResult, Needed, le_u32};

use super::util::fixed_length_ascii;
use ::errors::{Result, nom_to_result};
use ::archive::wad::{BareWAD, BareWADHeader, BareWADDirectoryEntry, WADType};


named!(iwad_tag<WADType>, value!(WADType::IWAD, tag!(b"IWAD")));
named!(pwad_tag<WADType>, value!(WADType::PWAD, tag!(b"PWAD")));

named!(wad_header<BareWADHeader>, do_parse!(
    identification: return_error!(
        nom::ErrorKind::Custom(1),
        alt!(iwad_tag | pwad_tag)) >>
    numlumps: le_u32 >>
    infotableofs: le_u32 >>
    (BareWADHeader{ identification, numlumps, infotableofs })
));


named!(wad_entry<BareWADDirectoryEntry>, dbg_dmp!(do_parse!(
    filepos: le_u32 >>
    size: le_u32 >>
    name: apply!(fixed_length_ascii, 8) >>
    (BareWADDirectoryEntry{ filepos, size, name })
)));

fn wad_directory<'a>(buf: &'a [u8], header: &BareWADHeader) -> IResult<&'a [u8], Vec<BareWADDirectoryEntry<'a>>> {
    let lumpct = header.numlumps as usize;
    let offset = header.infotableofs as usize;
    // TODO can i unhardcode the size of a wad entry here?
    let tablelen = lumpct * 16;
    if buf.len() < offset + tablelen {
        return Err(nom::Err::Incomplete(Needed::Size(tablelen)));
    }

    let mut ret = Vec::with_capacity(lumpct);
    let mut parse_from = &buf[offset..];
    for _ in 0..lumpct {
        let (leftovers, entry) = try_parse!(parse_from, wad_entry);
        ret.push(entry);
        parse_from = leftovers;
    }
    Ok((parse_from, ret))
}


// TODO problems to scan a wad map for:
// - missing a required lump
// TODO curiosities to scan a wad map for:
// - content in ENDMAP
// - UDMF along with the old-style text maps


// FIXME use Result, but, figure out how to get an actual error out of here
// FIXME actually this fairly simple format is a good place to start thinking about how to return errors in general; like, do i want custom errors for tags?  etc
pub fn parse_wad(buf: &[u8]) -> Result<BareWAD> {
    // FIXME ambiguous whether the error was from parsing the header or the entries
    let header = nom_to_result("wad header", buf, wad_header(buf))?;
    // TODO buf is not actually the right place here
    let entries = nom_to_result("wad index", buf, wad_directory(buf, &header))?;
    Ok(BareWAD{ buffer: buf, header, directory: entries })
}
