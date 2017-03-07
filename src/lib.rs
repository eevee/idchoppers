#[macro_use]
extern crate nom;

use std::str;

use nom::{IResult, Needed, le_u32};

#[derive(Debug)]
pub enum WADType {
    IWAD,
    PWAD,
}

// FIXME think about privacy here; i think it makes sense to expose all these
// details to anyone who's interested, but i guess accessors would be good so
// you can't totally fuck up an archive
pub struct WADHeader {
    pub identification: WADType,
    pub numlumps: u32,
    pub infotableofs: u32,
}


named!(iwad_tag(&[u8]) -> WADType, do_parse!(tag!(b"IWAD") >> (WADType::IWAD)));
named!(pwad_tag(&[u8]) -> WADType, do_parse!(tag!(b"PWAD") >> (WADType::PWAD)));

named!(wad_header<&[u8], WADHeader>, do_parse!(
    identification: alt!(iwad_tag | pwad_tag) >>
    numlumps: le_u32 >>
    infotableofs: le_u32 >>
    (WADHeader{ identification: identification, numlumps: numlumps, infotableofs: infotableofs })
));


pub struct WADDirectoryEntry<'a> {
    pub filepos: u32,
    pub size: u32,
    pub name: &'a str,
}

named!(wad_entry(&[u8]) -> WADDirectoryEntry, do_parse!(
    filepos: le_u32 >>
    size: le_u32 >>
    name: take!(8) >>
    // FIXME the name is ascii, not utf-8, and is zero-padded
    (WADDirectoryEntry{ filepos: filepos, size: size, name: std::str::from_utf8(name).unwrap() })
));

fn wad_directory<'a>(buf: &'a [u8], header: &WADHeader) -> IResult<&'a [u8], Vec<WADDirectoryEntry<'a>>> {
    let lumpct = header.numlumps as usize;
    let offset = header.infotableofs as usize;
    // TODO can i unhardcode the size of a wad entry here?
    let tablelen = lumpct * 16;
    if buf.len() < offset + tablelen {
        return IResult::Incomplete(Needed::Size(tablelen));
    }

    let mut ret = Vec::with_capacity(lumpct);
    let mut parse_from = &buf[offset..];
    for _ in 0..lumpct {
        match wad_entry(parse_from) {
            IResult::Done(leftovers, entry) => {
                parse_from = leftovers;
                ret.push(entry);
            }
            // TODO this seems unnecessarily verbose
            IResult::Incomplete(needed) => {
                return IResult::Incomplete(needed);
            }
            IResult::Error(err) => {
                return IResult::Error(err);
            }
        }
    }
    return IResult::Done(parse_from, ret);
}


pub struct WADArchive<'a> {
    pub buffer: &'a [u8],
    pub header: WADHeader,
    pub directory: Vec<WADDirectoryEntry<'a>>,
}


// FIXME use Result, but, figure out how to get an actual error out of here
// FIXME actually this fairly simple format is a good place to start thinking about how to return errors in general; like, do i want custom errors for tags?  etc
pub fn parse_wad(buf: &[u8]) -> Option<WADArchive> {
    match wad_header(buf) {
        IResult::Done(_leftovers, header) => {
            if let IResult::Done(_leftovers, entries) = wad_directory(buf, &header) {
                return Some(WADArchive{ buffer: buf, header: header, directory: entries });
            }
            else {
                return None;
            }
        }
        IResult::Incomplete(needed) => {
            println!("early termination what {:?}", needed);
            return None;
        }
        IResult::Error(err) => {
            println!("boom!  {:?}", err);
            return None;
        }
    }
}





#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
