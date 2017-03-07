#[macro_use]
extern crate nom;

use std::io::{self, Read};
use std::str;

use nom::{IResult, le_u32};

#[derive(Debug)]
pub enum WADType {
    IWAD,
    PWAD,
}

pub struct WADHeader {
    identification: WADType,
    numlumps: u32,
    infotableofs: u32,
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
    filepos: u32,
    size: u32,
    name: &'a str,
}

named!(wad_entry(&[u8]) -> WADDirectoryEntry, do_parse!(
    filepos: le_u32 >>
    size: le_u32 >>
    name: take!(8) >>
    // FIXME the name is ascii, not utf-8, and is zero-padded
    (WADDirectoryEntry{ filepos: filepos, size: size, name: std::str::from_utf8(name).unwrap() })
));

fn wad_directory<'a>(buf: &'a [u8], header: &WADHeader) -> Vec<WADDirectoryEntry<'a>> {
    let lumpct = header.numlumps as usize;
    let offset = header.infotableofs as usize;
    // FIXME check that offset is within the buffer AND that it's long enough for all the entries
    let mut ret = Vec::with_capacity(lumpct);
    let mut parse_from = &buf[offset..];
    for i in 0..lumpct {
        match wad_entry(parse_from) {
            IResult::Done(leftovers, entry) => {
                parse_from = leftovers;
                ret.push(entry);
            }
            // FIXME how do i return an error from here???
            IResult::Incomplete(needed) => {
                println!("early termination what {:?}", needed);
                break;
            }
            IResult::Error(err) => {
                println!("boom!  {:?}", err);
                break;
            }
        }
    }
    return ret;
}


pub fn parse_wad() {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer);
    println!("buffer len is {:?}", buffer.len());

    match wad_header(buffer.as_slice()) {
        IResult::Done(_leftovers, header) => {
            println!("found {:?}, {:?}, {:?}", header.identification, header.numlumps, header.infotableofs);
            let entries = wad_directory(buffer.as_slice(), &header);
            for entry in entries.iter() {
                println!("{:8x}  {:8}  {}", entry.filepos, entry.size, entry.name);
            }
        }
        IResult::Incomplete(needed) => {
            println!("early termination what {:?}", needed);
        }
        IResult::Error(err) => {
            println!("boom!  {:?}", err);
        }
    }
}





#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
