extern crate idchoppers;

use std::io::{self, Read};

use idchoppers::parse_wad;

fn main() {
    let mut buf = Vec::new();
    io::stdin().read_to_end(&mut buf);

    if let Some(wad) = parse_wad(buf.as_slice()) {
        println!("found {:?}, {:?}, {:?}", wad.header.identification, wad.header.numlumps, wad.header.infotableofs);
        for entry in wad.directory.iter() {
            println!("{:8x}  {:8}  {}", entry.filepos, entry.size, entry.name);
        }
    }
}
