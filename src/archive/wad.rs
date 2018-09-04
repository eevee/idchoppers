use std;
use std::borrow::Cow;

use super::Archive;
use ::map::{MapFormat, MapName};
use ::parse::vanilla_map_name;


// TODO does this distinction apply to other kinds of archives?
/// Type of the WAD.
#[derive(Copy, Clone, Debug)]
pub enum WADType {
    /// full standalone game
    IWAD,
    /// patch wad, a small mod
    PWAD,
}

// TODO this should be able to contain an arbitrarily-typed entry, right?  but when does the type
// detection happen?
pub struct WADEntry<'a> {
    pub name: Cow<'a, str>,
    pub data: Cow<'a, [u8]>,
}

/// High-level interface to a WAD archive.  Does its best to prevent you from producing an invalid
/// WAD.  This is probably what you want.
#[allow(dead_code)]
pub struct WADArchive<'a> {
    // TODO it would be nice if we could take ownership of the slice somehow, but i don't know how
    // to do that really.  i also don't know how to tell rust that the entry slices are owned by
    // this buffer?
    buffer: &'a [u8],

    /// WAD type for this archive
    pub wadtype: WADType,

    // Pairs of (name, data)
    entries: Vec<WADEntry<'a>>,
}
impl<'a> Archive for WADArchive<'a> {
}

impl<'a> WADArchive<'a> {
    pub fn first_entry(&self, name: &str) -> Option<&WADEntry> {
        self.entries.iter()
        .find(|entry| entry.name == name)
    }
    
    // TODO:
    pub fn iter_entry(&self, name: &str) {
        unimplemented!()
    }
    
    pub fn iter_between(&self, _start: &str, _end: &str) {
        unimplemented!()
    }
    
    pub fn iter_maps(&self) {
        unimplemented!()
    }
    
    pub fn iter_flats(&self) {
        unimplemented!()
    }
    // TODO interesting things:
    // - find suspected markers that contain data
}

// TODO:
// - get entry by name (REMEMBER: NAMES CAN REPEAT)
// - detect entry type (using name as a hint as well)
// - how the hell do i iterate over /either/ an entry or a map?  maybe that's not even a useful thing to do
// things to check:
// - error: lump outside the bounds of the wad
// - warning: lump overlaps the directory
// - warning: lumps overlap
// - warning: lumps not in the same order physically as in the listing
// - interesting: lumps have gaps
/// Low-level interface to a parsed WAD.  This is really only useful for, uh, shenanigans.
pub struct BareWAD<'n> {
    pub buffer: &'n [u8],
    pub header: BareWADHeader,
    pub directory: Vec<BareWADDirectoryEntry<'n>>,
}

// TODO expand these into separate types, probably, so the severity can be an associated value...
// either that or use a method with a big ol match block Ugh
pub enum Diagnostic {
    InvalidLumpBounds(usize, usize),
    LumpOverlapsIndex,
    LumpsOverlap,
    LumpsOutOfOrder,
    UnusedSpace,
}

impl<'n> BareWAD<'n> {
    pub fn diagnose(&self) -> Vec<Diagnostic> {
        let ret = vec![];
        // TODO this
        ret
    }

    pub fn to_archive(&self) -> WADArchive {
        WADArchive{
            buffer: self.buffer,
            wadtype: self.header.identification,
            entries: self.directory.iter()
                .map(|bare_entry| WADEntry{
                    name: Cow::from(bare_entry.name),
                    data: Cow::from(bare_entry.extract_slice(self.buffer))
                })
                .collect(),
        }
    }

    pub fn entry_slice(&self, index: usize) -> &[u8] {
        let entry = &self.directory[index];
        let start = entry.filepos as usize;
        let end = start + entry.size as usize;
        &self.buffer[start..end]
    }

    pub fn first_entry(&self, name: &str) -> Option<&[u8]> {
        self.directory.iter()
        .find(|entry| entry.name == name)
        .map(|entry| {
            let start = entry.filepos as usize;
            let end = start + entry.size as usize;
            // TODO what should this do if the offsets are bogus?
            &self.buffer[start..end]
        })
    }
    
    pub fn iter_entries_between(&self, begin_marker: &str, end_marker: &str) -> BareWADBetweenIterator {
        // TODO: deal with unwraps somehow?
        let start = self.directory.iter()
            .position(|entry| entry.name == begin_marker)
            .unwrap();
        let end = self.directory.iter()
            .position(|entry| entry.name == end_marker)
            .unwrap();
        
        BareWADBetweenIterator {
            wad_buffer: self.buffer,
            entries: &self.directory[start + 1..end],
        }
    }
    
    pub fn iter_maps(&self) -> WADMapIterator {
        WADMapIterator{
            archive: self,
            entry_iter: self.directory.iter().enumerate().peekable()
        }
    }
}
pub struct BareWADBetweenIterator<'wad> {
    entries: &'wad [BareWADDirectoryEntry<'wad>],
    wad_buffer: &'wad [u8],
}
impl<'w> Iterator for BareWADBetweenIterator<'w> {
    type Item = WADEntry<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.split_first()
        .map(|(entry, entries)| {
            self.entries = entries;
            WADEntry {
                name: Cow::from(entry.name),
                data: Cow::from(entry.extract_slice(self.wad_buffer)),
            }
        })
    }
}

pub struct BareWADHeader {
    pub identification: WADType,
    pub numlumps: u32,
    pub infotableofs: u32,
}

#[derive(Debug)]
pub struct BareWADDirectoryEntry<'name> {
    pub filepos: u32,
    pub size: u32,
    pub name: &'name str,
}

impl<'n> BareWADDirectoryEntry<'n> {
    /// Extract the slice described by this entry from a buffer.
    pub fn extract_slice<'b>(&self, buf: &'b [u8]) -> &'b [u8] {
        let start = self.filepos as usize;
        let end = start + self.size as usize;
        &buf[start..end]
    }
}

// -----------------------------------------------------------------------------
// Map stuff

// Standard lumps and whether they're required
const MAP_LUMP_ORDER: [(&str, bool); 11] = [
    ("THINGS",	 true),
    ("LINEDEFS", true),
    ("SIDEDEFS", true),
    ("VERTEXES", true),
    ("SEGS",	 false),
    ("SSECTORS", false),
    ("NODES",	 false),
    ("SECTORS",	 true),
    ("REJECT",	 false),
    ("BLOCKMAP", false),
    ("BEHAVIOR", false),
];

#[allow(dead_code)]
pub struct WADMapIterator<'a> {
    archive: &'a BareWAD<'a>,
    entry_iter: std::iter::Peekable<std::iter::Enumerate<std::slice::Iter<'a, BareWADDirectoryEntry<'a>>>>,
}

impl<'a> Iterator for WADMapIterator<'a> {
    type Item = WADMapEntryBlock;

    fn next(&mut self) -> Option<WADMapEntryBlock> {
        let (marker_index, map_name) = loop {
            if let Some((i, entry)) = self.entry_iter.next() {
                if let Ok((_, found_map_name)) = vanilla_map_name(entry.name.as_bytes()) {
                    break (i, found_map_name);
                }
            }
            else {
                // Hit the end of the entries
                return None;
            }
        };

        let mut range = WADMapEntryBlock{
            format: MapFormat::Doom,
            name: map_name,
            marker_index,
            last_index: marker_index,

            things_index: None,
            linedefs_index: None,
            sidedefs_index: None,
            vertexes_index: None,
            segs_index: None,
            ssectors_index: None,
            nodes_index: None,
            sectors_index: None,
            reject_index: None,
            blockmap_index: None,
            behavior_index: None,
            textmap_index: None,
        };

        // Use peeking here, so that if we stumble onto the next map header, we don't consume it
        let (mut i, mut entry) = match self.entry_iter.peek() {
            Some(&next) => next,
            None => { return None; }
        };

        if entry.name == "TEXTMAP" {
            // This is a UDMF map, which has a completely different scheme: it
            // goes until an explicit ENDMAP marker
            range.format = MapFormat::UDMF;
            // TODO continue this logic
        }

        for &(lump_name, is_required) in MAP_LUMP_ORDER.iter() {
            // TODO i am pretty sure this is supposed to be case-insensitive?
            if entry.name == lump_name {
                match entry.name {
                    "THINGS" => { range.things_index = Some(i); }
                    "LINEDEFS" => { range.linedefs_index = Some(i); }
                    "SIDEDEFS" => { range.sidedefs_index = Some(i); }
                    "VERTEXES" => { range.vertexes_index = Some(i); }
                    "SEGS" => { range.segs_index = Some(i); }
                    "SSECTORS" => { range.ssectors_index = Some(i); }
                    "NODES" => { range.nodes_index = Some(i); }
                    "SECTORS" => { range.sectors_index = Some(i); }
                    "REJECT" => { range.reject_index = Some(i); }
                    "BLOCKMAP" => { range.blockmap_index = Some(i); }
                    "BEHAVIOR" => {
                        range.behavior_index = Some(i);
                        // The presence of a BEHAVIOR lump is the sole indication of Hexen format
                        range.format = MapFormat::Hexen;
                    }
                    "TEXTMAP" => { range.textmap_index = Some(i); }
                    _ => {
                        // TODO wait, what's the right thing here
                        break;
                    }
                }
                self.entry_iter.next();
                match self.entry_iter.peek() {
                    Some(&(next_i, next_entry)) => {
                        i = next_i;
                        entry = next_entry;
                    }
                    None => {
                        // FIXME this needs to check whether there are any
                        // /required/ lumps not yet seen, ugh
                        break;
                    }
                }
            }
            else if is_required {
                // FIXME return a better error: expected lump X, found Y
                // FIXME should this really stop us from iterating over any further maps?
                // TODO should we try to cleverly detect what happened here?  what DID happen here, anyway?
                // TODO maybe we should return what we have so far, and let the conversion to a real map take care of it?  but then how do we handle missing only one lump (do we grab the rest)?  what about duplicate lumps?
                // TODO same questions go for the places i used try!(), except i think i got the logic even worse there, idk.  write some tests
                return None;
            }
        }

        range.last_index = i;
        Some(range)
    }
}

#[derive(Debug)]
pub struct WADMapEntryBlock {
    pub format: MapFormat,
    pub name: MapName,  // TODO what are the rules in zdoom?  can you really use any map name?
    pub marker_index: usize,
    pub last_index: usize,

    pub things_index: Option<usize>,
    pub linedefs_index: Option<usize>,
    pub sidedefs_index: Option<usize>,
    pub vertexes_index: Option<usize>,
    pub segs_index: Option<usize>,
    pub ssectors_index: Option<usize>,
    pub nodes_index: Option<usize>,
    pub sectors_index: Option<usize>,
    pub reject_index: Option<usize>,
    pub blockmap_index: Option<usize>,
    pub behavior_index: Option<usize>,
    pub textmap_index: Option<usize>,
    // TODO endmap
}

