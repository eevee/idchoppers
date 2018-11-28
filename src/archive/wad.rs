use std;
use std::borrow::Cow;

use super::{Archive, Namespace};
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

pub struct MapBlock {
    pub format: MapFormat,
    pub name: MapName,  // TODO what are the rules in zdoom?  can you really use any map name?
    pub marker: Entry,

    pub things: Option<Entry>,
    pub linedefs: Option<Entry>,
    pub sidedefs: Option<Entry>,
    pub vertexes: Option<Entry>,
    pub segs: Option<Entry>,
    pub ssectors: Option<Entry>,
    pub nodes: Option<Entry>,
    pub sectors: Option<Entry>,
    pub reject: Option<Entry>,
    pub blockmap: Option<Entry>,
    pub behavior: Option<Entry>,
    pub textmap: Option<Entry>,
    // TODO unknown lumps (udmf only)
    // TODO endmap
}
pub struct Entry {
    pub name: String,
    pub data: Vec<u8>,
    pub namespace: Namespace,
}
pub enum Item {
    Map(MapBlock),
    Entry(Entry),
}

/// High-level interface to a WAD archive.  Does its best to prevent you from producing an invalid
/// WAD.  This is probably what you want.
#[allow(dead_code)]
pub struct WADArchive {
    /// WAD type for this archive
    pub wadtype: WADType,

    contents: Vec<Item>,
}
impl Archive for WADArchive {
}

impl WADArchive {
    pub fn from_bare(wad: &BareWAD) -> Self {
        let make_entry = |i: usize| {
            let wad_entry = &wad.directory[i];
            Entry {
                name: wad_entry.name.to_owned(),
                data: wad_entry.extract_slice(wad.buffer).to_owned(),
                namespace: Namespace::Map,
            }
        };

        let mut namespace = Namespace::Unknown;
        let mut contents = Vec::new();
        for item in wad.iter() {
            match item {
                WADItem::Map(map_range) => {
                    contents.push(Item::Map(MapBlock {
                        format: map_range.format,
                        name: map_range.name,
                        marker: make_entry(map_range.marker_index),

                        // TODO better handling for these unwraps
                        things: map_range.things_index.map(make_entry),
                        linedefs: map_range.linedefs_index.map(make_entry),
                        sidedefs: map_range.sidedefs_index.map(make_entry),
                        vertexes: map_range.vertexes_index.map(make_entry),
                        segs: map_range.segs_index.map(make_entry),
                        ssectors: map_range.ssectors_index.map(make_entry),
                        nodes: map_range.nodes_index.map(make_entry),
                        sectors: map_range.sectors_index.map(make_entry),
                        reject: map_range.reject_index.map(make_entry),
                        blockmap: map_range.blockmap_index.map(make_entry),
                        behavior: map_range.behavior_index.map(make_entry),
                        textmap: map_range.textmap_index.map(make_entry),
                    }));
                }
                WADItem::StartMarker(entry) => {
                }
                WADItem::EndMarker(entry) => {
                }
                WADItem::Entry(entry) => {
                    contents.push(Item::Entry(Entry {
                        name: entry.name.to_owned(),
                        data: entry.extract_slice(wad.buffer).to_owned(),
                        namespace: Namespace::Unknown,
                    }));
                }
            }
        }

        WADArchive {
            wadtype: wad.header.identification,
            contents: contents,
        }
    }

    pub fn first_entry(&self, name: &str) -> Option<&Entry> {
        for item in &self.contents {
            if let &Item::Entry(ref entry) = item {
                if entry.name == name {
                    return Some(entry);
                }
            }
        }
        None
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

impl<'a> ::std::iter::IntoIterator for &'a WADArchive {
    type Item = &'a Item;
    type IntoIter = <&'a Vec<Item> as ::std::iter::IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.contents.iter()
    }
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
        WADArchive::from_bare(self)
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

    pub fn iter_entries_between<'a>(&'a self, begin_marker: &'a str, end_marker: &'a str) -> BareWADBetweenIterator<'a> {
        BareWADBetweenIterator {
            archive: self,
            next_index: 0,
            begin_marker,
            end_marker,
            between_markers: false,
        }
    }

    pub fn iter_maps(&self) -> impl Iterator<Item=WADMapEntryBlock> + '_ {
        self.iter().filter_map(|item|
            if let WADItem::Map(map_block) = item {
                Some(map_block)
            }
            else {
                None
            }
        )
    }

    pub fn iter(&self) -> WADIterator {
        WADIterator {
            archive: self,
            entry_iter: self.directory.iter().enumerate().peekable(),
        }
    }
}
pub struct BareWADBetweenIterator<'wad> {
    archive: &'wad BareWAD<'wad>,
    next_index: usize,
    begin_marker: &'wad str,
    end_marker: &'wad str,
    between_markers: bool,
}
impl<'w> Iterator for BareWADBetweenIterator<'w> {
    type Item = WADEntry<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.next_index >= self.archive.directory.len() {
                return None;
            }

            let entry = &self.archive.directory[self.next_index];
            self.next_index += 1;

            if self.between_markers && entry.name == self.end_marker {
                self.between_markers = false;
            }
            else if ! self.between_markers && entry.name == self.begin_marker {
                self.between_markers = true;
            }
            else if self.between_markers {
                return Some(WADEntry {
                    name: Cow::from(entry.name),
                    data: Cow::from(entry.extract_slice(self.archive.buffer)),
                })
            }
        }
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


//#[derive(Debug)]
pub enum WADItem<'a> {
    Map(WADMapEntryBlock),
    StartMarker(&'a BareWADDirectoryEntry<'a>),
    EndMarker(&'a BareWADDirectoryEntry<'a>),
    Entry(&'a BareWADDirectoryEntry<'a>),
}

pub struct WADIterator<'a> {
    archive: &'a BareWAD<'a>,
    entry_iter: std::iter::Peekable<std::iter::Enumerate<std::slice::Iter<'a, BareWADDirectoryEntry<'a>>>>,
}

impl<'a> Iterator for WADIterator<'a> {
    type Item = WADItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (i, entry) = self.entry_iter.next()?;
        let map_name;
        if let Ok((_, found_map_name)) = vanilla_map_name(entry.name.as_bytes()) {
            map_name = found_map_name;
        }
        // TODO this seems hokey?  it's tied specifically to doom stuff, so it shouldn't be
        // generic, so it should also know what these things /mean/
        else if entry.name.ends_with("_START") {
            return Some(WADItem::StartMarker(entry));
        }
        else if entry.name.ends_with("_END") {
            return Some(WADItem::EndMarker(entry));
        }
        else {
            return Some(WADItem::Entry(entry));
        }

        // The rest of this is assembling the map lumps!
        let mut range = WADMapEntryBlock{
            format: MapFormat::Doom,
            name: map_name,
            marker_index: i,
            last_index: i,

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
        Some(WADItem::Map(range))
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
    // TODO unknown lumps (udmf only)
    // TODO endmap
}

