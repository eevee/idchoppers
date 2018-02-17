extern crate byteorder;
extern crate euclid;
extern crate typed_arena;
#[macro_use]
extern crate nom;
#[macro_use]
extern crate error_chain;
extern crate svg;  // TODO temp for debugging

use std::borrow::Cow;
use std::io::Write;
use std::str::FromStr;
use std::str;
use std::u8;

use byteorder::{LittleEndian, WriteBytesExt};
use nom::{IResult, Needed, digit, le_i16, le_i32, le_u16, le_u32, le_u8};

pub mod errors;
use errors::{ErrorKind, Result, nom_to_result};
pub mod map;
pub mod universe;
pub mod shapeops;
mod vanilladoom;

// FIXME so, this whole file is kind of a mess.  i was trying to make the raw binary data available
// for inspection without needing to parse into a whole map object, and that turns out to be
// complicated?  designing a wad browsing api is also kinda hard, since...  well.
// i'm not entirely sure what i should do to remedy all of this.  a Map is fairly heavy-handed, and
// you can still do plenty of interesting stuff without one.



// TODO Doom 64?  PlayStation Doom?  others?
pub enum SourcePort {
    // TODO distinguish doom 1 and 2, since they have different sets of things?  final doom?
    Doom,
    Heretic,
    Hexen,
    Strife,
    Boom,
    // TODO boom plus skybox transfer
    // TODO try to distinguish versions of these...??
    ZDoom,
    GZDoom,

    // I don't know so much about these
    MBF,
    Legacy,
    Eternity,
    Vavoom,
}

pub enum BaseGame {
    None,  // i.e. this IS a base game, or is a TC, or whatever
    Doom1,  // TODO distinguish from ultimate doom?
    Doom2,
    TNTEvilution,
    Plutonia,
    Heretic,
    Hexen,
    Strife,
}

#[derive(Debug)]
pub enum MapName {
    ExMy(u8, u8),
    MAPxx(u8),
}

impl std::fmt::Display for MapName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            MapName::ExMy(x, y) => write!(f, "E{}M{}", x, y),
            MapName::MAPxx(x) => write!(f, "MAP{:02}", x),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MapFormat {
    Doom,
    Hexen,
    UDMF,
}



named!(exmy_map_name(&[u8]) -> MapName, do_parse!(
    tag!(b"E") >>
    e: digit >>
    tag!(b"M") >>
    m: digit >>
    eof!() >>
    (MapName::ExMy(
        // We already know e and m are digits, so this is all safe
        u8::from_str(str::from_utf8(e).unwrap()).unwrap(),
        u8::from_str(str::from_utf8(m).unwrap()).unwrap(),
    ))
));

named!(mapxx_map_name(&[u8]) -> MapName, do_parse!(
    tag!(b"MAP") >>
    xx: digit >>
    eof!() >>
    (MapName::MAPxx(
        // TODO need to enforce that xx is exactly two digits!  and also in [1, 32]
        u8::from_str(str::from_utf8(xx).unwrap()).unwrap()
    ))
));

named!(vanilla_map_name(&[u8]) -> MapName, alt!(exmy_map_name | mapxx_map_name));


#[derive(Copy, Clone, Debug)]
pub enum WADType {
    IWAD,
    PWAD,
}


// TODO do...  these
trait Archive {
    // TODO needs a feature atm -- const supports_duplicate_names: bool;
}
trait WAD {

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

    /// Type of the WAD, either an IWAD (full standalone game) or PWAD (patch wad, a small mod).
    pub wadtype: WADType,

    // Pairs of (name, data)
    entries: Vec<WADEntry<'a>>,
}
impl<'a> Archive for WADArchive<'a> {
}

impl<'a> WADArchive<'a> {
    // TODO:
    // first_entry(name)
    // iter_entry(name)
    // iter_between(_start, _end)
    // iter_maps()
    // iter_flats()
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
pub struct BareWAD<'a> {
    pub buffer: &'a [u8],
    pub header: BareWADHeader,
    pub directory: Vec<BareWADDirectoryEntry<'a>>,
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

impl<'a> BareWAD<'a> {
    pub fn diagnose(&'a self) -> Vec<Diagnostic> {
        let ret = vec![];
        // TODO this
        return ret;
    }

    pub fn to_archive(&'a self) -> WADArchive<'a> {
        let entries = self.directory.iter().map(|bare_entry| WADEntry{ name: Cow::from(bare_entry.name), data: Cow::from(bare_entry.extract_slice(self.buffer))} ).collect();
        return WADArchive{
            buffer: self.buffer,
            wadtype: self.header.identification,
            entries: entries,
        };
    }

    pub fn entry_slice(&'a self, index: usize) -> &'a [u8] {
        let entry = &self.directory[index];
        let start = entry.filepos as usize;
        let end = start + entry.size as usize;
        return &self.buffer[start..end];
    }

    pub fn first_entry(&'a self, name: &str) -> Option<&[u8]> {
        for entry in self.directory.iter() {
            if entry.name == name {
                let start = entry.filepos as usize;
                let end = start + entry.size as usize;
                // TODO what should this do if the offsets are bogus?
                return Some(&self.buffer[start..end]);
            }
        }
        return None;
    }

    pub fn iter_entries_between(&'a self, begin_marker: &'a str, end_marker: &'a str) -> BareWADBetweenIterator<'a> {
        BareWADBetweenIterator{
            bare_wad: self,
            entry_iter: self.directory.iter(),
            begin_marker: begin_marker,
            end_marker: end_marker,
            between_markers: false,
        }
    }
}
pub struct BareWADBetweenIterator<'a> {
    bare_wad: &'a BareWAD<'a>,
    entry_iter: std::slice::Iter<'a, BareWADDirectoryEntry<'a>>,
    begin_marker: &'a str,
    end_marker: &'a str,
    between_markers: bool,
}
impl<'a> Iterator for BareWADBetweenIterator<'a> {
    type Item = WADEntry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.entry_iter.next() {
            if entry.name == self.begin_marker {
                self.between_markers = true;
            }
            else if entry.name == self.end_marker {
                self.between_markers = false;
            }
            else if self.between_markers {
                return Some(WADEntry{
                    name: Cow::from(entry.name),
                    data: Cow::from(entry.extract_slice(self.bare_wad.buffer)),
                });
            }
        }
        return None;
    }
}

pub struct BareWADHeader {
    pub identification: WADType,
    pub numlumps: u32,
    pub infotableofs: u32,
}

named!(iwad_tag(&[u8]) -> WADType, do_parse!(tag!(b"IWAD") >> (WADType::IWAD)));
named!(pwad_tag(&[u8]) -> WADType, do_parse!(tag!(b"PWAD") >> (WADType::PWAD)));

named!(wad_header(&[u8]) -> BareWADHeader, do_parse!(
    identification: return_error!(
        nom::ErrorKind::Custom(1),
        alt!(iwad_tag | pwad_tag)) >>
    numlumps: le_u32 >>
    infotableofs: le_u32 >>
    (BareWADHeader{ identification: identification, numlumps: numlumps, infotableofs: infotableofs })
));


#[derive(Debug)]
pub struct BareWADDirectoryEntry<'a> {
    pub filepos: u32,
    pub size: u32,
    pub name: &'a str,
}

impl<'a> BareWADDirectoryEntry<'a> {
    /// Extract the slice described by this entry from a buffer.
    pub fn extract_slice(&'a self, buf: &'a [u8]) -> &'a [u8] {
        let start = self.filepos as usize;
        let end = start + self.size as usize;
        return &buf[start..end];
    }
}

fn fixed_length_ascii(input: &[u8], len: usize) -> IResult<&[u8], &str> {
    let mut input = input;
    if input.len() < len {
        return IResult::Incomplete(Needed::Size(len));
    }

    for i in 0..len {
        match input[i] {
            0 => {
                // This is the end
                let s = unsafe { str::from_utf8_unchecked(&input[..i]) };
                return IResult::Done(&input[len..], s);
            }
            32 ... 128 => {
                // OK
            }
            _ => {
                // Totally bogus character
                return IResult::Error(nom::Err::Code(nom::ErrorKind::Custom(0)));
            }
        }
    }

    return IResult::Done(&input[len..], unsafe { str::from_utf8_unchecked(&input[..len]) });
}

named!(wad_entry(&[u8]) -> BareWADDirectoryEntry, dbg_dmp!(do_parse!(
    filepos: le_u32 >>
    size: le_u32 >>
    name: apply!(fixed_length_ascii, 8) >>
    (BareWADDirectoryEntry{ filepos: filepos, size: size, name: name })
)));

fn wad_directory<'a>(buf: &'a [u8], header: &BareWADHeader) -> IResult<&'a [u8], Vec<BareWADDirectoryEntry<'a>>> {
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
        let (leftovers, entry) = try_parse!(parse_from, wad_entry);
        ret.push(entry);
        parse_from = leftovers;
    }
    return IResult::Done(parse_from, ret);
}


impl<'a> BareWAD<'a> {
    pub fn iter_maps(&'a self) -> WADMapIterator<'a> {
        return WADMapIterator{ archive: self, entry_iter: self.directory.iter().enumerate().peekable() };
    }
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
    let header = try!(nom_to_result("wad header", buf, wad_header(buf)));
    // TODO buf is not actually the right place here
    let entries = try!(nom_to_result("wad index", buf, wad_directory(buf, &header)));
    return Ok(BareWAD{ buffer: buf, header: header, directory: entries });
}


// -----------------------------------------------------------------------------
// Map stuff

// Standard lumps and whether they're required
const MAP_LUMP_ORDER: [(&'static str, bool); 11] = [
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
        let (marker_index, map_name);
        loop {
            if let Some((i, entry)) = self.entry_iter.next() {
                if let IResult::Done(_, found_map_name) = vanilla_map_name(entry.name.as_bytes()) {
                    marker_index = i;
                    map_name = found_map_name;
                    break;
                }
            }
            else {
                // Hit the end of the entries
                return None;
            }
        }

        let mut range = WADMapEntryBlock{
            format: MapFormat::Doom,
            name: map_name,
            marker_index: marker_index,
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

        let mut i;
        let mut entry;
        // Use peeking here, so that if we stumble onto the next map header, we don't consume it
        match self.entry_iter.peek() {
            Some(&(next_i, next_entry)) => {
                i = next_i;
                entry = next_entry;
            }
            None => { return None; }
        }

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
        return Some(range);
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

// TODO map parsing requires:
// - come up with some way to treat a map as a single unit in a wad (is there anything else that acts this way?)
// - parsers for:
//   - THINGS
//   - LINEDEFS
//   - SIDEDEFS
//   - SEGS
//   - SSECTORS (deferrable)
//   - NODES (deferrable)
//   - SECTORS
//   - REJECT (deferrable)
//   - BLOCKMAP (deferrable)
// - put all this in its own module/hierarchy

named!(hexen_args(&[u8]) -> [u8; 5], do_parse!(
    arg0: le_u8 >>
    arg1: le_u8 >>
    arg2: le_u8 >>
    arg3: le_u8 >>
    arg4: le_u8 >>
    ([arg0, arg1, arg2, arg3, arg4])
));

pub struct BareDoomThing {
    pub x: i16,
    pub y: i16,
    // TODO what is this
    pub angle: i16,
    pub doomednum: i16,
    // NOTE: boom added two flags, and mbf one more, so this is a decent signal for targeting those (but not 100%)
    pub flags: u16,
}

impl BareDoomThing {
    pub fn write_to(&self, writer: &mut Write) -> Result<()> {
        try!(writer.write_i16::<LittleEndian>(self.x));
        try!(writer.write_i16::<LittleEndian>(self.y));
        try!(writer.write_i16::<LittleEndian>(self.angle));
        try!(writer.write_i16::<LittleEndian>(self.doomednum));
        try!(writer.write_u16::<LittleEndian>(self.flags));
        Ok(())
    }
}

// TODO totally different in hexen
named!(doom_things_lump(&[u8]) -> Vec<BareDoomThing>, terminated!(many0!(do_parse!(
    x: le_i16 >>
    y: le_i16 >>
    angle: le_i16 >>
    doomednum: le_i16 >>
    flags: le_u16 >>
    (BareDoomThing{ x: x, y: y, angle: angle, doomednum: doomednum, flags: flags })
)), eof!()));

pub struct BareHexenThing {
    // TODO is this really signed in hexen?
    pub tid: i16,
    pub x: i16,
    pub y: i16,
    pub z: i16,
    // TODO what is this
    pub angle: i16,
    pub doomednum: i16,
    pub flags: u16,
    pub special: u8,
    pub args: [u8; 5],
}

// TODO totally different in hexen
named!(hexen_things_lump(&[u8]) -> Vec<BareHexenThing>, terminated!(many0!(do_parse!(
    tid: le_i16 >>
    x: le_i16 >>
    y: le_i16 >>
    z: le_i16 >>
    angle: le_i16 >>
    doomednum: le_i16 >>
    flags: le_u16 >>
    special: le_u8 >>
    args: hexen_args >>
    (BareHexenThing{
        tid: tid,
        x: x,
        y: y,
        z: z,
        angle: angle,
        doomednum: doomednum,
        flags: flags,
        special: special,
        args: args,
    })
)), eof!()));

pub trait BareBinaryThing {
    fn coords(&self) -> (i16, i16);
    fn doomednum(&self) -> i16;
}
impl BareBinaryThing for BareDoomThing {
    fn coords(&self) -> (i16, i16) {
        (self.x, self.y)
    }
    fn doomednum(&self) -> i16 {
        self.doomednum
    }
}
impl BareBinaryThing for BareHexenThing {
    fn coords(&self) -> (i16, i16) {
        (self.x, self.y)
    }
    fn doomednum(&self) -> i16 {
        self.doomednum
    }
}


// FIXME vertex/sidedef indices are i16 in vanilla, but extended to u16 in most source ports; note that for true vanilla, a negative index makes no sense anyway
// FIXME hexen extends this, which requires detecting hexen format
// FIXME what exactly is the higher-level structure that holds actual references to the sidedefs?
pub struct BareDoomLine {
    pub v0: i16,
    pub v1: i16,
    pub flags: i16,
    pub special: i16,
    pub sector_tag: i16,
    // NOTE: -1 to mean none
    pub front_sidedef: i16,
    pub back_sidedef: i16,
}

impl BareDoomLine {
    pub fn write_to(&self, writer: &mut Write) -> Result<()> {
        try!(writer.write_i16::<LittleEndian>(self.v0));
        try!(writer.write_i16::<LittleEndian>(self.v1));
        try!(writer.write_i16::<LittleEndian>(self.flags));
        try!(writer.write_i16::<LittleEndian>(self.special));
        try!(writer.write_i16::<LittleEndian>(self.sector_tag));
        try!(writer.write_i16::<LittleEndian>(self.front_sidedef));
        try!(writer.write_i16::<LittleEndian>(self.back_sidedef));
        Ok(())
    }
}

named!(doom_linedefs_lump(&[u8]) -> Vec<BareDoomLine>, terminated!(many0!(do_parse!(
    v0: le_i16 >>
    v1: le_i16 >>
    flags: le_i16 >>
    special: le_i16 >>
    sector_tag: le_i16 >>
    front_sidedef: le_i16 >>
    back_sidedef: le_i16 >>
    (BareDoomLine{ v0: v0, v1: v1, flags: flags, special: special, sector_tag: sector_tag, front_sidedef: front_sidedef, back_sidedef: back_sidedef })
)), eof!()));

// TODO source ports extended ids to unsigned here too
pub struct BareHexenLine {
    pub v0: i16,
    pub v1: i16,
    pub flags: i16,
    pub special: u8,
    pub args: [u8; 5],
    // NOTE: -1 to mean none
    pub front_sidedef: i16,
    pub back_sidedef: i16,
}

named!(hexen_linedefs_lump(&[u8]) -> Vec<BareHexenLine>, terminated!(many0!(do_parse!(
    v0: le_i16 >>
    v1: le_i16 >>
    flags: le_i16 >>
    special: le_u8 >>
    args: hexen_args >>
    front_sidedef: le_i16 >>
    back_sidedef: le_i16 >>
    (BareHexenLine{
        v0: v0,
        v1: v1,
        flags: flags,
        special: special,
        args: args,
        front_sidedef: front_sidedef,
        back_sidedef: back_sidedef,
    })
)), eof!()));

pub trait BareBinaryLine {
    fn vertex_indices(&self) -> (i16, i16);
    fn side_indices(&self) -> (i16, i16);
    fn has_special(&self) -> bool;
}
impl BareBinaryLine for BareDoomLine {
    fn vertex_indices(&self) -> (i16, i16) {
        (self.v0, self.v1)
    }
    fn side_indices(&self) -> (i16, i16) {
        (self.front_sidedef, self.back_sidedef)
    }
    fn has_special(&self) -> bool {
        self.special != 0
    }
}
impl BareBinaryLine for BareHexenLine {
    fn vertex_indices(&self) -> (i16, i16) {
        (self.v0, self.v1)
    }
    fn side_indices(&self) -> (i16, i16) {
        (self.front_sidedef, self.back_sidedef)
    }
    fn has_special(&self) -> bool {
        self.special != 0
    }
}

pub struct BareSide<'a> {
    pub x_offset: i16,
    pub y_offset: i16,
    pub upper_texture: &'a str,
    pub lower_texture: &'a str,
    pub middle_texture: &'a str,
    pub sector: i16,
}

impl<'a> BareSide<'a> {
    pub fn write_to(&self, writer: &mut Write) -> Result<()> {
        try!(writer.write_i16::<LittleEndian>(self.x_offset));
        try!(writer.write_i16::<LittleEndian>(self.y_offset));
        try!(writer.write(self.upper_texture.as_bytes()));
        for _ in self.upper_texture.len() .. 8 {
            try!(writer.write(&[0]));
        }
        try!(writer.write(self.lower_texture.as_bytes()));
        for _ in self.lower_texture.len() .. 8 {
            try!(writer.write(&[0]));
        }
        try!(writer.write(self.middle_texture.as_bytes()));
        for _ in self.middle_texture.len() .. 8 {
            try!(writer.write(&[0]));
        }
        try!(writer.write_i16::<LittleEndian>(self.sector));
        Ok(())
    }
}

// FIXME using many0! followed by eof! means that if the parse fails, many0! thinks that's a
// success, stops, and then hits the eof! and fails, which loses the original error and is really
// confusing
// TODO file some tickets on nom:
// - docs for call! are actually for apply!
// - many_till! with eof! gives a bizarre error about being unable to infer a type for E
// - error management guide seems to be pre-2.0; mentions importing from nom::util, which is
//   private, and makes no mention of verbose vs simple errors at all
//   - also, even with verbose errors, error handling kinda sucks?  i'm not even sure why this is
//     an enum when it gives me completely useless alternations, some of which (ManyTill) are
//     thrown in multiple places
//   - seems impossible to use a different error type due to rust's not very good inference rules
//   - many_till throws away the underlying error.
macro_rules! typed_eof (
    ($i:expr,) => (
        {
            let res: IResult<_, _> = eof!($i,);
            res
        }
    );
);

named!(sidedefs_lump(&[u8]) -> Vec<BareSide>, map!(many_till!(do_parse!(
    x_offset: le_i16 >>
    y_offset: le_i16 >>
    upper_texture: apply!(fixed_length_ascii, 8) >>
    lower_texture: apply!(fixed_length_ascii, 8) >>
    middle_texture: apply!(fixed_length_ascii, 8) >>
    sector: le_i16 >>
    (BareSide{
        x_offset: x_offset,
        y_offset: y_offset,
        upper_texture: upper_texture,
        lower_texture: lower_texture,
        middle_texture: middle_texture,
        sector: sector
    })
), typed_eof!()), |(r, _)| r));

// FIXME: vertices are i16 for vanilla, 15/16 fixed for ps/n64, effectively infinite but really f32 for udmf
#[derive(Debug)]
pub struct BareVertex {
    pub x: i16,
    pub y: i16,
}

impl BareVertex {
    pub fn write_to(&self, writer: &mut Write) -> Result<()> {
        try!(writer.write_i16::<LittleEndian>(self.x));
        try!(writer.write_i16::<LittleEndian>(self.y));
        Ok(())
    }
}

named!(vertexes_lump<&[u8], Vec<BareVertex>>, terminated!(many0!(do_parse!(
    x: le_i16 >>
    y: le_i16 >>
    (BareVertex{ x: x, y: y })
)), eof!()));



pub struct BareSector<'a> {
    pub floor_height: i16,
    pub ceiling_height: i16,
    pub floor_texture: &'a str,
    pub ceiling_texture: &'a str,
    pub light: i16,  // XXX what??  light can only go up to 255!
    pub sector_type: i16,  // TODO check if these are actually signed or what
    pub sector_tag: i16,
}

impl<'a> BareSector<'a> {
    pub fn write_to(&self, writer: &mut Write) -> Result<()> {
        try!(writer.write_i16::<LittleEndian>(self.floor_height));
        try!(writer.write_i16::<LittleEndian>(self.ceiling_height));
        try!(writer.write(self.floor_texture.as_bytes()));
        for _ in self.floor_texture.len() .. 8 {
            try!(writer.write(&[0]));
        }
        try!(writer.write(self.ceiling_texture.as_bytes()));
        for _ in self.ceiling_texture.len() .. 8 {
            try!(writer.write(&[0]));
        }
        try!(writer.write_i16::<LittleEndian>(self.light));
        try!(writer.write_i16::<LittleEndian>(self.sector_type));
        try!(writer.write_i16::<LittleEndian>(self.sector_tag));
        Ok(())
    }
}

named!(sectors_lump(&[u8]) -> Vec<BareSector>, terminated!(many0!(do_parse!(
    floor_height: le_i16 >>
    ceiling_height: le_i16 >>
    floor_texture: apply!(fixed_length_ascii, 8) >>
    ceiling_texture: apply!(fixed_length_ascii, 8) >>
    light: le_i16 >>
    sector_type: le_i16 >>
    sector_tag: le_i16 >>
    (BareSector{
        floor_height: floor_height,
        ceiling_height: ceiling_height,
        floor_texture: floor_texture,
        ceiling_texture: ceiling_texture,
        light: light,
        sector_type: sector_type,
        sector_tag: sector_tag,
    })
)), eof!()));


pub struct BareBinaryMap<'a, L: BareBinaryLine, T: BareBinaryThing> {
    pub vertices: Vec<BareVertex>,
    pub sectors: Vec<BareSector<'a>>,
    pub sides: Vec<BareSide<'a>>,
    pub lines: Vec<L>,
    pub things: Vec<T>,
}

/// The result of parsing a Doom-format map definition.  The contained
/// structures have not been changed in any way.  Everything is public, and
/// nothing is preventing you from meddling with the contained data in a way
/// that might make it invalid.
pub type BareDoomMap<'a> = BareBinaryMap<'a, BareDoomLine, BareDoomThing>;

/// The result of parsing a Hexen-format map definition.  The contained
/// structures have not been changed in any way.  Everything is public, and
/// nothing is preventing you from meddling with the contained data in a way
/// that might make it invalid.
pub type BareHexenMap<'a> = BareBinaryMap<'a, BareHexenLine, BareHexenThing>;

pub enum BareMap<'a> {
    Doom(BareDoomMap<'a>),
    Hexen(BareHexenMap<'a>),
}


// TODO much more error handling wow lol
pub fn parse_doom_map<'a>(archive: &'a BareWAD, range: &WADMapEntryBlock) -> Result<BareMap<'a>> {
    // TODO the map being parsed doesn't appear in the returned error...  sigh
    let vertexes_index = try!( range.vertexes_index.ok_or(ErrorKind::MissingMapLump("VERTEXES")) );
    let buf = archive.entry_slice(vertexes_index);
    let vertices = try!( nom_to_result("VERTEXES lump", buf, vertexes_lump(buf)) );

    let sectors_index = try!( range.sectors_index.ok_or(ErrorKind::MissingMapLump("SECTORS")) );
    let buf = archive.entry_slice(sectors_index);
    let sectors = try!( nom_to_result("SECTORS lump", buf, sectors_lump(buf)) );

    let sidedefs_index = try!( range.sidedefs_index.ok_or(ErrorKind::MissingMapLump("SIDEDEFS")) );
    let buf = archive.entry_slice(sidedefs_index);
    let sides = try!( nom_to_result("SIDEDEFS lump", buf, sidedefs_lump(buf)) );

    if range.format == MapFormat::Doom {
        let linedefs_index = try!( range.linedefs_index.ok_or(ErrorKind::MissingMapLump("LINEDEFS")) );
        let buf = archive.entry_slice(linedefs_index);
        let lines = try!( nom_to_result("LINEDEFS lump", buf, doom_linedefs_lump(buf)) );

        let things_index = try!( range.things_index.ok_or(ErrorKind::MissingMapLump("THINGS")) );
        let buf = archive.entry_slice(things_index);
        let things = try!( nom_to_result("THINGS lump", buf, doom_things_lump(buf)) );

        return Ok(BareMap::Doom(BareDoomMap{
            vertices: vertices,
            sectors: sectors,
            sides: sides,
            lines: lines,
            things: things,
        }));
    }
    else {
        let linedefs_index = try!( range.linedefs_index.ok_or(ErrorKind::MissingMapLump("LINEDEFS")) );
        let buf = archive.entry_slice(linedefs_index);
        let lines = try!( nom_to_result("LINEDEFS lump", buf, hexen_linedefs_lump(buf)) );

        let things_index = try!( range.things_index.ok_or(ErrorKind::MissingMapLump("THINGS")) );
        let buf = archive.entry_slice(things_index);
        let things = try!( nom_to_result("THINGS lump", buf, hexen_things_lump(buf)) );

        return Ok(BareMap::Hexen(BareHexenMap{
            vertices: vertices,
            sectors: sectors,
            sides: sides,
            lines: lines,
            things: things,
        }));
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Facing {
    Front,
    Back,
}

use std::collections::HashMap;
// TODO ok so this is mildly clever but won't work once we get to UDMF champ
impl<'a, L: BareBinaryLine, T: BareBinaryThing> BareBinaryMap<'a, L, T> {
    // TODO this is a horrible fucking mess.  but it's a /contained/ horrible fucking mess, so.
    pub fn sector_to_polygons(&'a self, s: usize) -> Vec<Vec<&'a BareVertex>> {
        struct Edge<'a, L: 'a> {
            line: &'a L,
            side: &'a BareSide<'a>,
            facing: Facing,
            v0: &'a BareVertex,
            v1: &'a BareVertex,
            done: bool,
        }
        // This is just to convince HashMap to hash on the actual reference, not the underlying
        // BareVertex value
        struct VertexRef<'a>(&'a BareVertex);
        impl<'a> PartialEq for VertexRef<'a> {
            fn eq(&self, other: &VertexRef) -> bool {
                return (self.0 as *const _) == (other.0 as *const _);
            }
        }
        impl<'a> Eq for VertexRef<'a> {}
        impl<'a> std::hash::Hash for VertexRef<'a> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                (self.0 as *const _).hash(state)
            }
        }

        let mut edges = vec![];
        let mut vertices_to_edges = HashMap::new();
        // TODO linear scan -- would make more sense to turn the entire map into polygons in one go
        for line in self.lines.iter() {
            let (frontid, backid) = line.side_indices();
            for &(facing, sideid) in [(Facing::Front, frontid), (Facing::Back, backid)].iter() {
                if sideid == -1 {
                    continue;
                }
                // TODO this and the vertices lookups might be bogus and crash...
                let side = &self.sides[sideid as usize];
                if side.sector as usize == s {
                    let (v0, v1) = line.vertex_indices();
                    let edge = Edge{
                        line: line,
                        side: side,
                        facing: facing,
                        // TODO should these be swapped depending on the line facing?
                        v0: &self.vertices[v0 as usize],
                        v1: &self.vertices[v1 as usize],
                        done: false,
                    };
                    edges.push(edge);
                    vertices_to_edges.entry(VertexRef(&self.vertices[v0 as usize])).or_insert_with(|| Vec::new()).push(edges.len() - 1);
                    vertices_to_edges.entry(VertexRef(&self.vertices[v1 as usize])).or_insert_with(|| Vec::new()).push(edges.len() - 1);
                }
            }
        }

        // Trace sectors by starting at the first side's first vertex and attempting to walk from
        // there
        let mut outlines = Vec::new();
        let mut seen_vertices = HashMap::new();
        while edges.len() > 0 {
            let mut next_vertices = vec![];
            for edge in edges.iter() {
                // TODO having done-ness for both edges and vertices seems weird, idk
                if !seen_vertices.contains_key(&VertexRef(edge.v0)) {
                    next_vertices.push(edge.v0);
                    break;
                }
                if !seen_vertices.contains_key(&VertexRef(edge.v1)) {
                    next_vertices.push(edge.v1);
                    break;
                }
            }
            if next_vertices.len() == 0 {
                break;
            }

            let mut outline = Vec::new();
            while next_vertices.len() > 0 {
                let mut vertices = next_vertices;
                next_vertices = Vec::new();
                for vertex in vertices.iter() {
                    if seen_vertices.contains_key(&VertexRef(vertex)) {
                        continue;
                    }
                    seen_vertices.insert(VertexRef(vertex), true);
                    outline.push(*vertex);

                    // TODO so, problems occur here if:
                    // - a vertex has more than two edges
                    //   - special case: double-sided edges are OK!  but we have to eliminate
                    //   those, WITHOUT ruining entirely self-referencing sectors
                    // - a vertex has one edge
                    for e in vertices_to_edges.get(&VertexRef(vertex)).unwrap().iter() {
                        let edge = &mut edges[*e];
                        if edge.done {
                            // TODO actually this seems weird?  why would this happen.
                            continue;
                        }
                        edge.done = true;
                        if !seen_vertices.contains_key(&VertexRef(edge.v0)) {
                            outline.push(edge.v0);
                            next_vertices.push(edge.v0);
                        }
                        else if !seen_vertices.contains_key(&VertexRef(edge.v1)) {
                            outline.push(edge.v1);
                            next_vertices.push(edge.v1);
                        }
                        // Only add EXACTLY ONE vertex at a time for now -- so, assuming simple
                        // polygons!  Figure out the rest, uh, later.
                        break;
                    }
                }
            }
            if outline.len() > 0 {
                outlines.push(outline);
            }
        }

        return outlines;
    }

    // TODO of course, this doesn't take later movement of sectors into account, dammit
    pub fn count_textures(&'a self) -> HashMap<&'a str, (usize, f32)> {
        let mut counts: HashMap<&'a str, (usize, f32)> = HashMap::new();

        // This block exists only so `add` goes out of scope (and stops borrowing counts) before we
        // return; I don't know why the compiler cares when `add` clearly doesn't escape
        {
            let mut add = |tex: &'a str, area: f32| {
                // TODO iirc doom64 or something uses a different empty texture name, "?"
                if tex != "-" {
                    let entry = counts.entry(tex).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += area;
                }
            };

            for line in self.lines.iter() {
                let (frontid, backid) = line.side_indices();
                if frontid == -1 && backid == -1 {
                    // No sides; skip
                    continue;
                }

                let (v0i, v1i) = line.vertex_indices();
                let v0 = &self.vertices[v0i as usize];
                let v1 = &self.vertices[v1i as usize];
                let dx = (v1.x - v0.x) as f32;
                let dy = (v1.y - v0.y) as f32;
                let length = (dx * dx + dy * dy).sqrt();

                if frontid != -1 && backid != -1 {
                    // Two-sided line
                    // TODO checking for the two-sided flag is an interesting map check
                    // TODO this might be bogus and crash...
                    // TODO actually that's a good thing to put in a map check
                    let front_side = &self.sides[frontid as usize];
                    let back_side = &self.sides[backid as usize];
                    // TODO sector is an i16??  can it be negative???  indicating no sector?
                    // (i mean obviously it can be bogus regardless, but can it be deliberately bogus?)
                    let front_sector = &self.sectors[front_side.sector as usize];
                    let back_sector = &self.sectors[back_side.sector as usize];

                    let lowest_ceiling;
                    let ceiling_diff = front_sector.ceiling_height - back_sector.ceiling_height;
                    if ceiling_diff > 0 {
                        let front_upper_height = ceiling_diff as f32;
                        add(front_side.upper_texture, length * front_upper_height);
                        lowest_ceiling = back_sector.ceiling_height;
                    }
                    else {
                        let back_upper_height = -ceiling_diff as f32;
                        add(front_side.upper_texture, length * back_upper_height);
                        lowest_ceiling = front_sector.ceiling_height;
                    }

                    let highest_floor;
                    let floor_diff = front_sector.floor_height - back_sector.floor_height;
                    if floor_diff > 0 {
                        let back_lower_height = floor_diff as f32;
                        add(back_side.lower_texture, length * back_lower_height);
                        highest_floor = back_sector.floor_height;
                    }
                    else {
                        let front_lower_height = -floor_diff as f32;
                        add(front_side.lower_texture, length * front_lower_height);
                        highest_floor = front_sector.floor_height;
                    }

                    let middle_height = (lowest_ceiling - highest_floor) as f32;
                    // TODO map check for negative height (but this is valid for vavoom-style 3d floors!)
                    if middle_height > 0.0 {
                        add(front_side.middle_texture, length * middle_height);
                        add(back_side.middle_texture, length * middle_height);
                    }
                }
                else if backid == -1 {
                    // Typical one-sided wall
                    // TODO map check for no two-sided flag
                    let front_side = &self.sides[frontid as usize];
                    let front_sector = &self.sectors[front_side.sector as usize];
                    let middle_height = (front_sector.ceiling_height - front_sector.floor_height) as f32;
                    add(front_side.middle_texture, length * middle_height);
                }
                else if frontid == -1 {
                    // Backwards one-sided wall
                    // TODO map check for no two-sided flag
                    // TODO maybe a warning for this case too because it's weird
                    let back_side = &self.sides[backid as usize];
                    let back_sector = &self.sectors[back_side.sector as usize];
                    let middle_height = (back_sector.ceiling_height - back_sector.floor_height) as f32;
                    add(back_side.middle_texture, length * middle_height);
                }
            }
        }

        return counts;
    }
}


pub struct TEXTURExEntry<'a> {
    pub name: &'a str,
    pub width: i16,
    pub height: i16,
}

named!(texturex_lump_header(&[u8]) -> Vec<i32>, do_parse!(
    numtextures: le_i32 >>
    offsets: many_m_n!(numtextures as usize, numtextures as usize, le_i32) >>
    (offsets)
));

named!(texturex_lump_entry(&[u8]) -> TEXTURExEntry, do_parse!(
    name: apply!(fixed_length_ascii, 8) >>
    le_i32 >>  // "masked", unused
    // TODO these should be positive
    width: le_i16 >>
    height: le_i16 >>
    le_i32 >>  // "columndirectory", unused
    patchcount: le_i16 >>
    // TODO patches
    (TEXTURExEntry{
        name: name,
        width: width,
        height: height,
    })
));

pub fn parse_texturex_names<'a>(buf: &'a [u8]) -> Result<Vec<TEXTURExEntry<'a>>> {
    let offsets = try!(nom_to_result("TEXTUREx header", buf, texturex_lump_header(buf)));
    let mut ret = Vec::with_capacity(offsets.len());
    for (i, &offset) in offsets.iter().enumerate() {
        if offset < 0 {
            bail!(ErrorKind::NegativeOffset("TEXTUREx", i, offset as isize));
        }
        // TODO check for too large offset too, instead of Incomplete
        ret.push(try!(nom_to_result("TEXTUREx", buf, texturex_lump_entry(&buf[(offset as usize)..]))));
    }
    return Ok(ret);
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
