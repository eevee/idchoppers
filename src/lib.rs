#[macro_use]
extern crate nom;
#[macro_use]
extern crate error_chain;

use std::str::FromStr;
use std::str;
use std::u8;

use nom::{IResult, Needed, digit, le_i16, le_u32};

pub mod errors;
use errors::{ErrorKind, Result};



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

pub enum MapName {
    ExMy(u8, u8),
    MAPxx(u8),
}

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
    x1: digit >>
    x2: digit >>
    eof!() >>
    (MapName::MAPxx(
        // TODO i can't help but feel that this is suboptimal
        10 * u8::from_str(str::from_utf8(x1).unwrap()).unwrap()
        + u8::from_str(str::from_utf8(x2).unwrap()).unwrap()
    ))
));

named!(vanilla_map_name(&[u8]) -> MapName, alt!(exmy_map_name | mapxx_map_name));



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

fn nom_to_result<I, O>(result: IResult<I, O>) -> Result<O> {
    match result {
        IResult::Done(_, value) => {
            return Ok(value);
        }
        IResult::Incomplete(needed) => {
            // TODO what is incomplete, exactly?  where did we run out of data?
            bail!(ErrorKind::TruncatedData);
        }
        IResult::Error(err) => {
            let error_kind = match err {
                nom::Err::Code(ref k) | nom::Err::Node(ref k, _) | nom::Err::Position(ref k, _) | nom::Err::NodePosition(ref k, _, _) => k
            };
            match *error_kind {
                nom::ErrorKind::Custom(1) => bail!(ErrorKind::InvalidMagic),
                _ => bail!(ErrorKind::ParseError),
            }
        }
    }
}

named!(iwad_tag(&[u8]) -> WADType, do_parse!(tag!(b"IWAD") >> (WADType::IWAD)));
named!(pwad_tag(&[u8]) -> WADType, do_parse!(tag!(b"PWAD") >> (WADType::PWAD)));

named!(wad_header(&[u8]) -> WADHeader, do_parse!(
    identification: return_error!(
        nom::ErrorKind::Custom(1),
        alt!(iwad_tag | pwad_tag)) >>
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
        let (leftovers, entry) = try_parse!(parse_from, wad_entry);
        ret.push(entry);
        parse_from = leftovers;
    }
    return IResult::Done(parse_from, ret);
}


// TODO:
// - get entry by name (REMEMBER: NAMES CAN REPEAT)
// - detect entry type (using name as a hint as well)
// - how the hell do i iterate over /either/ an entry or a map?  maybe that's not even a useful thing to do
pub struct WADArchive<'a> {
    pub buffer: &'a [u8],
    pub header: WADHeader,
    pub directory: Vec<WADDirectoryEntry<'a>>,
}

pub struct WADEntry<'a> {
    pub archive: &'a WADArchive<'a>,
    pub index: usize,
}

impl<'a> WADArchive<'a> {
    pub fn iter_maps(&'a self) -> WADMapIterator<'a> {
        return WADMapIterator{ archive: self, entry_iter: self.directory.iter().enumerate() };
    }

    pub fn entry_slice(&'a self, index: usize) -> &'a [u8] {
        let entry = &self.directory[index];
        let start = entry.filepos as usize;
        let end = start + entry.size as usize;
        return &self.buffer[start..end];
    }
}

// TODO problems to scan a wad map for:
// - missing a required lump
// TODO curiosities to scan a wad map for:
// - content in ENDMAP
// - UDMF along with the old-style text maps


// FIXME use Result, but, figure out how to get an actual error out of here
// FIXME actually this fairly simple format is a good place to start thinking about how to return errors in general; like, do i want custom errors for tags?  etc
pub fn parse_wad(buf: &[u8]) -> Result<WADArchive> {
    // FIXME ambiguous whether the error was from parsing the header or the entries
    let header = try!(nom_to_result(wad_header(buf)));
    let entries = try!(nom_to_result(wad_directory(buf, &header)));
    return Ok(WADArchive{ buffer: buf, header: header, directory: entries });
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

pub struct WADMapIterator<'a> {
    archive: &'a WADArchive<'a>,
    entry_iter: std::iter::Enumerate<std::slice::Iter<'a, WADDirectoryEntry<'a>>>,
}

impl<'a> Iterator for WADMapIterator<'a> {
    type Item = WADMapEntryBlock;

    fn next(&mut self) -> Option<WADMapEntryBlock> {
        // Alas!  The compiler is not smart enough to realize that the
        // following loop is inescapable without both of these getting values.
        // TODO maybe i can reword to make this work?
        //let mut marker_index = 0;
        //let mut map_name = "";
        let (marker_index, map_name);
        loop {
            if let Some((i, entry)) = self.entry_iter.next() {
                if let IResult::Done(_leftovers, found_map_name) = vanilla_map_name(entry.name.as_bytes()) {
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
        match self.entry_iter.next() {
            Some((next_i, next_entry)) => {
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
                    "BEHAVIOR" => { range.behavior_index = Some(i); }
                    "TEXTMAP" => { range.textmap_index = Some(i); }
                    _ => {
                        // TODO wait, what's the right thing here
                        break;
                    }
                }
                match self.entry_iter.next() {
                    Some((next_i, next_entry)) => {
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
// FIXME should this really be a list of general lump types?
// FIXME not actually used lol
/// List of lumps that can appear in a (non-UDMF) map, in this order.
enum MapLumps {
    /// A marker that provides the map name, usually ExMy or MAPxx.  Generally
    /// empty, although FraggleScript is put in this lump.
	Marker,             // A separator, name, ExMx or MAPxx

    /// List of things (actors) in the map.  Format is different for Doom
    /// versus Hexen maps.
    Things,

    /// List of lines in the map.  Format is different for Doom versus Hexen
    /// maps.
    Linedefs,

    /// List of sidedefs in the map, the front and back sides of a line.
    Sidedefs,

    /// List of vertices in the map.
    Vertexes,

    Segs,
    Subsectors,
    Nodes,
    Sectors,
    Reject,
    Blockmap,
    Behavior,
    Conversation,

    // FIXME: also, ZNODES, GLZNODES, and of course TEXTMAP and ENDMAP
}

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

// FIXME: vertices are i16 for vanilla, 15/16 fixed for ps/n64, effectively infinite but really f32 for udmf
pub struct DoomVertex {
    pub x: i16,
    pub y: i16,
}

named!(vertexes_lump<&[u8], Vec<DoomVertex>>, terminated!(many0!(do_parse!(
    x: le_i16 >>
    y: le_i16 >>
    (DoomVertex{ x: x, y: y })
)), eof!()));


// FIXME vertex/sidedef indices are i16 in vanilla, but extended to u16 in most source ports; note that for true vanilla, a negative index makes no sense anyway
// FIXME hexen extends this, which requires detecting hexen format
// FIXME what exactly is the higher-level structure that holds actual references to the sidedefs?
pub struct DoomLine {
    pub v0: i16,
    pub v1: i16,
    pub flags: i16,
    pub special: i16,
    pub sector_tag: i16,
    // NOTE: -1 to mean none
    pub front_sidedef: i16,
    pub back_sidedef: i16,
}

named!(doom_linedefs_lump<&[u8], Vec<DoomLine>>, terminated!(many0!(do_parse!(
    v0: le_i16 >>
    v1: le_i16 >>
    flags: le_i16 >>
    special: le_i16 >>
    sector_tag: le_i16 >>
    front_sidedef: le_i16 >>
    back_sidedef: le_i16 >>
    (DoomLine{ v0: v0, v1: v1, flags: flags, special: special, sector_tag: sector_tag, front_sidedef: front_sidedef, back_sidedef: back_sidedef })
)), eof!()));


pub struct DoomSide {}
pub struct DoomThing {}
pub struct DoomSector {}

/// The result of parsing a Doom-format map definition.  The contained
/// structures have not been changed in any way.  Everything is public, and
/// nothing is preventing you from meddling with the contained data in a way
/// that might make it invalid.
pub struct BareDoomMap {
    pub vertices: Vec<DoomVertex>,
    pub sectors: Vec<DoomSector>,
    pub sides: Vec<DoomSide>,
    pub lines: Vec<DoomLine>,
    pub things: Vec<DoomThing>,
}


// TODO much more error handling wow lol
pub fn parse_doom_map(archive: &WADArchive, range: &WADMapEntryBlock) -> Result<BareDoomMap> {
    let vertexes_index = try!( range.vertexes_index.ok_or(ErrorKind::MissingMapLump("VERTEXES")) );
    let buf = archive.entry_slice(vertexes_index);
    let vertices = try!( nom_to_result(vertexes_lump(buf)) );

    let linedefs_index = try!( range.linedefs_index.ok_or(ErrorKind::MissingMapLump("LINEDEFS")) );
    let buf = archive.entry_slice(linedefs_index);
    let lines = try!( nom_to_result(doom_linedefs_lump(buf)) );

    return Ok(BareDoomMap{
        vertices: vertices,
        lines: lines,
        sectors: vec![],
        sides: vec![],
        things: vec![],
    });
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
