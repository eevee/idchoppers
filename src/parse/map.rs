use std::io::Write;
use std::str;
use std::u8;

use byteorder::{LittleEndian, WriteBytesExt};
use nom::{le_i16, le_u16, le_u8};

use super::util::fixed_length_ascii;
use ::errors::{ErrorKind, Result, nom_to_result};
use ::map::{MapFormat};
use ::archive::wad::{BareWAD, WADMapEntryBlock};
use ::util::RefKey;

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

named!(hexen_args<[u8; 5]>, count_fixed!(u8, le_u8, 5));

#[derive(Debug)]
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
        writer.write_i16::<LittleEndian>(self.x)?;
        writer.write_i16::<LittleEndian>(self.y)?;
        writer.write_i16::<LittleEndian>(self.angle)?;
        writer.write_i16::<LittleEndian>(self.doomednum)?;
        writer.write_u16::<LittleEndian>(self.flags)?;
        Ok(())
    }
}

named!(doom_things_lump<Vec<BareDoomThing>>, many0!(complete!(do_parse!(
    x: le_i16 >>
    y: le_i16 >>
    angle: le_i16 >>
    doomednum: le_i16 >>
    flags: le_u16 >>
    (BareDoomThing{ x, y, angle, doomednum, flags })
))));

#[derive(Debug)]
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

named!(hexen_things_lump<Vec<BareHexenThing>>, many0!(complete!(do_parse!(
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
        tid,
        x,
        y,
        z,
        angle,
        doomednum,
        flags,
        special,
        args,
    })
))));

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
#[derive(Debug)]
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
        writer.write_i16::<LittleEndian>(self.v0)?;
        writer.write_i16::<LittleEndian>(self.v1)?;
        writer.write_i16::<LittleEndian>(self.flags)?;
        writer.write_i16::<LittleEndian>(self.special)?;
        writer.write_i16::<LittleEndian>(self.sector_tag)?;
        writer.write_i16::<LittleEndian>(self.front_sidedef)?;
        writer.write_i16::<LittleEndian>(self.back_sidedef)?;
        Ok(())
    }
}

named!(doom_linedefs_lump<Vec<BareDoomLine>>, many0!(complete!(do_parse!(
    v0: le_i16 >>
    v1: le_i16 >>
    flags: le_i16 >>
    special: le_i16 >>
    sector_tag: le_i16 >>
    front_sidedef: le_i16 >>
    back_sidedef: le_i16 >>
    (BareDoomLine{ v0, v1, flags, special, sector_tag, front_sidedef, back_sidedef })
))));

// TODO source ports extended ids to unsigned here too
#[derive(Debug)]
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

named!(hexen_linedefs_lump<Vec<BareHexenLine>>, many0!(complete!(do_parse!(
    v0: le_i16 >>
    v1: le_i16 >>
    flags: le_i16 >>
    special: le_u8 >>
    args: hexen_args >>
    front_sidedef: le_i16 >>
    back_sidedef: le_i16 >>
    (BareHexenLine{
        v0,
        v1,
        flags,
        special,
        args,
        front_sidedef,
        back_sidedef,
    })
))));

pub trait BareBinaryLine {
    fn vertex_indices(&self) -> (i16, i16);
    fn side_indices(&self) -> (i16, i16);
    fn has_special(&self) -> bool;
    fn flags(&self) -> i16;
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
    fn flags(&self) -> i16 {
        self.flags
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
    fn flags(&self) -> i16 {
        self.flags
    }
}

#[derive(Debug)]
pub struct BareSide<'tex> {
    pub x_offset: i16,
    pub y_offset: i16,
    pub upper_texture: &'tex str,
    pub lower_texture: &'tex str,
    pub middle_texture: &'tex str,
    pub sector: i16,
}

impl<'t> BareSide<'t> {
    pub fn write_to(&self, writer: &mut Write) -> Result<()> {
        writer.write_i16::<LittleEndian>(self.x_offset)?;
        writer.write_i16::<LittleEndian>(self.y_offset)?;
        writer.write(self.upper_texture.as_bytes())?;
        for _ in self.upper_texture.len() .. 8 {
            writer.write(&[0])?;
        }
        writer.write(self.lower_texture.as_bytes())?;
        for _ in self.lower_texture.len() .. 8 {
            writer.write(&[0])?;
        }
        writer.write(self.middle_texture.as_bytes())?;
        for _ in self.middle_texture.len() .. 8 {
            writer.write(&[0])?;
        }
        writer.write_i16::<LittleEndian>(self.sector)?;
        Ok(())
    }
}

named!(sidedefs_lump<Vec<BareSide>>, many0!(complete!(do_parse!(
    x_offset: le_i16 >>
    y_offset: le_i16 >>
    upper_texture: apply!(fixed_length_ascii, 8) >>
    lower_texture: apply!(fixed_length_ascii, 8) >>
    middle_texture: apply!(fixed_length_ascii, 8) >>
    sector: le_i16 >>
    (BareSide{
        x_offset,
        y_offset,
        upper_texture,
        lower_texture,
        middle_texture,
        sector
    })
))));

// FIXME: vertices are i16 for vanilla, 15/16 fixed for ps/n64, effectively infinite but really f32 for udmf
#[derive(Debug)]
pub struct BareVertex {
    pub x: i16,
    pub y: i16,
}

impl BareVertex {
    pub fn write_to(&self, writer: &mut Write) -> Result<()> {
        writer.write_i16::<LittleEndian>(self.x)?;
        writer.write_i16::<LittleEndian>(self.y)?;
        Ok(())
    }
}

named!(vertexes_lump<Vec<BareVertex>>, many0!(complete!(do_parse!(
    x: le_i16 >>
    y: le_i16 >>
    (BareVertex{ x, y })
))));



#[derive(Debug)]
pub struct BareSector<'tex> {
    pub floor_height: i16,
    pub ceiling_height: i16,
    pub floor_texture: &'tex str,
    pub ceiling_texture: &'tex str,
    pub light: i16,  // XXX what??  light can only go up to 255!
    pub sector_type: i16,  // TODO check if these are actually signed or what
    pub sector_tag: i16,
}

impl<'t> BareSector<'t> {
    pub fn write_to(&self, writer: &mut Write) -> Result<()> {
        writer.write_i16::<LittleEndian>(self.floor_height)?;
        writer.write_i16::<LittleEndian>(self.ceiling_height)?;
        writer.write(self.floor_texture.as_bytes())?;
        for _ in self.floor_texture.len() .. 8 {
            writer.write(&[0])?;
        }
        writer.write(self.ceiling_texture.as_bytes())?;
        for _ in self.ceiling_texture.len() .. 8 {
            writer.write(&[0])?;
        }
        writer.write_i16::<LittleEndian>(self.light)?;
        writer.write_i16::<LittleEndian>(self.sector_type)?;
        writer.write_i16::<LittleEndian>(self.sector_tag)?;
        Ok(())
    }
}

named!(sectors_lump<Vec<BareSector>>, many0!(complete!(do_parse!(
    floor_height: le_i16 >>
    ceiling_height: le_i16 >>
    floor_texture: apply!(fixed_length_ascii, 8) >>
    ceiling_texture: apply!(fixed_length_ascii, 8) >>
    light: le_i16 >>
    sector_type: le_i16 >>
    sector_tag: le_i16 >>
    (BareSector{
        floor_height,
        ceiling_height,
        floor_texture,
        ceiling_texture,
        light,
        sector_type,
        sector_tag,
    })
))));


#[derive(Debug)]
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

#[derive(Debug)]
pub enum BareMap<'a> {
    Doom(BareDoomMap<'a>),
    Hexen(BareHexenMap<'a>),
}


// TODO much more error handling wow lol
pub fn parse_doom_map<'a>(archive: &'a BareWAD, range: &WADMapEntryBlock) -> Result<BareMap<'a>> {
    // TODO the map being parsed doesn't appear in the returned error...  sigh
    let vertexes_index = range.vertexes_index.ok_or(ErrorKind::MissingMapLump("VERTEXES"))?;
    let buf = archive.entry_slice(vertexes_index);
    let vertices = nom_to_result("VERTEXES lump", buf, vertexes_lump(buf))?;

    let sectors_index = range.sectors_index.ok_or(ErrorKind::MissingMapLump("SECTORS"))?;
    let buf = archive.entry_slice(sectors_index);
    let sectors = nom_to_result("SECTORS lump", buf, sectors_lump(buf))?;

    let sidedefs_index = range.sidedefs_index.ok_or(ErrorKind::MissingMapLump("SIDEDEFS"))?;
    let buf = archive.entry_slice(sidedefs_index);
    let sides = nom_to_result("SIDEDEFS lump", buf, sidedefs_lump(buf))?;

    if range.format == MapFormat::Doom {
        let linedefs_index = range.linedefs_index.ok_or(ErrorKind::MissingMapLump("LINEDEFS"))?;
        let buf = archive.entry_slice(linedefs_index);
        let lines = nom_to_result("LINEDEFS lump", buf, doom_linedefs_lump(buf))?;

        let things_index = range.things_index.ok_or(ErrorKind::MissingMapLump("THINGS"))?;
        let buf = archive.entry_slice(things_index);
        let things = nom_to_result("THINGS lump", buf, doom_things_lump(buf))?;

        Ok(BareMap::Doom(BareDoomMap{
            vertices,
            sectors,
            sides,
            lines,
            things,
        }))
    }
    else {
        let linedefs_index = range.linedefs_index.ok_or(ErrorKind::MissingMapLump("LINEDEFS"))?;
        let buf = archive.entry_slice(linedefs_index);
        let lines = nom_to_result("LINEDEFS lump", buf, hexen_linedefs_lump(buf))?;

        let things_index = range.things_index.ok_or(ErrorKind::MissingMapLump("THINGS"))?;
        let buf = archive.entry_slice(things_index);
        let things = nom_to_result("THINGS lump", buf, hexen_things_lump(buf))?;

        Ok(BareMap::Hexen(BareHexenMap{
            vertices,
            sectors,
            sides,
            lines,
            things,
        }))
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
    pub fn sector_to_polygons(&self, s: usize) -> Vec<Vec<&BareVertex>> {
        struct Edge<'a, L: 'a> {
            _line: &'a L,
            _side: &'a BareSide<'a>,
            _facing: Facing,
            v0: &'a BareVertex,
            v1: &'a BareVertex,
            done: bool,
        }

        let mut edges = Vec::new();
        let mut vertices_to_edges = HashMap::new();
        // TODO linear scan -- would make more sense to turn the entire map into polygons in one go
        for line in self.lines.iter() {
            let (frontid, backid) = line.side_indices();
            // FIXME need to do this better
            if frontid != -1 && backid != -1 && self.sides[frontid as usize].sector == self.sides[backid as usize].sector {
                continue;
            }

            for &(facing, sideid) in [(Facing::Front, frontid), (Facing::Back, backid)].iter() {
                if sideid == -1 {
                    continue;
                }
                // TODO this and the vertices lookups might be bogus and crash...
                let side = &self.sides[sideid as usize];
                if side.sector as usize == s {
                    let (v0, v1) = line.vertex_indices();
                    let edge = Edge{
                        _line: line,
                        _side: side,
                        _facing: facing,
                        // TODO should these be swapped depending on the line facing?
                        v0: &self.vertices[v0 as usize],
                        v1: &self.vertices[v1 as usize],
                        done: false,
                    };
                    edges.push(edge);
                    vertices_to_edges
                        .entry(RefKey(&self.vertices[v0 as usize]))
                        .or_insert_with(Vec::new)
                        .push(edges.len() - 1);
                    vertices_to_edges
                        .entry(RefKey(&self.vertices[v1 as usize]))
                        .or_insert_with(Vec::new)
                        .push(edges.len() - 1);
                }
            }
        }

        // Trace sectors by starting at the first side's first vertex and attempting to walk from
        // there
        let mut outlines = Vec::new();
        let mut seen_vertices: HashMap<RefKey<BareVertex>, bool> = HashMap::new();
        while edges.len() > 0 {
            let mut next_vertices = Vec::new();
            for edge in edges.iter() {
                // TODO having done-ness for both edges and vertices seems weird, idk
                if !seen_vertices.contains_key(&RefKey(edge.v0)) {
                    next_vertices.push(edge.v0);
                    break;
                }
                if !seen_vertices.contains_key(&RefKey(edge.v1)) {
                    next_vertices.push(edge.v1);
                    break;
                }
            }
            if next_vertices.is_empty() {
                break;
            }

            let mut outline = Vec::new();
            while next_vertices.len() > 0 {
                let vertices = next_vertices;
                next_vertices = Vec::new();
                for vertex in vertices.iter() {
                    if seen_vertices.contains_key(&RefKey(vertex)) {
                        continue;
                    }
                    seen_vertices.insert(RefKey(vertex), true);
                    outline.push(*vertex);

                    // TODO so, problems occur here if:
                    // - a vertex has more than two edges
                    //   - special case: double-sided edges are OK!  but we have to eliminate
                    //   those, WITHOUT ruining entirely self-referencing sectors
                    // - a vertex has one edge
                    for e in vertices_to_edges.get(&RefKey(vertex)).unwrap().iter() {
                        let edge = &mut edges[*e];
                        if edge.done {
                            // TODO actually this seems weird?  why would this happen.
                            continue;
                        }
                        edge.done = true;
                        if !seen_vertices.contains_key(&RefKey(edge.v0)) {
                            next_vertices.push(edge.v0);
                        }
                        else if !seen_vertices.contains_key(&RefKey(edge.v1)) {
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

        outlines
    }

    // TODO of course, this doesn't take later movement of sectors into account, dammit
    pub fn count_textures(&self) -> HashMap<&str, (usize, f32)> {
        let mut counts = HashMap::new();

        // This block exists only so `add` goes out of scope (and stops borrowing counts) before we
        // return; I don't know why the compiler cares when `add` clearly doesn't escape
        {
            let mut add = |tex, area| {
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

        counts
    }
}
