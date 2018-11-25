use super::parse::map::BareDoomMap;
use super::geom::{Coord, Point, Rect};

use std::collections::HashMap;
use std::marker::PhantomData;

use std;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MapFormat {
    Doom,
    Hexen,
    UDMF,
}

// FIXME zdoom allows arbitrary map names, i think?  how does that even work
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

// TODO
// map diagnostics
// - error: 
// - info: unused vertex
// - info: unused side
// - info: sector with no sides
// - info: thing not in the map (polyobjs excluded)

/// A fully-fledged map, independent (more or less) of any particular underlying format.
// TODO actually i'm not so sure about that!  should i, say, have different map impls that use
// different types...
pub struct Map {
    /*
    lines: Vec<Rc<RefCell<Line>>>,
    sides: Vec<Rc<RefCell<Side>>>,
    sectors: Vec<Rc<RefCell<Sector>>>,
    things: Vec<Rc<RefCell<Thing>>>,
    vertices: Vec<Rc<RefCell<Vertex>>>,
    */
    lines: Vec<Line>,
    sides: Vec<Side>,
    sectors: Vec<Sector>,
    things: Vec<Thing>,
    vertices: Vec<Vertex>,

    _bbox: Option<Rect>,
}

impl Map {
    pub fn new() -> Self {
        Map {
            lines: Vec::new(),
            sides: Vec::new(),
            sectors: Vec::new(),
            things: Vec::new(),
            vertices: Vec::new(),

            _bbox: None,
        }
    }
    pub fn from_bare(bare_map: &BareDoomMap) -> Self {
        let mut map = Map::new();
        for bare_sector in bare_map.sectors.iter() {
            let sectorh = map.add_sector();
            let sector = &mut map.sectors[sectorh.0];
            sector.tag = bare_sector.sector_tag as u32;
            sector.special = bare_sector.sector_type as u32;
            sector.floor_height = bare_sector.floor_height as i32;
            sector.ceiling_height = bare_sector.ceiling_height as i32;
            sector.light = bare_sector.light as u8;
        }
        for bare_vertex in bare_map.vertices.iter() {
            map.add_vertex(bare_vertex.x as f64, bare_vertex.y as f64);
        }
        for bare_side in bare_map.sides.iter() {
            let handle = map.add_side((bare_side.sector as usize).into());
            let side = map.side_mut(handle);
            side.lower_texture = bare_side.lower_texture.into();
            side.middle_texture = bare_side.middle_texture.into();
            side.upper_texture = bare_side.upper_texture.into();
        }
        for bare_line in bare_map.lines.iter() {
            let handle = map.add_line((bare_line.v0 as usize).into(), (bare_line.v1 as usize).into());
            let line = map.line_mut(handle);
            line.flags = bare_line.flags as u32;
            line.special = bare_line.special as usize;
            line.sector_tag = bare_line.sector_tag as u32;
            // FIXME and here's where we start to go awry -- this should use a method.  so should
            // new side w/ sector
            if bare_line.front_sidedef != -1 {
                line.front = Some((bare_line.front_sidedef as usize).into());
            }
            if bare_line.back_sidedef != -1 {
                line.back = Some((bare_line.back_sidedef as usize).into());
            }
        }
        for bare_thing in bare_map.things.iter() {
            map.things.push(Thing{
                point: Point::new(bare_thing.x as Coord, bare_thing.y as Coord),
                doomednum: bare_thing.doomednum as u32,
            });
        }

        map
    }

    fn side_mut(&mut self, handle: Handle<Side>) -> &mut Side {
        &mut self.sides[handle.0]
    }
    fn line_mut(&mut self, handle: Handle<Line>) -> &mut Line {
        &mut self.lines[handle.0]
    }

    fn add_sector(&mut self) -> Handle<Sector> {
        self.sectors.push(Sector {
            special: 0,
            tag: 0,
            floor_height: 0,
            ceiling_height: 0,
            light: 0,
        });
        (self.sectors.len() - 1).into()
    }
    fn add_side(&mut self, sector: Handle<Sector>) -> Handle<Side> {
        self.sides.push(Side{
            id: 0,
            lower_texture: "".into(),
            middle_texture: "".into(),
            upper_texture: "".into(),
            sector: sector,
        });
        (self.sides.len() - 1).into()
    }
    fn add_vertex(&mut self, x: f64, y: f64) {
        self.vertices.push(Vertex{ x, y });
        //self.vertices.push(vertex);
        //return vertex;
    }
    fn add_line(&mut self, start: Handle<Vertex>, end: Handle<Vertex>) -> Handle<Line> {
        let index = self.lines.len();
        self.lines.push(Line{
            index,
            start,
            end,
            flags: 0,
            special: 0,
            sector_tag: 0,
            front: None,
            back: None,
        });
        index.into()
    }

    pub fn iter_lines(&self) -> <Vec<BoundLine> as IntoIterator>::IntoIter {
        let bound: Vec<_> = self.lines.iter().map(|a| BoundLine(a, self)).collect();
        bound.into_iter()
//        return self.lines.iter().map(|a| BoundLine(a, self));
    }
    pub fn iter_sectors(&self) -> std::slice::Iter<Sector> {
        self.sectors.iter()
    }
    pub fn iter_things(&self) -> std::slice::Iter<Thing> {
        self.things.iter()
    }

    pub fn vertex(&self, handle: Handle<Vertex>) -> &Vertex {
        &self.vertices[handle.0]
    }

    pub fn side(&self, handle: Handle<Side>) -> &Side {
        &self.sides[handle.0]
    }

    pub fn sector(&self, handle: Handle<Sector>) -> &Sector {
        &self.sectors[handle.0]
    }

    pub fn bbox(&self) -> Rect {
        // TODO ah heck, should include Things too
        let points: Vec<_> = self.vertices.iter().map(|v| Point::new(v.x, v.y)).collect();
        Rect::from_points(points.iter())
    }

    pub fn find_player_start(&self) -> Option<Point> {
        self.things.iter()
        .find(|thing| thing.doomednum() == 1)
        .map(|thing| thing.point())
    }

    pub fn sector_to_polygons(&self, s: usize) -> Vec<Vec<Point>> {
        struct Edge<'a> {
            _line: &'a Line,
            _side: &'a Side,
            _facing: Facing,
            v0: &'a Vertex,
            v1: &'a Vertex,
            done: bool,
        }
        // This is just to convince HashMap to hash on the actual reference, not the underlying
        // BareVertex value
        struct VertexRef<'a>(&'a Vertex);
        impl<'a> PartialEq for VertexRef<'a> {
            fn eq(&self, other: &VertexRef) -> bool {
                (self.0 as *const _) == (other.0 as *const _)
            }
        }
        impl<'a> Eq for VertexRef<'a> {}
        impl<'a> std::hash::Hash for VertexRef<'a> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                (self.0 as *const Vertex).hash(state)
            }
        }

        let mut edges = Vec::new();
        let mut vertices_to_edges = HashMap::new();
        // TODO linear scan -- would make more sense to turn the entire map into polygons in one go
        for line in &self.lines {
            let (frontid, backid) = line.side_indices();
            // FIXME need to handle self-referencing sectors, but also 
            if let Some(front) = line.front.map(|h| &self.sides[h.0]) {
                if let Some(back) = line.back.map(|h| &self.sides[h.0]) {
                    if front.sector == back.sector {
                        continue;
                    }
                }
            }

            // TODO seems like a good case for a custom iterator
            for &(facing, sideid) in [(Facing::Front, frontid), (Facing::Back, backid)].iter() {
                if sideid.is_none() {
                    continue;
                }
                // TODO this and the vertices lookups might be bogus and crash...
                let side = &self.sides[sideid.unwrap().0];
                if side.sector.0 == s {
                    let v0 = &self.vertices[line.start.0];
                    let v1 = &self.vertices[line.end.0];
                    let edge = Edge{
                        _line: line,
                        _side: side,
                        _facing: facing,
                        // TODO should these be swapped depending on the line facing?
                        v0,
                        v1,
                        done: false,
                    };
                    edges.push(edge);
                    vertices_to_edges.entry(VertexRef(v0))
                        .or_insert_with(Vec::new)
                        .push(edges.len() - 1);
                    vertices_to_edges.entry(VertexRef(v1))
                        .or_insert_with(Vec::new)
                        .push(edges.len() - 1);
                }
            }
        }

        // Trace sectors by starting at the first side's first vertex and attempting to walk from
        // there
        let mut outlines = Vec::new();
        let mut seen_vertices = HashMap::new();
        while edges.len() > 0 {
            let mut next_vertices = Vec::new();
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
            if next_vertices.is_empty() {
                break;
            }

            let mut outline = Vec::new();
            while next_vertices.len() > 0 {
                let vertices = next_vertices;
                next_vertices = Vec::new();
                for vertex in vertices.iter() {
                    if seen_vertices.contains_key(&VertexRef(vertex)) {
                        continue;
                    }
                    seen_vertices.insert(VertexRef(vertex), true);
                    outline.push(Point::new(vertex.x, vertex.y));

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
                            next_vertices.push(edge.v0);
                        }
                        else if !seen_vertices.contains_key(&VertexRef(edge.v1)) {
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
}

#[derive(Copy, Clone, Debug)]
pub enum Facing {
    Front,
    Back,
}

pub struct Handle<T>(pub usize, PhantomData<*const T>);

// These traits are implemented by hand because #derive'd impls only apply when T implements the
// same trait, but we don't actually own a T, so that bound is unnecessary.
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle(self.0, PhantomData)
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}


impl<T> From<usize> for Handle<T> {
    fn from(index: usize) -> Self {
        Handle(index, PhantomData)
    }
}

trait MapComponent {}

pub struct Thing {
    point: Point,
    doomednum: u32,
}

impl Thing {
    pub fn point(&self) -> Point {
        self.point
    }

    pub fn doomednum(&self) -> u32 {
        self.doomednum
    }
}

pub struct Line {
    index: usize,
    start: Handle<Vertex>,
    end: Handle<Vertex>,
    flags: u32,
    special: usize,
    // TODO doom maps only!
    // TODO should this be...  something more strongly typed to separate 0 meaning none?
    sector_tag: u32,
    front: Option<Handle<Side>>,
    back: Option<Handle<Side>>,
}

impl Line {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn vertex_indices(&self) -> (Handle<Vertex>, Handle<Vertex>) {
        (self.start, self.end)
    }

    pub fn side_indices(&self) -> (Option<Handle<Side>>, Option<Handle<Side>>) {
        (self.front, self.back)
    }

    pub fn has_special(&self) -> bool {
        self.special != 0
    }

    pub fn sector_tag(&self) -> u32 {
        return self.sector_tag;
    }

    pub fn blocks_player(&self) -> bool {
        self.flags & 1 != 0
    }

    pub fn is_one_sided(&self) -> bool {
        self.front.is_some() != self.back.is_some()
    }

    pub fn is_two_sided(&self) -> bool {
        self.front.is_some() && self.back.is_some()
    }
}

// A Line that knows what map it came from, so it can look up its actual sides and vertices
#[derive(Clone, Copy)]
pub struct BoundLine<'a>(&'a Line, &'a Map);
impl<'a> BoundLine<'a> {
    pub fn start(&self) -> &Vertex {
        self.1.vertex(self.0.start)
    }

    pub fn end(&self) -> &Vertex {
        self.1.vertex(self.0.end)
    }

    pub fn front(&self) -> Option<&Side> {
        self.0.front.map(|s| self.1.side(s))
    }

    pub fn back(&self) -> Option<&Side> {
        self.0.back.map(|s| self.1.side(s))
    }

    // TODO these are all delegates, eugh
    pub fn index(&self) -> usize {
        self.0.index()
    }

    pub fn vertex_indices(&self) -> (Handle<Vertex>, Handle<Vertex>) {
        self.0.vertex_indices()
    }

    pub fn side_indices(&self) -> (Option<Handle<Side>>, Option<Handle<Side>>) {
        self.0.side_indices()
    }

    pub fn has_special(&self) -> bool {
        self.0.has_special()
    }

    pub fn sector_tag(&self) -> u32 {
        self.0.sector_tag()
    }

    pub fn blocks_player(&self) -> bool {
        self.0.blocks_player()
    }

    pub fn is_one_sided(&self) -> bool {
        self.0.is_one_sided()
    }

    pub fn is_two_sided(&self) -> bool {
        self.0.is_two_sided()
    }
}

pub struct Sector {
    tag: u32,
    special: u32,

    floor_height: i32,
    ceiling_height: i32,
    light: u8,
}

impl Sector {
    pub fn tag(&self) -> u32 {
        self.tag
    }

    pub fn special(&self) -> u32 {
        self.special
    }

    pub fn floor_height(&self) -> i32 {
        self.floor_height
    }

    pub fn ceiling_height(&self) -> i32 {
        self.ceiling_height
    }

    pub fn light(&self) -> u8 {
        return self.light;
    }
}

pub struct Side {
    //map: Rc<Map>,
    pub id: u32,
    pub upper_texture: String,
    pub lower_texture: String,
    pub middle_texture: String,
    pub sector: Handle<Sector>,
}

#[derive(Clone, Copy)]
pub struct BoundSide<'a>(&'a Side, &'a Map);
impl<'a> BoundSide<'a> {
    //pub fn sector(&self) -> 
}

pub struct Vertex {
    pub x: f64,
    pub y: f64,
}
