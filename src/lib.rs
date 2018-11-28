extern crate byteorder;
extern crate euclid;
extern crate typed_arena;
extern crate bit_vec;
extern crate memmap;
#[macro_use]
extern crate nom;
#[macro_use]
extern crate error_chain;
extern crate svg;  // TODO temp for debugging

pub mod archive;
pub mod errors;
pub mod geom;
pub mod input_buffer;
pub mod map;
pub mod parse;
pub mod universe;
pub mod shapeops;
mod util;
mod vanilladoom;

// TODO yeah probably not.  also: consider renaming from BareX to XData, since it's supposed to be
// just a dumb hunk of data.  also also: cut up sector_to_polygons into a few pieces.  and make a
// SimplePolygon type, and use it in shapeops instead of Contour hey why not.
pub use ::parse::map::{BareMap, BareBinaryLine, BareBinaryThing, BareBinaryMap, parse_doom_map, parse_doom_map_from_archive};
pub use ::parse::texturex::parse_texturex_names;
pub use ::parse::wad::parse_wad;
pub use ::map::MapName;
pub use ::archive::wad::BareWAD;
pub use ::archive::wad::BareWADDirectoryEntry;

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





trait WAD {

}

// TODO i might want wad namespace /or/ actual filetype
// TODO i might determine those via actual vanilla doom rules (what it SHOULD be) or inspection (what it IS)
// TODO zdoom has different rules for guessing what stuff is
enum EntryType {
    Flat,
    Sprite,
    Patch,
}
