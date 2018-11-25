pub mod map;
pub mod texturex;
pub mod wad;

mod util;

use std::str::{self, FromStr};
use std::u8;

use nom::{is_digit, le_u8};

use self::util::naive_eof;
use ::map::MapName;


// Map name parsing -- doesn't clearly belong anywhere in particular

named!(exmy_map_name<MapName>, do_parse!(
    tag!(b"E") >>
    e: verify!(le_u8, is_digit) >>
    tag!(b"M") >>
    m: verify!(le_u8, is_digit) >>
    naive_eof >>
    (MapName::ExMy(e - b'0', m - b'0'))
));

named!(mapxx_map_name<MapName>, do_parse!(
    tag!(b"MAP") >>
    xx: verify!(
        map_res!(
            map_res!(
                take!(2),
                str::from_utf8
            ),
            u8::from_str
        ),
        |v| v >= 1 && v <= 32
    ) >>
    naive_eof >>
    (MapName::MAPxx(xx))
));

named!(pub vanilla_map_name<MapName>, alt!(exmy_map_name | mapxx_map_name));
