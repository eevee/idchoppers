use std::io::{self, Read, Write};

extern crate idchoppers;
use idchoppers::errors::Result;
extern crate svg;
use svg::Document;
use svg::node::Node;
use svg::node::element::{Circle, Group, Line, Path, Rectangle, Style};
use svg::node::element::path::Data;
extern crate termcolor;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

fn main() {
    match run() {
        Ok(()) => {}
        Err(err) => {
            let mut stderr = StandardStream::stderr(ColorChoice::Auto);
            stderr.set_color(ColorSpec::new().set_fg(Some(Color::Red)).set_bold(true));
            write!(&mut stderr, "error: ");
            stderr.set_color(&ColorSpec::new());
            writeln!(&mut stderr, "{}", err);
        }
    }
}

fn run() -> Result<()> {
    let mut buf = Vec::new();
    io::stdin().read_to_end(&mut buf);

    let wad = try!(idchoppers::parse_wad(buf.as_slice()));
    println!("found {:?}, {:?}, {:?}", wad.header.identification, wad.header.numlumps, wad.header.infotableofs);
    for map_range in wad.iter_maps() {
        let bare_map = try!(idchoppers::parse_doom_map(&wad, &map_range));
        match bare_map {
            // TODO interesting diagnostic: mix of map formats in the same wad
            idchoppers::BareMap::Doom(map) => {
                println!("{} - Doom format map", map_range.name);
            }
            idchoppers::BareMap::Hexen(map) => {
                println!("{} - Hexen format map", map_range.name);
            }
        }
    }

    Ok(())
}

fn write_bare_map_as_svg<L: idchoppers::BareBinaryLine, T: idchoppers::BareBinaryThing>(map: &idchoppers::BareBinaryMap<L, T>) {
    let mut group = Group::new();
    let mut minx;
    let mut miny;
    if let Some(vertex) = map.vertices.first() {
        minx = vertex.x;
        miny = vertex.y;
    }
    else if let Some(thing) = map.things.first() {
        let (x, y) = thing.coords();
        minx = x;
        miny = y;
    }
    else {
        minx = 0;
        miny = 0;
    }
    let mut maxx = minx;
    let mut maxy = miny;

    for vertex in map.vertices.iter() {
        if vertex.x < minx {
            minx = vertex.x;
        }
        if vertex.x > maxx {
            maxx = vertex.x;
        }
        if vertex.y < miny {
            miny = vertex.y;
        }
        if vertex.y > maxy {
            maxy = vertex.y;
        }
    }

    let mut classes = Vec::new();
    for line in map.lines.iter() {
        classes.clear();
        let (v0i, v1i) = line.vertex_indices();
        let v0 = &map.vertices[v0i as usize];
        let v1 = &map.vertices[v1i as usize];
        classes.push("line");
        let (frontid, backid) = line.side_indices();
        if frontid == -1 && backid == -1 {
            classes.push("zero-sided");
        }
        else if frontid == -1 || backid == -1 {
            classes.push("one-sided");
        }
        else {
            classes.push("two-sided");
        }
        if line.has_special() {
            classes.push("has-special");
        }

        group.append(
            Line::new()
            .set("x1", v0.x)
            .set("y1", v0.y)
            .set("x2", v1.x)
            .set("y2", v1.y)
            .set("class", classes.join(" "))
        );
    }

    for thing in map.things.iter() {
        let (x, y) = thing.coords();
        if x < minx {
            minx = x;
        }
        if x > maxx {
            maxx = x;
        }
        if y < miny {
            miny = y;
        }
        if y > maxy {
            maxy = y;
        }
        let (color, radius);
        if let Some(thing_type) = idchoppers::universe::lookup_thing_type(thing.doomednum() as u32) {
            color = match thing_type.category {
                idchoppers::universe::ThingCategory::PlayerStart(_) => "green",
                idchoppers::universe::ThingCategory::Monster => "red",
            };
            radius = thing_type.radius;
        }
        else {
            color = "gray";
            radius = 8;
        }
        group.append(
            Rectangle::new()
            .set("x", x - (radius as i16))
            .set("y", y - (radius as i16))
            .set("width", radius * 2)
            .set("height", radius * 2)
            .set("fill", color));
    }

    for (s, sector) in map.sectors.iter().enumerate() {
        let mut data = Data::new();
        let polys = map.sector_to_polygons(s);
        // TODO wait, hang on, can i have multiple shapes in one path?  i think so...
        for poly in polys.iter() {
            let (first, others) = poly.split_first().unwrap();
            data = data.move_to((first.x, first.y));
            for vertex in others.iter() {
                data = data.line_to((vertex.x, vertex.y));
            }
            data = data.line_to((first.x, first.y));
        }

        let mut path = Path::new().set("d", data);
        if sector.sector_tag != 0 {
            path.assign("data-sector-tag", sector.sector_tag);
        }
        classes.clear();
        classes.push("sector");
        if sector.sector_type == 9 {
            classes.push("secret");
        }
        path.assign("class", classes.join(" "));
        group.append(path);
    }

    // Doom's y-axis points up, but SVG's points down.  Rather than mucking with coordinates
    // everywhere we write them, just flip the entire map.  (WebKit doesn't support "transform" on
    // the <svg> element, hence the need for this group.)
    group.assign("transform", "scale(1 -1)");
    let doc = Document::new()
        .set("viewBox", (minx, -maxy, maxx - minx, maxy - miny))
        .add(Style::new(include_str!("map-svg.css")))
        .add(group);
    svg::save("idchoppers-temp.svg", &doc).unwrap();
}
