use std::io::{self, Read};

extern crate idchoppers;
extern crate svg;
use svg::Document;
use svg::node::Node;
use svg::node::element::{Circle, Line};

fn main() {
    let mut buf = Vec::new();
    io::stdin().read_to_end(&mut buf);

    match idchoppers::parse_wad(buf.as_slice()) {
        Ok(wad) => {
            println!("found {:?}, {:?}, {:?}", wad.header.identification, wad.header.numlumps, wad.header.infotableofs);
            for map_range in wad.iter_maps() {
                match idchoppers::parse_doom_map(&wad, &map_range) {
                    Ok(bare_map) => {
                        write_bare_map_as_svg(&bare_map);
                        println!("wrote a map");
                    }
                    Err(err) => {
                        println!("oh noooo got an error {:?}", err);
                    }
                }
                break;
            }
        }
        Err(err) => {
            println!("oh no, an error {:?}", err.description());
        }
    }
}

fn write_bare_map_as_svg(map: &idchoppers::BareDoomMap) {
    let mut doc = Document::new();
    let mut minx = 0;
    let mut miny = 0;
    let mut maxx = 0;
    let mut maxy = 0;
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

    for line in map.lines.iter() {
        let v0 = &map.vertices[line.v0 as usize];
        let v1 = &map.vertices[line.v1 as usize];
        let classname;
        if line.front_sidedef == -1 || line.back_sidedef == -1 {
            classname = "onesided";
        }

        doc.append(Line::new().set("x1", v0.x).set("y1", v0.y).set("x2", v1.x).set("y2", v1.y).set("stroke-width", 1).set("stroke", "black"));
    }

    for thing in map.things.iter() {
        if thing.x < minx {
            minx = thing.x;
        }
        if thing.x > maxx {
            maxx = thing.x;
        }
        if thing.y < miny {
            miny = thing.y;
        }
        if thing.y > maxy {
            maxy = thing.y;
        }
        doc.append(Circle::new().set("cx", thing.x).set("cy", thing.y).set("r", 16).set("fill", "red"));
    }
    doc.assign("viewBox", (minx, miny, maxx - minx, maxy - miny));
    svg::save("idchoppers-temp.svg", &doc).unwrap();
}
