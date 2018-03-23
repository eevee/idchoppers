use std::fs::File;
use std::io::{self, Read, Write};
use std::cmp::Ordering::Equal;

extern crate byteorder;
use byteorder::{LittleEndian, WriteBytesExt};
extern crate idchoppers;
use idchoppers::errors::{Error, Result};
extern crate svg;
use svg::Document;
use svg::node::Node;
use svg::node::element::{Group, Line, Path, Rectangle, Style};
use svg::node::element::path::Data;
extern crate termcolor;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
#[macro_use]
extern crate clap;

fn main() {
    match run() {
        Ok(()) => {}
        Err(err) => {
            drop(write_err(err));
        }
    }
}

fn write_err(err: Error) -> Result<()> {
    let mut stderr = StandardStream::stderr(ColorChoice::Auto);
    stderr.set_color(ColorSpec::new().set_fg(Some(Color::Red)).set_bold(true))?;
    write!(&mut stderr, "error: ")?;
    stderr.set_color(&ColorSpec::new())?;
    writeln!(&mut stderr, "{}", err)?;
    if let Some(backtrace) = err.backtrace() {
        writeln!(&mut stderr, "{:?}", backtrace)?;
    }
    Ok(())
}

fn run() -> Result<()> {
    // TODO clap's error output should match mine
    let args = clap_app!(idchoppers =>
        (about: "Parse and manipulates Doom wads")
        (@arg color: -c --color +takes_value "Choose whether to use colored output")
        (@arg file: +required "Input WAD file")
        (@subcommand info =>
            (about: "Print generic information about a WAD or lump")
            (@arg verbose: -v --verbose "Print more information")
        )
        (@subcommand chart =>
            (about: "Render an SVG copy of a map")
            (@arg outfile: +required "Output file")
        )
        (@subcommand shapeops =>
            (about: "test shapeops")
        )
        (@subcommand route =>
            (about: "test routefinding")
            (@arg outfile: +required "Output file")
        )
    ).get_matches();

    // Read input file
    // TODO this won't make sense for creating a new one from scratch...
    // TODO for files that aren't stdin, it would be nice to avoid slurping them all in if not
    // necessary
    let mut buf = Vec::new();
    let filename = args.value_of("file").unwrap();
    if filename == "-" {
        io::stdin().read_to_end(&mut buf)?;
    }
    else {
        let mut file = File::open(filename)?;
        file.read_to_end(&mut buf)?;
    }

    let wad = idchoppers::parse_wad(buf.as_slice())?;

    // Dispatch!
    match args.subcommand() {
        ("info", Some(subargs)) /* | (_, None) */ => { do_info(&args, &subargs, &wad)? },
        ("chart", Some(subargs)) => { do_chart(&args, &subargs, &wad)? },
        ("shapeops", Some(subargs)) => { do_shapeops()? },
        ("route", Some(subargs)) => { do_route(&args, &subargs, &wad)? },
        _ => { println!("????"); /* TODO bogus */ },
    }

    Ok(())
}

fn do_info(args: &clap::ArgMatches, subargs: &clap::ArgMatches, wad: &idchoppers::BareWAD) -> Result<()> {
    match wad.header.identification {
        idchoppers::WADType::IWAD => {
            println!("IWAD");
        }
        idchoppers::WADType::PWAD => {
            println!("PWAD");
        }
    }

    println!("found {:?}, {:?}, {:?}", wad.header.identification, wad.header.numlumps, wad.header.infotableofs);
    for map_range in wad.iter_maps() {
        let bare_map = try!(idchoppers::parse_doom_map(&wad, &map_range));
        match bare_map {
            // TODO interesting diagnostic: mix of map formats in the same wad
            idchoppers::BareMap::Doom(map) => {
                let full_map = idchoppers::map::Map::from_bare(&map);
                println!("");
                println!("{} - Doom format map", map_range.name);
                /*
                let texture_counts = map.count_textures();
                let mut pairs: Vec<_> = texture_counts.iter().collect();
                pairs.sort_by(|&(_, &(_, area0)), &(_, &(_, area1))| area0.partial_cmp(&area1).unwrap_or(Equal));
                for &(name, &(count, area)) in pairs.iter().rev() {
                    println!("{:8} - {} uses, total area {} ≈ {} tiles", name, count, area, area / (64.0 * 64.0));
                }
                */
                for line in map.lines {

                }
            }
            idchoppers::BareMap::Hexen(map) => {
                println!("{} - Hexen format map", map_range.name);
            }
        }
    }

    // FIXME this also catches F1_START etc, dammit
    for entry in wad.iter_entries_between("F_START", "F_END") {
        println!("{}", entry.name);
    }
    println!("---");

    let texture_entries;
    if let Some(texbuf) = wad.first_entry("TEXTURE1") {
        texture_entries = try!(idchoppers::parse_texturex_names(texbuf));
    }
    else if let Some(texbuf) = wad.first_entry("TEXTURE2") {
        texture_entries = try!(idchoppers::parse_texturex_names(texbuf));
    }
    else {
        texture_entries = vec![];
    }
    for entry in texture_entries.iter() {
        println!("{}", entry.name);
    }

    Ok(())
}

fn do_chart(args: &clap::ArgMatches, subargs: &clap::ArgMatches, wad: &idchoppers::BareWAD) -> Result<()> {
    for map_range in wad.iter_maps() {
        let bare_map = try!(idchoppers::parse_doom_map(&wad, &map_range));
        match bare_map {
            // TODO interesting diagnostic: mix of map formats in the same wad
            idchoppers::BareMap::Doom(map) => {
                let doc = bare_map_as_svg(&map);
                svg::save(subargs.value_of("outfile").unwrap(), &doc).unwrap();
                break;
            }
            idchoppers::BareMap::Hexen(map) => {
                let doc = bare_map_as_svg(&map);
                svg::save(subargs.value_of("outfile").unwrap(), &doc).unwrap();
                break;
            }
        }
    }

    Ok(())
}

fn bare_map_as_svg<L: idchoppers::BareBinaryLine, T: idchoppers::BareBinaryThing>(map: &idchoppers::BareBinaryMap<L, T>) -> Document {
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

        // FIXME should annotate with line id
        group.append(
            Line::new()
            .set("x1", v0.x)
            .set("y1", v0.y)
            .set("x2", v1.x)
            .set("y2", v1.y)
            .set("class", classes.join(" "))
        );

        // Draw a temporary outline showing the space each line prevents you from occupying (dilate
        // the world!)
        let mut data = Data::new();
        // TODO can be affected by dehacked, decorate, etc.  also at runtime (oh dear)
        let radius = 16;
        // Always start with the top vertex.  The player is always a square AABB, which yields
        // two cases: down-right or down-left.  (Vertical or horizontal lines can be expressed just
        // as well the same ways, albeit with an extra vertex.)
        let top;
        let bottom;
        if v0.y > v1.y {
            top = v0;
            bottom = v1;
        }
        else {
            top = v1;
            bottom = v0;
        }
        if top.x < bottom.x {
            // Down and to the right: start with the bottom-left corner of the top box
            data = data
            .move_to((top.x - radius, top.y - radius))
            .line_to((top.x - radius, top.y + radius))
            .line_to((top.x + radius, top.y + radius))
            .line_to((bottom.x + radius, bottom.y + radius))
            .line_to((bottom.x + radius, bottom.y - radius))
            .line_to((bottom.x - radius, bottom.y - radius));
        }
        else {
            // Down and to the left: start with the top-left corner of the top box
            data = data
            .move_to((top.x - radius, top.y + radius))
            .line_to((top.x + radius, top.y + radius))
            .line_to((top.x + radius, top.y - radius))
            .line_to((bottom.x + radius, bottom.y - radius))
            .line_to((bottom.x - radius, bottom.y - radius))
            .line_to((bottom.x - radius, bottom.y + radius));
        }
        group.append(
            Path::new()
            .set("d", data)
            .set("class", "line-dilation")
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
                idchoppers::universe::ThingCategory::PlayerStart => "green",
                idchoppers::universe::ThingCategory::Monster => "red",
                idchoppers::universe::ThingCategory::Miscellaneous => "gray",
                _ => "magenta",
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
    return Document::new()
        .set("viewBox", (minx, -maxy, maxx - minx, maxy - miny))
        .add(Style::new(include_str!("map-svg.css")))
        .add(group);
}

fn do_flip(args: &clap::ArgMatches, subargs: &clap::ArgMatches, wad: &idchoppers::BareWAD) -> Result<()> {
    let mut buffer = Vec::new();
    let mut directory = Vec::new();
    let mut filepos: usize = 0;
    for map_range in wad.iter_maps() {
        let mut bare_map = try!(idchoppers::parse_doom_map(&wad, &map_range));
        if let idchoppers::BareMap::Doom(mut map) = bare_map {
            directory.push(idchoppers::BareWADDirectoryEntry{
                filepos: (filepos + 12) as u32,
                size: 0,
                name: wad.directory[map_range.marker_index].name,
            });

            // Things
            for thing in map.things.iter_mut() {
                thing.y = -thing.y;
                thing.angle = -thing.angle;
                thing.write_to(&mut buffer);
            }
            directory.push(idchoppers::BareWADDirectoryEntry{
                filepos: (filepos + 12) as u32,
                size: (buffer.len() - filepos) as u32,
                name: "THINGS",
            });
            filepos = buffer.len();

            // Lines
            for line in map.lines.iter_mut() {
                std::mem::swap(&mut line.v0, &mut line.v1);
                line.write_to(&mut buffer);
            }
            directory.push(idchoppers::BareWADDirectoryEntry{
                filepos: (filepos + 12) as u32,
                size: (buffer.len() - filepos) as u32,
                name: "LINEDEFS",
            });
            filepos = buffer.len();

            // Sides
            for side in map.sides.iter_mut() {
                side.write_to(&mut buffer);
            }
            directory.push(idchoppers::BareWADDirectoryEntry{
                filepos: (filepos + 12) as u32,
                size: (buffer.len() - filepos) as u32,
                name: "SIDEDEFS",
            });
            filepos = buffer.len();

            // Vertices
            for vertex in map.vertices.iter_mut() {
                vertex.y = -vertex.y;
                vertex.write_to(&mut buffer);
            }
            directory.push(idchoppers::BareWADDirectoryEntry{
                filepos: (filepos + 12) as u32,
                size: (buffer.len() - filepos) as u32,
                name: "VERTEXES",
            });
            filepos = buffer.len();

            // Sectors
            for sector in map.sectors.iter_mut() {
                sector.write_to(&mut buffer);
            }
            directory.push(idchoppers::BareWADDirectoryEntry{
                filepos: (filepos + 12) as u32,
                size: (buffer.len() - filepos) as u32,
                name: "SECTORS",
            });
            filepos = buffer.len();

            println!("{} - Doom format map", map_range.name);
        }
    }
    let mut f = try!(File::create("flipped.wad"));
    try!(f.write("PWAD".as_bytes()));
    try!(f.write_u32::<LittleEndian>(directory.len() as u32));
    try!(f.write_u32::<LittleEndian>((12 + buffer.len()) as u32));
    try!(f.write_all(&buffer[..]));
    for entry in directory.iter() {
        println!("{:?}", entry);
        try!(f.write_u32::<LittleEndian>(entry.filepos));
        try!(f.write_u32::<LittleEndian>(entry.size));
        try!(f.write(entry.name.as_bytes()));
        for _ in entry.name.len() .. 8 {
            try!(f.write(&[0]));
        }
    }

    Ok(())
}





use idchoppers::shapeops;
use idchoppers::shapeops::MapPoint;
fn do_shapeops() -> Result<()> {
    let mut poly1 = idchoppers::shapeops::Polygon::new();
    for points in [
        [(0., 0.), (0., 64.), (32., 64.), (32., 0.)],
        //[(0., 0.), (0., 64.), (64., 64.), (64., 0.)],
        //[(16., 16.), (16., 48.), (48., 48.), (48., 16.)],
        //[(8., 8.), (8., 56.), (56., 56.), (56., 8.)],
        //[(24., 24.), (24., 40.), (40., 40.), (40., 24.)],
        //[(0., 0.), (0., 64.), (64., 64.), (64., 0.)],
        //[(0., 32.), (0., 96.), (64., 96.), (64., 32.)],
    ].iter() {
        let mut contour = idchoppers::shapeops::Contour::new();
        contour.points = points.iter().map(|&(x, y)| idchoppers::shapeops::MapPoint::new(x, y)).collect();
        poly1.contours.push(contour);
    }
    poly1.compute_holes();
    for contour in &poly1.contours {
        println!("contour cw? {} external? {} holes? {:?}", contour.clockwise(), contour.external(), contour.holes);
    }

    let mut poly2 = idchoppers::shapeops::Polygon::new();
    for points in [
        //[(32., 32.), (32., 80.), (80., 32.)],
        // [(56., 32.), (56., 80.), (104., 32.)],
        //[(32., 32.), (32., 48.), (48., 48.), (48., 32.)],
        //[(32., 0.), (32., 64.), (64., 64.), (64., 0.)],
        [(0., 0.), (0., 64.), (64., 64.), (64., 0.)],
        //[(64., 32.), (64., 96.), (128., 96.), (128., 32.)],
        //[(64., 0.), (64., 64.), (128., 64.), (128., 0.)],
    ].iter() {
        let mut contour = idchoppers::shapeops::Contour::new();
        contour.points = points.iter().map(|&(x, y)| idchoppers::shapeops::MapPoint::new(x, y)).collect();
        poly2.contours.push(contour);
    }

    println!("");
    println!("");
    println!("");
    println!("bboxes: {:?}, {:?} / {:?} {:?}", poly1.bbox(), poly2.bbox(), poly1.bbox().intersects(&poly2.bbox()), poly1.bbox().intersection(&poly2.bbox()));
    /*
    println!("ok now my test sweep");
    let results = idchoppers::shapeops::test_sweep(vec![
        (MapPoint::new(0., 0.), MapPoint::new(16., 8.)),
        (MapPoint::new(4., 0.), MapPoint::new(8., 8.)),
        (MapPoint::new(8., 0.), MapPoint::new(12., 8.)),
    ]);
    for pair in results {
        println!("  {:?}", pair);
    }
    */

    let mut poly3 = idchoppers::shapeops::Polygon::new();
    let mut contour = idchoppers::shapeops::Contour::new();
    contour.points = vec![
        MapPoint::new(0., 0.),
        MapPoint::new(64., 0.),
        MapPoint::new(64., 32.),
        MapPoint::new(0., 32.),
    ];
    poly3.contours.push(contour);
    let result = idchoppers::shapeops::compute(&vec![poly1, poly2, poly3], idchoppers::shapeops::BooleanOpType::Union);

    let bbox = result.bbox();
    let mut doc = Document::new()
        .set("viewBox", (bbox.min_x() - 16., -bbox.max_y() - 16., bbox.size.width + 32., bbox.size.height + 32.))
        .add(Style::new(include_str!("map-svg.css")))
    ;
    //let mut data = Data::new();
    for (i, contour) in result.contours.iter().enumerate() {
        println!("contour #{}: external {:?}, counterclockwise {:?}, holes {:?}", i, contour.external(), contour.counterclockwise(), contour.holes);
    let mut data = Data::new();
        let point = contour.points.last().unwrap();
        data = data.move_to((point.x, -point.y));
        for point in &contour.points {
            data = data.line_to((point.x, -point.y));
        }
        doc = doc.add(
            Path::new()
            .set("d", data)
            //.set("class", "line")
        )
        .add(
            svg::node::element::Text::new()
            .add(svg::node::Text::new(format!("{}", i)))
            .set("x", contour.points[0].x + 8.)
            .set("y", -contour.points[0].y - 8.)
            .set("text-anchor", "middle")
            .set("alignment-baseline", "central")
            .set("font-size", 8)
        );
    }
    svg::save("idchoppers-shapeops.svg", &doc);

    return Ok(());
}


fn do_route(args: &clap::ArgMatches, subargs: &clap::ArgMatches, wad: &idchoppers::BareWAD) -> Result<()> {
    for map_range in wad.iter_maps() {
        let bare_map = try!(idchoppers::parse_doom_map(&wad, &map_range));
        match bare_map {
            // TODO interesting diagnostic: mix of map formats in the same wad
            idchoppers::BareMap::Doom(map) => {
                let doc = route_map_as_svg(&map);
                svg::save(subargs.value_of("outfile").unwrap(), &doc).unwrap();
                break;
            }
            idchoppers::BareMap::Hexen(map) => {
                let doc = route_map_as_svg(&map);
                svg::save(subargs.value_of("outfile").unwrap(), &doc).unwrap();
                break;
            }
        }
    }

    Ok(())
}

fn route_map_as_svg<L: idchoppers::BareBinaryLine, T: idchoppers::BareBinaryThing>(map: &idchoppers::BareBinaryMap<L, T>) -> Document {
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

    let mut polygons = Vec::with_capacity(map.lines.len() + map.sectors.len());
    //let mut polygons = Vec::with_capacity(map.lines.len() + map.sectors.len());

    for (s, sector) in map.sectors.iter().enumerate() {
        let mut polygon = idchoppers::shapeops::Polygon::new();
        for points in map.sector_to_polygons(s).iter() {
            println!("{} {:?}", s, points);
            let mut contour = idchoppers::shapeops::Contour::new();
            contour.points = points.iter().map(|p| MapPoint::new(p.x as f64, p.y as f64)).collect();
            polygon.contours.push(contour);
        }
        polygons.push(polygon);
    }

    for line in map.lines.iter() {
        let mut polygon = idchoppers::shapeops::Polygon::new();
        let mut contour = idchoppers::shapeops::Contour::new();

        let (v0i, v1i) = line.vertex_indices();
        let v0 = &map.vertices[v0i as usize];
        let v1 = &map.vertices[v1i as usize];
        let radius = 16;
        // Always start with the top vertex.  The player is always a square AABB, which yields
        // two cases: down-right or down-left.  (Vertical or horizontal lines can be expressed just
        // as well the same ways, albeit with an extra vertex.)
        let top;
        let bottom;
        if v0.y > v1.y {
            top = v0;
            bottom = v1;
        }
        else {
            top = v1;
            bottom = v0;
        }
        let Pt = |x, y| MapPoint::new(x as f64, y as f64);
        if top.x < bottom.x {
            // Down and to the right: start with the bottom-left corner of the top box
            contour.points = vec![
                Pt(top.x - radius, top.y - radius),
                Pt(top.x - radius, top.y + radius),
                Pt(top.x + radius, top.y + radius),
                Pt(bottom.x + radius, bottom.y + radius),
                Pt(bottom.x + radius, bottom.y - radius),
                Pt(bottom.x - radius, bottom.y - radius),
            ];
        }
        else {
            // Down and to the left: start with the top-left corner of the top box
            contour.points = vec![
                Pt(top.x - radius, top.y + radius),
                Pt(top.x + radius, top.y + radius),
                Pt(top.x + radius, top.y - radius),
                Pt(bottom.x + radius, bottom.y - radius),
                Pt(bottom.x - radius, bottom.y - radius),
                Pt(bottom.x - radius, bottom.y + radius),
            ];
        }
        polygon.contours.push(contour);
        polygons.push(polygon);
    }

    let result = idchoppers::shapeops::compute(&polygons, idchoppers::shapeops::BooleanOpType::Union);
    for (i, contour) in result.contours.iter().enumerate() {
        println!("contour #{}: external {:?}, counterclockwise {:?}, holes {:?}", i, contour.external(), contour.counterclockwise(), contour.holes);
        let mut data = Data::new();
        let point = contour.points.last().unwrap();
        data = data.move_to((point.x, point.y));
        for point in &contour.points {
            data = data.line_to((point.x, point.y));
        }
        group.append(Path::new().set("d", data));
    }

    // Doom's y-axis points up, but SVG's points down.  Rather than mucking with coordinates
    // everywhere we write them, just flip the entire map.  (WebKit doesn't support "transform" on
    // the <svg> element, hence the need for this group.)
    group.assign("transform", "scale(1 -1)");
    return Document::new()
        .set("viewBox", (minx, -maxy, maxx - minx, maxy - miny))
        .add(Style::new(include_str!("map-svg.css")))
        .add(group);
}
