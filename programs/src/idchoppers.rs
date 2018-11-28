use std::fs::File;
use std::io::Write;
use std::iter::repeat;

extern crate byteorder;
use byteorder::{LittleEndian, WriteBytesExt};
extern crate svg;
use svg::Document;
use svg::node::Node;
use svg::node::element::{Group, Line, Path, Rectangle, Style};
use svg::node::element::path::Data;
extern crate termcolor;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
#[macro_use]
extern crate clap;

extern crate idchoppers;
use idchoppers::errors::{Error, Result};
use idchoppers::input_buffer::InputBuffer;
use idchoppers::map::Map;

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
    let filename = args.value_of("file").unwrap();

    let input = match filename {
        "-" => InputBuffer::new_from_stdin()?,
        f => InputBuffer::new_from_file(f)?,
    };

    let wad = idchoppers::parse_wad(input.bytes())?;

    // Dispatch!
    match args.subcommand() {
        ("info", Some(subargs)) /* | (_, None) */ => { do_info(&args, &subargs, &wad)? },
        ("chart", Some(subargs)) => { do_chart(&args, &subargs, &wad)? },
        ("flip", Some(subargs)) => { do_flip(&args, &subargs, &wad)? },
        ("shapeops", Some(_subargs)) => { do_shapeops()? },
        ("route", Some(subargs)) => { do_route(&args, &subargs, &wad)? },
        _ => { println!("????"); /* TODO bogus */ },
    }

    Ok(())
}

fn do_info(_args: &clap::ArgMatches, _subargs: &clap::ArgMatches, wad: &idchoppers::BareWAD) -> Result<()> {
    println!("{:?}", wad.header.identification);

    println!("found {:?}, {:?}, {:?}", wad.header.identification, wad.header.numlumps, wad.header.infotableofs);

    // TODO low-level wad diagnostics

    // for UNDERSEA, i would want to know:
    // - this is for 1/ultimate
    // - it goes in map slot E2M1
    //   - coop is supported
    //   - skill levels are supported
    //   - deathmatch is NOT supported
    //   - there are no missing textures or unrecognized things
    //   - summary of ammo, health, armor, critters
    //   - detail level (texture variety, amount of big solid walls, variety in wall angles, decor...)
    //   - looks like a kind of simple but decently-made doom 1 era map
    // - it replaces the E2M1 music
    // - there are no other lump replacements
    // - it's vanilla compatible
    // - oddity: there are numerous empty and nameless extra lumps
    // for DYSTOPIA, i would want to know:
    // - this is for 2
    // - it goes in map slot MAP01
    //   - missing some textures!
    //   - ... etc ...
    // - no other lump replacements
    // - it's vanilla compatible
    // for OUTSIDE2:
    // - this is for 2
    // - oddity: it uses map slots 1 and 21?
    // - adds some new stuff, etc...

    let archive = wad.to_archive();

    // so, what does the api look like here?  i feel like i want something that iterates over
    // either single entries or maps, for starters.  but i want to skip markers, yet know the
    // namespace something goes in.
    // (and for "info" purposes, i do also want to know if e.g. the markers are misaligned, AND i
    // want to try to fix it if possible!  how does this fit into that world?)
    use idchoppers::archive::wad::Item;
    for item in &archive {
        match item {
            Item::Map(map_block) => {
                // use std::cmp::Ordering::Equal;

                let bare_map = idchoppers::parse_doom_map_from_archive(&map_block)?;
                match bare_map {
                    // TODO interesting diagnostic: mix of map formats in the same wad
                    idchoppers::BareMap::Doom(map) => {
                        let _full_map = idchoppers::map::Map::from_bare(&map);
                        println!("");
                        println!("{} - Doom format map", map_block.name);
                        /*
                        let texture_counts = map.count_textures();
                        let mut pairs: Vec<_> = texture_counts.iter().collect();
                        pairs.sort_by(|&(_, &(_, area0)), &(_, &(_, area1))| area0.partial_cmp(&area1).unwrap_or(Equal));
                        for &(name, &(count, area)) in pairs.iter().rev() {
                            println!("{:8} - {} uses, total area {} ≈ {} tiles", name, count, area, area / (64.0 * 64.0));
                        }
                        */
                        for _line in map.lines {

                        }
                    }
                    idchoppers::BareMap::Hexen(_map) => {
                        println!("{} - Hexen format map", map_block.name);
                    }
                }
            }
            Item::Entry(entry) => {
                println!("{} - lump", entry.name);
            }
        }
    }

    // FIXME this also catches F1_START etc, dammit
    for entry in wad.iter_entries_between("F_START", "F_END") {
        println!("{}", entry.name);
    }
    println!("---");

    // TODO even if there are new textures, they can't be used in vanilla without a texture lump
    // TODO also in vanilla, the lumps are /patches/ (P namespace) and are meaningless without
    // textures, so i should really be consulting this
    // TODO check against vanilla game's textures, see if any are missing!  but that only applies
    // to vanilla; i /think/ zdoom (others?) can combine multiple texture lumps
    let texture_entries =
        if let Some(texbuf) = wad.first_entry("TEXTURE1").or(wad.first_entry("TEXTURE2")) {
            idchoppers::parse_texturex_names(texbuf)?
        }
        else {
            vec![]
        }
    ;
    // TODO need to determine the right base game first
    // TODO note that an IWAD is its own base game, though it might be marked wrong too!  (what
    // things does an iwad absolutely need?  does it depend on the particular engine?)
    // TODO maybe check if any vanilla textures were /altered/, though that could be tricky
    // TODO repeated texture names in the same lump?
    let texture_names: HashSet<_> = texture_entries.iter().map(|entry| entry.name).collect();
    let stock_texture_names: HashSet<_> = idchoppers::universe::DOOM2_TEXTURES.iter().map(|v| *v).collect();
    for name in texture_names.difference(&stock_texture_names) {
        println!("{}", name);
    }

    Ok(())
}

fn do_chart(_args: &clap::ArgMatches, subargs: &clap::ArgMatches, wad: &idchoppers::BareWAD) -> Result<()> {
    for map_range in wad.iter_maps() {
        match idchoppers::parse_doom_map(&wad, &map_range)? {
            // TODO interesting diagnostic: mix of map formats in the same wad
            idchoppers::BareMap::Doom(map) => {
                let fullmap = Map::from_bare(&map);
                let doc = map_as_svg(&fullmap);
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
    let (mut minx, mut miny) =
        if let Some(vertex) = map.vertices.first() {
            (vertex.x, vertex.y)
        }
        else if let Some(thing) = map.things.first() {
            thing.coords()
        }
        else {
            (0, 0)
        }
    ;
    let (mut maxx, mut maxy) = (minx, miny);

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
        let (top, bottom) =
            if v0.y > v1.y { (v0, v1) }
            else           { (v1, v0) }
        ;
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
    Document::new()
        .set("viewBox", (minx, -maxy, maxx - minx, maxy - miny))
        .add(Style::new(include_str!("map-svg.css")))
        .add(group)
}

fn map_as_svg(map: &Map) -> Document {
    let mut group = Group::new();
    let bbox = map.bbox();

    let mut classes = Vec::new();
    for line in map.iter_lines() {
        classes.clear();
        let v0 = line.start();
        let v1 = line.end();
        classes.push("line");
        let (frontid, backid) = line.side_indices();
        if frontid.is_none() && backid.is_none() {
            classes.push("zero-sided");
        }
        else if frontid.is_none() || backid.is_none() {
            classes.push("one-sided");
        }
        else {
            classes.push("two-sided");
        }
        if line.has_special() {
            classes.push("has-special");
        }
        if line.blocks_player() {
            classes.push("blocking");
        }
        if line.is_two_sided() {
            let front = map.sector(line.front().unwrap().sector);
            let back = map.sector(line.back().unwrap().sector);
            if (front.floor_height() - back.floor_height()).abs() <= 24 && front.ceiling_height().min(back.ceiling_height()) - front.floor_height().max(back.floor_height()) >= 56 {
                classes.push("step");
            }
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
    }

    for thing in map.iter_things() {
        let point = thing.point();
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
            .set("x", point.x - (radius as idchoppers::geom::Coord))
            .set("y", point.y - (radius as idchoppers::geom::Coord))
            .set("width", radius * 2)
            .set("height", radius * 2)
            .set("fill", color));
    }

    for (s, sector) in map.iter_sectors().enumerate() {
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
        if sector.tag() != 0 {
            path.assign("data-sector-tag", sector.tag());
        }
        path.assign("data-sector-light", sector.light());
        path.assign("data-sector-light-color", format!("rgb({}, {}, {})", sector.light(), sector.light(), sector.light()));
        classes.clear();
        classes.push("sector");
        if sector.special() == 9 {
            classes.push("secret");
        }
        for line in map.iter_lines() {
            if /* Some(sector) == line.front().map(|side| map.sector(side.sector))
                || */ line.back().map(|side| side.sector) == Some(s.into())
            {
                // TODO obviously, this, is bad
                if line.has_special() && line.sector_tag() == 0 {
                    classes.push("implicit-tag");
                    break;
                }
            }
        }
        path.assign("class", classes.join(" "));
        group.append(path);
    }

    // Doom's y-axis points up, but SVG's points down.  Rather than mucking with coordinates
    // everywhere we write them, just flip the entire map.  (WebKit doesn't support "transform" on
    // the <svg> element, hence the need for this group.)
    group.assign("transform", "scale(1 -1)");
    Document::new()
        .set("viewBox", (bbox.min_x(), -bbox.max_y(), bbox.size.width, bbox.size.height))
        .add(Style::new(include_str!("map-svg.css")))
        .add(group)
}

fn do_flip(_args: &clap::ArgMatches, _subargs: &clap::ArgMatches, wad: &idchoppers::BareWAD) -> Result<()> {
    let mut buffer = Vec::new();
    let mut directory = Vec::new();
    let mut filepos: usize = 0;
    for map_range in wad.iter_maps() {
        let mut bare_map = idchoppers::parse_doom_map(&wad, &map_range)?;
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
                thing.write_to(&mut buffer)?;
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
                line.write_to(&mut buffer)?;
            }
            directory.push(idchoppers::BareWADDirectoryEntry{
                filepos: (filepos + 12) as u32,
                size: (buffer.len() - filepos) as u32,
                name: "LINEDEFS",
            });
            filepos = buffer.len();

            // Sides
            for side in map.sides.iter_mut() {
                side.write_to(&mut buffer)?;
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
                vertex.write_to(&mut buffer)?;
            }
            directory.push(idchoppers::BareWADDirectoryEntry{
                filepos: (filepos + 12) as u32,
                size: (buffer.len() - filepos) as u32,
                name: "VERTEXES",
            });
            filepos = buffer.len();

            // Sectors
            for sector in map.sectors.iter_mut() {
                sector.write_to(&mut buffer)?;
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
    let mut f = File::create("flipped.wad")?;
    f.write("PWAD".as_bytes())?;
    f.write_u32::<LittleEndian>(directory.len() as u32)?;
    f.write_u32::<LittleEndian>((12 + buffer.len()) as u32)?;
    f.write_all(&buffer[..])?;
    for entry in directory.iter() {
        println!("{:?}", entry);
        f.write_u32::<LittleEndian>(entry.filepos)?;
        f.write_u32::<LittleEndian>(entry.size)?;
        f.write(entry.name.as_bytes())?;
        for _ in entry.name.len() .. 8 {
            f.write(&[0])?;
        }
    }

    Ok(())
}





use idchoppers::shapeops::{self, MapPoint};
fn do_shapeops() -> Result<()> {
    let mut poly1 = shapeops::Polygon::new();
    for points in [
        [(0., 0.), (0., 64.), (32., 64.), (32., 0.)],
        //[(0., 0.), (0., 64.), (64., 64.), (64., 0.)],
        //[(16., 16.), (16., 48.), (48., 48.), (48., 16.)],
        //[(8., 8.), (8., 56.), (56., 56.), (56., 8.)],
        //[(24., 24.), (24., 40.), (40., 40.), (40., 24.)],
        //[(0., 0.), (0., 64.), (64., 64.), (64., 0.)],
        //[(0., 32.), (0., 96.), (64., 96.), (64., 32.)],
    ].iter() {
        let mut contour = shapeops::Contour::new();
        contour.points = points.iter().map(|&(x, y)| shapeops::MapPoint::new(x, y)).collect();
        poly1.contours.push(contour);
    }
    poly1.compute_holes();
    for contour in &poly1.contours {
        println!("contour cw? {} external? {} holes? {:?}", contour.clockwise(), contour.external(), contour.holes);
    }

    let mut poly2 = shapeops::Polygon::new();
    for points in [
        //[(32., 32.), (32., 80.), (80., 32.)],
        // [(56., 32.), (56., 80.), (104., 32.)],
        //[(32., 32.), (32., 48.), (48., 48.), (48., 32.)],
        //[(32., 0.), (32., 64.), (64., 64.), (64., 0.)],
        [(0., 0.), (0., 64.), (64., 64.), (64., 0.)],
        //[(64., 32.), (64., 96.), (128., 96.), (128., 32.)],
        //[(64., 0.), (64., 64.), (128., 64.), (128., 0.)],
    ].iter() {
        let mut contour = shapeops::Contour::new();
        contour.points = points.iter().map(|&(x, y)| shapeops::MapPoint::new(x, y)).collect();
        poly2.contours.push(contour);
    }

    println!("");
    println!("");
    println!("");
    println!("bboxes: {:?}, {:?} / {:?} {:?}", poly1.bbox(), poly2.bbox(), poly1.bbox().intersects(&poly2.bbox()), poly1.bbox().intersection(&poly2.bbox()));
    /*
    println!("ok now my test sweep");
    let results = shapeops::test_sweep(vec![
        (MapPoint::new(0., 0.), MapPoint::new(16., 8.)),
        (MapPoint::new(4., 0.), MapPoint::new(8., 8.)),
        (MapPoint::new(8., 0.), MapPoint::new(12., 8.)),
    ]);
    for pair in results {
        println!("  {:?}", pair);
    }
    */

    let mut poly3 = shapeops::Polygon::new();
    let mut contour = shapeops::Contour::new();
    contour.points = vec![
        MapPoint::new(0., 0.),
        MapPoint::new(64., 0.),
        MapPoint::new(64., 32.),
        MapPoint::new(0., 32.),
    ];
    poly3.contours.push(contour);
    let result = shapeops::compute(&vec![
        (poly1, shapeops::PolygonMode::Normal),
        (poly2, shapeops::PolygonMode::Normal),
        (poly3, shapeops::PolygonMode::Normal)
    ], shapeops::BooleanOpType::Union);

    let bbox = result.bbox();
    let mut doc = Document::new()
        .set("viewBox", (bbox.min_x() - 16., -bbox.max_y() - 16., bbox.size.width + 32., bbox.size.height + 32.))
        .add(Style::new(include_str!("map-svg.css")))
    ;
    //let mut data = Data::new();
    for (i, contour) in result.contours.iter().enumerate() {
        println!("contour #{}: external {:?}, counterclockwise {:?}, holes {:?}, neighbors {:?}", i, contour.external(), contour.counterclockwise(), contour.holes, contour.neighbors);
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
    svg::save("idchoppers-shapeops.svg", &doc)?;

    Ok(())
}


fn do_route(_args: &clap::ArgMatches, subargs: &clap::ArgMatches, wad: &idchoppers::BareWAD) -> Result<()> {
    println!("routing 1");
    for map_range in wad.iter_maps() {
        println!("map: {:?}", map_range.name);
        if let idchoppers::MapName::MAPxx(n) = map_range.name {
            if n < 15 {
                continue;
            }
        }
        let bare_map = idchoppers::parse_doom_map(&wad, &map_range)?;
        match bare_map {
            // TODO interesting diagnostic: mix of map formats in the same wad
            idchoppers::BareMap::Doom(map) => {
                let fullmap = Map::from_bare(&map);
                let doc = route_map_as_svg(&fullmap);
                svg::save(subargs.value_of("outfile").unwrap(), &doc).unwrap();
                break;
            }
            idchoppers::BareMap::Hexen(_map) => {
                //let doc = route_map_as_svg(&map);
                //svg::save(subargs.value_of("outfile").unwrap(), &doc).unwrap();
                println!("(sorry, can't do hexen maps atm)");
                break;
            }
        }
    }

    Ok(())
}

use std::collections::BTreeSet;
use std::collections::HashSet;

const PLAYER_STEP_HEIGHT: i32 = 24;
const PLAYER_HEIGHT: i32 = 56;
const PLAYER_USE_RANGE: i32 = 64;

fn route_map_as_svg(map: &Map) -> Document {
    #[derive(Clone)]
    enum PolygonRef<'a> {
        Sector(idchoppers::map::Handle<idchoppers::map::Sector>),
        Line(idchoppers::map::BoundLine<'a>),
    }


    let mut map_group = Group::new();
    let mut group = Group::new();
    let bbox = map.bbox();

    let mut polygons = Vec::new();
    let mut polygon_refs = Vec::new();
    //let mut polygons = Vec::with_capacity(map.lines.len() + map.sectors.len());

    for (s, _sector) in map.iter_sectors().enumerate() {
        let mut polygon = shapeops::Polygon::new();
        for points in map.sector_to_polygons(s).iter() {
            println!("{} {:?}", s, points);
            let mut contour = shapeops::Contour::new();
            contour.points = points.iter().map(|p| MapPoint::new(p.x as f64, p.y as f64)).collect();
            polygon.contours.push(contour);
        }
        polygons.push((polygon, shapeops::PolygonMode::RemoveEdges));
        polygon_refs.push(PolygonRef::Sector(s.into()));
    }

    for line in map.iter_lines() {
        let mut polygon = shapeops::Polygon::new();
        let mut contour = shapeops::Contour::new();
        let mut classes = vec!["line"];

        let mode;
        if line.is_one_sided() {
            mode = shapeops::PolygonMode::RemoveInterior;
            classes.push("one-sided");
        }
        else if line.is_two_sided() {
            mode = shapeops::PolygonMode::Normal;
            classes.push("two-sided");
        }
        else {
            mode = shapeops::PolygonMode::RemoveInterior;
            classes.push("zero-sided");
        }

        let v0 = line.start();
        let v1 = line.end();
        map_group.append(
            Line::new()
            .set("x1", v0.x)
            .set("y1", v0.y)
            .set("x2", v1.x)
            .set("y2", v1.y)
            .set("class", classes.join(" "))
        );

        let radius = 16.;
        // Always start with the top vertex.  The player is always a square AABB, which yields
        // two cases: down-right or down-left.  (Vertical or horizontal lines can be expressed just
        // as well the same ways, albeit with an extra vertex.)
        let (top, bottom) =
            if v0.y > v1.y { (v0, v1) }
            else           { (v1, v0) }
        ;
        let pt = |x, y| MapPoint::new(x as f64, y as f64);
        if top.x < bottom.x {
            // Down and to the right: start with the bottom-left corner of the top box
            contour.points = vec![
                pt(top.x - radius, top.y - radius),
                pt(top.x - radius, top.y + radius),
                pt(top.x + radius, top.y + radius),
                pt(bottom.x + radius, bottom.y + radius),
                pt(bottom.x + radius, bottom.y - radius),
                pt(bottom.x - radius, bottom.y - radius),
            ];
        }
        else {
            // Down and to the left: start with the top-left corner of the top box
            contour.points = vec![
                pt(top.x - radius, top.y + radius),
                pt(top.x + radius, top.y + radius),
                pt(top.x + radius, top.y - radius),
                pt(bottom.x + radius, bottom.y - radius),
                pt(bottom.x - radius, bottom.y - radius),
                pt(bottom.x - radius, bottom.y + radius),
            ];
        }
        polygon.contours.push(contour);
        polygons.push((polygon, mode));
        polygon_refs.push(PolygonRef::Line(line));
    }

    let start = map.find_player_start().unwrap_or(MapPoint::new(0., 0.));

    let result = shapeops::compute(&polygons, shapeops::BooleanOpType::Union);
    let mut seen_contours = BTreeSet::new();
    let mut next_contours = Vec::new();
    let mut contour_origins = Vec::new();
    for (i, contour) in result.contours.iter().enumerate() {
        if contour.bbox().contains(&start) {
            next_contours.push(i);
            //seen_contours.insert(i);
        }

        let mut origins = Vec::new();
        for (p, from) in contour.from_polygons.iter().enumerate() {
            if ! from {
                continue;
            }
            origins.push(polygon_refs[p].clone());
        }
        contour_origins.push(origins);
    }

    // Floodfill to determine visitability and distance
    // FIXME: clean this up, clean up map interface, this was really hard for me to reread.  change
    // how data gets out of shapeops if necessary, i don't think i'll need it for much else?
    // TODO making this actually work:
    // - need to figure out where switches are hittable from; not 100% sure how to do that, since a
    // switch's reachability is a hemicapsule shape (circles augh!), AND several things can get in the
    // way (solid walls, closed doors, other switches).  this requires some drawing.  on the other
    // hand, i don't actually care about the exact point from which you can hit a switch, only
    // which contours.  (point might matter for timing analysis but that seems like a fool's errand
    // anyway tbh)
    // TODO interesting experiments:
    // - collect reachable places into a traversable "chunk" (tricky bit: a special might alter it
    // later?  do i need to scan for all possible states of all sectors?)
    // - in a chunk, keep track of what sectors (or lines) are preventing us from reaching other
    // chunks
    // - i think the idea is to break the map into a simplified graph of adjacent spaces.  each
    // edge has some set of conditions attached to it (may be whether the target can be moved into
    // /at all/, or may be whether a lift is lowered etc), and each node has some set of switches
    // that can be pressed
    // - so how do we combine contours into nodes?  i have two approaches in mind here:
    // (1) floodfill, as i do below.  check what's reachable, check for switches, try flipping
    // them, repeat until we find the exit.
    // pros: relatively simple
    // cons: not as much information.  requires me to get switches figured out right now, ahem.
    // doesn't work in cases where a switch or line or whatever causes a formerly passable area to
    // become impassable again.
    // (2) "analyze", i.e. split into chunks ahead of time.  i'm not quite sure what this means
    // yet.  consider the lowering walls in map02; how do i decide that those are a single node?
    // or what about the sewer area; that's surely its own node, since it has special ways in and
    // out?  but the starting area isn't.
    // so then is anything with stairs considered all one node?  like some of the buildings in
    // industrial zone?  that's a weird case since it's such a huge area and clearly a separate
    // thing...
    // - speaking of, we need to handle jumping!  i am still genuinely not sure how this could
    // work speedily.

    let mut contour_distance: Vec<_> = repeat(0isize).take(result.contours.len()).collect();
    let mut d: isize = 1;
    let mut chunks = Vec::new();
    let mut contour_tags = Vec::new();
    for (i, contour) in result.contours.iter().enumerate() {
        let mut tags = HashSet::new();
        for (p, from) in result[i].from_polygons.iter().enumerate() {
            if ! from {
                continue;
            }
            match polygon_refs[p] {
                PolygonRef::Sector(sectorh) => {
                    let sector = map.sector(sectorh);
                    let tag = sector.tag();
                    if tag != 0 {
                        tags.insert(tag);
                    }
                }
                PolygonRef::Line(line) => {
                    if let Some(front) = line.front() {
                        let sector = map.sector(front.sector);
                        let tag = sector.tag();
                        if tag != 0 {
                            tags.insert(tag);
                        }
                    }
                    if let Some(back) = line.back() {
                        let sector = map.sector(back.sector);
                        let tag = sector.tag();
                        if tag != 0 {
                            tags.insert(tag);
                        }
                    }
                }
            }
        }
        // FIXME dumb hack for map02, this is a teleport-only tag; really need to do this correctly
        tags.remove(&13);
        contour_tags.push(tags);
    }
    let mut seed_ixs = BTreeSet::new();
    seed_ixs.extend(next_contours.drain(..));
    println!("seeds: {:?}", seed_ixs);
    while ! seed_ixs.is_empty() {
        let &seed = seed_ixs.iter().next().unwrap();
        seed_ixs.remove(&seed);
        if seen_contours.contains(&seed) {
            continue;
        }
        seen_contours.insert(seed);

        let mut chunk = BTreeSet::new();

        next_contours.clear();
        next_contours.push(seed);
        println!("--- beginning chunk {} ---", d);
        while ! next_contours.is_empty() {
            let contours: Vec<_> = next_contours.drain(..).collect();
            for &contour_ix in &contours {
                println!("contour {}, tags {:?}", contour_ix, contour_tags[contour_ix]);
                chunk.insert(contour_ix);
                contour_distance[contour_ix] = d;

                // Figure out which sectors this countour represents (to stand here, the player
                // must be able to fit in ALL these sectors)
                let mut in_sectors = HashSet::new();
                for (p, from) in result[contour_ix].from_polygons.iter().enumerate() {
                    if ! from {
                        continue;
                    }
                    match polygon_refs[p] {
                        PolygonRef::Sector(sector) => {
                            in_sectors.insert(sector);
                        }
                        PolygonRef::Line(line) => {
                            if let Some(front) = line.front() {
                                in_sectors.insert(front.sector);
                            }
                            if let Some(back) = line.back() {
                                in_sectors.insert(back.sector);
                            }
                        }
                    }
                }
                // Figure out the floor height
                // TODO it's possible for the unwraps to panic here if we have an orphan line with no sides
                let highest_floor = in_sectors.iter().map(|&sectorh| map.sector(sectorh).floor_height()).max().unwrap();
                let lowest_floor = in_sectors.iter().map(|&sectorh| map.sector(sectorh).floor_height()).min().unwrap();
                println!("  floor height {}", highest_floor);
                // TODO ceiling, too; requires player height.  currently i check if the target
                // sector is too short, but i don't even check the current sector?

                // Check neighbors
                let mut neighbors = result[contour_ix].neighbors.clone();
                for &hole_ix in &result[contour_ix].holes {
                    neighbors.union(&result[hole_ix].neighbors);
                }
                for (neighbor_ix, touches) in neighbors.iter().enumerate() {
                    if ! touches || seen_contours.contains(&neighbor_ix) {
                        continue;
                    }
                    print!("  neighbor {}, tags {:?} ...", neighbor_ix, contour_tags[neighbor_ix]);

                    let mut ok = true;

                    // If the tags are different, this can't be in the same chunk
                    if contour_tags[contour_ix] != contour_tags[neighbor_ix] {
                        ok = false;
                        println!("NOT OK because sector tags don't match");
                    }

                    // Compute which sectors this neighbor is in
                    let mut in_sectors = HashSet::new();
                    for (p, from) in result[neighbor_ix].from_polygons.iter().enumerate() {
                        if ! from {
                            continue;
                        }
                        match polygon_refs[p] {
                            PolygonRef::Sector(sector) => {
                                in_sectors.insert(sector);
                            }
                            PolygonRef::Line(line) => {
                                if line.blocks_player() {
                                    ok = false;
                                    println!("NOT OK because line blocks player");
                                    break;
                                }
                                if let Some(front) = line.front() {
                                    in_sectors.insert(front.sector);
                                }
                                if let Some(back) = line.back() {
                                    in_sectors.insert(back.sector);
                                }
                            }
                        }
                    }
                    // Check if they can be stepped into
                    // TODO i fear this is too naïve.  what if several contiguous sectors all have
                    // a tag, and a switch can de-chunk them because they have some weird behavior
                    // like "raise to nearest neighbor" that moves them to different heights?
                    // (incidentally, how does that even work if they're also neighbors?)
                    // TODO unwrap could fail
                    let neighbor_highest_floor = in_sectors.iter().map(|&sectorh| map.sector(sectorh).floor_height()).max().unwrap();
                    if neighbor_highest_floor - highest_floor > PLAYER_STEP_HEIGHT
                        || highest_floor - neighbor_highest_floor > PLAYER_STEP_HEIGHT
                        // TODO i don't think this is right; it would unnecessarily split up
                        // multi-sector doors, bars, etc., because AT MAP START none of them
                        // are traversible.  but for static geometry it's clearly correct, and
                        // necessary even, for stuff like sound tunnels.  so what's the right
                        // thing here?
                        // || sector.ceiling_height() - sector.floor_height() < PLAYER_HEIGHT
                    {
                        ok = false;
                        println!("NOT OK because neighbor's floor {} is too far away", neighbor_highest_floor);
                    }

                    if ok {
                        // OK, they're traversible, so same chunk
                        // TODO this doesn't correctly account for one-way drop-offs if we happen
                        // to start from the higher side; should probably ok = false if EITHER
                        // direction doesn't work, so we're forced to find another route
                        // TODO should i track a game-logic view of individual contour neighbors
                        // within a chunk?  i guess i'm gonna need that anyway since teleporters
                        seen_contours.insert(neighbor_ix);
                        next_contours.push(neighbor_ix);
                        println!("OK");
                    }
                    else {
                        seed_ixs.insert(neighbor_ix);
                    }
                }
            }
        }

        d += 1;
        chunks.push(chunk);
        println!("seeds: {:?}", seed_ixs);
    }

    for (i, contour) in result.contours.iter().enumerate() {
        println!("contour #{}: external {:?}, counterclockwise {:?}, holes {:?}, neighbors {:?}", i, contour.external(), contour.counterclockwise(), contour.holes, contour.neighbors.iter().enumerate().filter(|(i, p)| *p).map(|(i, p)| i).collect::<Vec<_>>());

        if ! contour.external() {
            continue;
        }

        let mut data = Data::new();
        let point = contour.points.last().unwrap();
        data = data.move_to((point.x, point.y));
        for point in &contour.points {
            data = data.line_to((point.x, point.y));
        }

        for &hole_id in &contour.holes {
            let hole = &result[hole_id];

            let point = hole.points.last().unwrap();
            data = data.move_to((point.x, point.y));
            for point in &hole.points {
                data = data.line_to((point.x, point.y));
            }
        }

        let mut origin = String::new();
        for ref polyref in &contour_origins[i] {
            origin.push_str(", ");
            match polyref {
                PolygonRef::Sector(sectorh) => origin.push_str(&format!("sector {}", sectorh.0)),
                PolygonRef::Line(line) => origin.push_str(&format!("line {}", line.index())),
            }
        }

        let color;
        let distance = contour_distance[i];
        if distance == -1 {
            color = String::from("red");
        }
        else if distance == 0 {
            color = String::from("darkred");
        }
        else {
            let frac = distance as f64 / d as f64 * 255.;
            //color = format!("rgb({}, {}, {})", frac, frac, frac);
            color = format!("hsl({}, 100%, 75%)", frac/255.*330.);
        }
        group.append(
            Path::new()
            .set("d", data)
            .set("fill", color)
            .set("data-origin", origin)
            .set("data-contour-id", format!("{}", i))
        );
    }

    // Doom's y-axis points up, but SVG's points down.  Rather than mucking with coordinates
    // everywhere we write them, just flip the entire map.  (WebKit doesn't support "transform" on
    // the <svg> element, hence the need for this group.)
    map_group.assign("transform", "scale(1 -1)");
    group.assign("transform", "scale(1 -1)");
    let MARGIN = 32.;
    Document::new()
        .set("viewBox", (bbox.min_x() - MARGIN, -bbox.max_y() - MARGIN, bbox.size.width + MARGIN * 2., bbox.size.height + MARGIN * 2.))
        .add(Style::new(include_str!("map-svg.css")))
        .add(map_group)
        .add(group)
}
