# idchoppers

idchoppers is a Rust library for parsing and manipulating WAD files and other data associated with classic Doom and Doom-era games.

The idea is to get all this stuff done once and for all in a safe, fast language that everything else can bind to.  (Don't @ me with that xkcd strip.)

It is **extremely** unfinished and a work in progress, unless I get bored with it, in which case it's not.  I never finish anything.  Sorry.  (It does help if people show continued active interest, ahem.)


## Roadmap?

No promises.  I'm just throwing stuff at the wall here.

### The basics

I have a dream that someday [SLADE](http://slade.mancubus.net/) will become a GUI that merely wraps idchoppers, if that gives you a general idea.  (I don't actually have buy-in from SLADE's author, but we can cross that bridge later.)

So, that would require:

- Read and write various kinds of archives (WAD, PK3, directories...)
- Read, write, and convert between various image formats, sound formats, etc.
- Load, render, modify (!), and save maps

### The intermediates

- Understand _all_ of ZDoom's extra lumps, plus DEHACKED
- Detect which game (Doom, Doom II, Heretic, Hexen, Strife, etc.) a PWAD is intended for
- Detect which port (Boom, Boom plus MBF sky transfers, ZDoom, GZDoom, Eternity, lord knows what else) a PWAD is intended for
- Heuristically determine whether jumping/ducking are intended
- Distinguish simple map sets from megawads from TCs
- Extract map names, including using "OCR" on title lumps
- Convert between map formats, including porting all of ZDoom's UDMF features to line specials, or "downgrading" a ZDoom map to Boom or vanilla when possible (and explaining why when not)
- Guess at the overall aesthetic of a map by measuring the most-used textures by surface area
- Detect, fix, and avoid a wide variety of mistakes and errors (unreachable secrets, duplicate secrets, missing keys, slime trails, bogus BLOCKMAP, visplane overflows, placement of the MAP02 sargeant, monster-filled hallways straddling a blockmap edge...  and that's just vanilla!)

### The advanceds

I sort of doubt many of these will happen without some serious third-party interest/assistance, but they're interesting ideas.

- "Compress" resources like `cc4tex.wad` by using ZDoom features to express translations, rotations, flips, etc.
- A good enough software renderer to generate the opening scene of a map
- Estimate the difficulty of maps, based on monster count vs reachable area vs available resources
- Try to find the intended route through maps
- Decompile ACS to trace how scripts affect map geometry and progression
- Generate a par time automatically
- Find "interesting" vantage points for the sake of `wadinfo_txt`
- Be useful enough to base something like Oblige on
