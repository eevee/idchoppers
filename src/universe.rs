/// For lack of a better name, a "universe" is the set of everything that can exist within a
/// running Doom-engine game.  This includes actor types, sprites, textures, flats, and the like,
/// as well as configuration like MAPINFO and LOCKDEFS, plus behavioral properties of the engine
/// itself.  If it's not part of a map, it's part of a Universe.
pub struct Universe {
}



#[derive(Copy, Clone, Debug)]
pub enum ThingCategory {
    PlayerStart(u8),
    Monster,
}

pub struct ThingType {
    pub doomednum: u32,
    pub radius: u32,
    pub height: u32,

    pub category: ThingCategory,
    pub zdoom_actor_class: &'static str,
}

pub static TEMP_DOOM_THING_TYPES: [ThingType; 4] = [
    ThingType{
        doomednum: 3004,
        radius: 20,
        height: 56,
        category: ThingCategory::Monster,
        zdoom_actor_class: "ZombieMan",
    },
    ThingType{
        doomednum: 9,
        radius: 20,
        height: 56,
        category: ThingCategory::Monster,
        zdoom_actor_class: "ShotgunGuy",
    },
    ThingType{
        doomednum: 3001,
        radius: 20,
        height: 56,
        category: ThingCategory::Monster,
        zdoom_actor_class: "DoomImp",
    },
    ThingType{
        doomednum: 1,
        radius: 16,
        height: 56,
        category: ThingCategory::PlayerStart(1),
        zdoom_actor_class: "Player1Start",
    },
];

pub fn lookup_thing_type(doomednum: u32) -> Option<&'static ThingType> {
    for thing_type in TEMP_DOOM_THING_TYPES.iter() {
        if thing_type.doomednum == doomednum {
            return Some(thing_type);
        }
    }
    return None;
}

pub static DOOM2_FLATS: [&'static str; 147] = [
    "FLOOR0_1", "FLOOR0_3", "FLOOR0_6", "FLOOR1_1", "FLOOR1_7", "FLOOR3_3", "FLOOR4_1", "FLOOR4_5",
    "FLOOR4_6", "FLOOR4_8", "FLOOR5_1", "FLOOR5_2", "FLOOR5_3", "FLOOR5_4", "STEP1", "STEP2",
    "FLOOR6_1", "FLOOR6_2", "TLITE6_1", "TLITE6_4", "TLITE6_5", "TLITE6_6", "FLOOR7_1", "FLOOR7_2",
    "MFLR8_1", "DEM1_1", "DEM1_2", "DEM1_3", "DEM1_4", "CEIL3_1", "CEIL3_2", "CEIL3_5", "CEIL4_2",
    "CEIL4_3", "CEIL5_1", "CEIL5_2", "FLAT1", "FLAT2", "FLAT5", "FLAT10", "FLAT14", "FLAT18",
    "FLAT20", "FLAT22", "FLAT23", "FLAT5_4", "FLAT5_5", "CONS1_1", "CONS1_5", "CONS1_7", "NUKAGE1",
    "NUKAGE2", "NUKAGE3", "F_SKY1", "SFLR6_1", "SFLR6_4", "SFLR7_1", "SFLR7_4", "FLOOR0_2",
    "FLOOR0_5", "FLOOR0_7", "FLOOR1_6", "GATE1", "GATE2", "GATE3", "GATE4", "FWATER1", "FWATER2",
    "FWATER3", "FWATER4", "LAVA1", "LAVA2", "LAVA3", "LAVA4", "DEM1_5", "DEM1_6", "MFLR8_2",
    "MFLR8_3", "MFLR8_4", "CEIL1_1", "CEIL1_2", "CEIL1_3", "CEIL3_3", "CEIL3_4", "CEIL3_6",
    "CEIL4_1", "BLOOD1", "BLOOD2", "BLOOD3", "FLAT1_1", "FLAT1_2", "FLAT1_3", "FLAT5_1", "FLAT5_2",
    "FLAT5_3", "FLAT5_6", "FLAT5_7", "FLAT5_8", "CRATOP1", "CRATOP2", "FLAT3", "FLAT4", "FLAT8",
    "FLAT9", "FLAT17", "FLAT19", "COMP01", "GRASS1", "GRASS2", "GRNLITE1", "GRNROCK", "RROCK01",
    "RROCK02", "RROCK03", "RROCK04", "RROCK05", "RROCK06", "RROCK07", "RROCK08", "RROCK09",
    "RROCK10", "RROCK11", "RROCK12", "RROCK13", "RROCK14", "RROCK15", "RROCK16", "RROCK17",
    "RROCK18", "RROCK19", "RROCK20", "SLIME01", "SLIME02", "SLIME03", "SLIME04", "SLIME05",
    "SLIME06", "SLIME07", "SLIME08", "SLIME09", "SLIME10", "SLIME11", "SLIME12", "SLIME13",
    "SLIME14", "SLIME15", "SLIME16",
];

pub static DOOM2_TEXTURES: [&'static str; 428] = [
    "AASHITTY", "ASHWALL2", "ASHWALL3", "ASHWALL4", "ASHWALL6", "ASHWALL7", "BFALL1", "BFALL2",
    "BFALL3", "BFALL4", "BIGBRIK1", "BIGBRIK2", "BIGBRIK3", "BIGDOOR1", "BIGDOOR2", "BIGDOOR3",
    "BIGDOOR4", "BIGDOOR5", "BIGDOOR6", "BIGDOOR7", "BLAKWAL1", "BLAKWAL2", "BLODRIP1", "BLODRIP2",
    "BLODRIP3", "BLODRIP4", "BRICK1", "BRICK10", "BRICK11", "BRICK12", "BRICK2", "BRICK3",
    "BRICK4", "BRICK5", "BRICK6", "BRICK7", "BRICK8", "BRICK9", "BRICKLIT", "BRNPOIS", "BRNSMAL1",
    "BRNSMAL2", "BRNSMALC", "BRNSMALL", "BRNSMALR", "BRONZE1", "BRONZE2", "BRONZE3", "BRONZE4",
    "BROVINE2", "BROWN1", "BROWN144", "BROWN96", "BROWNGRN", "BROWNHUG", "BROWNPIP", "BRWINDOW",
    "BSTONE1", "BSTONE2", "BSTONE3", "CEMENT1", "CEMENT2", "CEMENT3", "CEMENT4", "CEMENT5",
    "CEMENT6", "CEMENT7", "CEMENT8", "CEMENT9", "COMPBLUE", "COMPSPAN", "COMPSTA1", "COMPSTA2",
    "COMPTALL", "COMPWERD", "CRACKLE2", "CRACKLE4", "CRATE1", "CRATE2", "CRATE3", "CRATELIT",
    "CRATINY", "CRATWIDE", "DBRAIN1", "DBRAIN2", "DBRAIN3", "DBRAIN4", "DOOR1", "DOOR3", "DOORBLU",
    "DOORBLU2", "DOORRED", "DOORRED2", "DOORSTOP", "DOORTRAK", "DOORYEL", "DOORYEL2", "EXITDOOR",
    "EXITSIGN", "EXITSTON", "FIREBLU1", "FIREBLU2", "FIRELAV2", "FIRELAV3", "FIRELAVA", "FIREMAG1",
    "FIREMAG2", "FIREMAG3", "FIREWALA", "FIREWALB", "FIREWALL", "GRAY1", "GRAY2", "GRAY4", "GRAY5",
    "GRAY7", "GRAYBIG", "GRAYPOIS", "GRAYTALL", "GRAYVINE", "GSTFONT1", "GSTFONT2", "GSTFONT3",
    "GSTGARG", "GSTLION", "GSTONE1", "GSTONE2", "GSTSATYR", "GSTVINE1", "GSTVINE2", "ICKWALL1",
    "ICKWALL2", "ICKWALL3", "ICKWALL4", "ICKWALL5", "ICKWALL7", "LITE3", "LITE5", "LITEBLU1",
    "LITEBLU4", "MARBFAC2", "MARBFAC3", "MARBFAC4", "MARBFACE", "MARBGRAY", "MARBLE1", "MARBLE2",
    "MARBLE3", "MARBLOD1", "METAL", "METAL1", "METAL2", "METAL3", "METAL4", "METAL5", "METAL6",
    "METAL7", "MIDBARS1", "MIDBARS3", "MIDBRN1", "MIDBRONZ", "MIDGRATE", "MIDSPACE", "MODWALL1",
    "MODWALL2", "MODWALL3", "MODWALL4", "NUKE24", "NUKEDGE1", "NUKEPOIS", "PANBLACK", "PANBLUE",
    "PANBOOK", "PANBORD1", "PANBORD2", "PANCASE1", "PANCASE2", "PANEL1", "PANEL2", "PANEL3",
    "PANEL4", "PANEL5", "PANEL6", "PANEL7", "PANEL8", "PANEL9", "PANRED", "PIPE1", "PIPE2",
    "PIPE4", "PIPE6", "PIPES", "PIPEWAL1", "PIPEWAL2", "PLAT1", "REDWALL", "ROCK1", "ROCK2",
    "ROCK3", "ROCK4", "ROCK5", "ROCKRED1", "ROCKRED2", "ROCKRED3", "SFALL1", "SFALL2", "SFALL3",
    "SFALL4", "SHAWN1", "SHAWN2", "SHAWN3", "SILVER1", "SILVER2", "SILVER3", "SK_LEFT", "SK_RIGHT",
    "SKIN2", "SKINCUT", "SKINEDGE", "SKINFACE", "SKINLOW", "SKINMET1", "SKINMET2", "SKINSCAB",
    "SKINSYMB", "SKSNAKE1", "SKSNAKE2", "SKSPINE1", "SKSPINE2", "SKY1", "SKY2", "SKY3", "SLADPOIS",
    "SLADSKUL", "SLADWALL", "SLOPPY1", "SLOPPY2", "SP_DUDE1", "SP_DUDE2", "SP_DUDE4", "SP_DUDE5",
    "SP_DUDE7", "SP_DUDE8", "SP_FACE1", "SP_FACE2", "SP_HOT1", "SP_ROCK1", "SPACEW2", "SPACEW3",
    "SPACEW4", "SPCDOOR1", "SPCDOOR2", "SPCDOOR3", "SPCDOOR4", "STARBR2", "STARG1", "STARG2",
    "STARG3", "STARGR1", "STARGR2", "STARTAN2", "STARTAN3", "STEP1", "STEP2", "STEP3", "STEP4",
    "STEP5", "STEP6", "STEPLAD1", "STEPTOP", "STONE", "STONE2", "STONE3", "STONE4", "STONE5",
    "STONE6", "STONE7", "STUCCO", "STUCCO1", "STUCCO2", "STUCCO3", "SUPPORT2", "SUPPORT3",
    "SW1BLUE", "SW1BRCOM", "SW1BRIK", "SW1BRN1", "SW1BRN2", "SW1BRNGN", "SW1BROWN", "SW1CMT",
    "SW1COMM", "SW1COMP", "SW1DIRT", "SW1EXIT", "SW1GARG", "SW1GRAY", "SW1GRAY1", "SW1GSTON",
    "SW1HOT", "SW1LION", "SW1MARB", "SW1MET2", "SW1METAL", "SW1MOD1", "SW1PANEL", "SW1PIPE",
    "SW1ROCK", "SW1SATYR", "SW1SKIN", "SW1SKULL", "SW1SLAD", "SW1STARG", "SW1STON1", "SW1STON2",
    "SW1STON6", "SW1STONE", "SW1STRTN", "SW1TEK", "SW1VINE", "SW1WDMET", "SW1WOOD", "SW1ZIM",
    "SW2BLUE", "SW2BRCOM", "SW2BRIK", "SW2BRN1", "SW2BRN2", "SW2BRNGN", "SW2BROWN", "SW2CMT",
    "SW2COMM", "SW2COMP", "SW2DIRT", "SW2EXIT", "SW2GARG", "SW2GRAY", "SW2GRAY1", "SW2GSTON",
    "SW2HOT", "SW2LION", "SW2MARB", "SW2MET2", "SW2METAL", "SW2MOD1", "SW2PANEL", "SW2PIPE",
    "SW2ROCK", "SW2SATYR", "SW2SKIN", "SW2SKULL", "SW2SLAD", "SW2STARG", "SW2STON1", "SW2STON2",
    "SW2STON6", "SW2STONE", "SW2STRTN", "SW2TEK", "SW2VINE", "SW2WDMET", "SW2WOOD", "SW2ZIM",
    "TANROCK2", "TANROCK3", "TANROCK4", "TANROCK5", "TANROCK7", "TANROCK8", "TEKBRON1", "TEKBRON2",
    "TEKGREN1", "TEKGREN2", "TEKGREN3", "TEKGREN4", "TEKGREN5", "TEKLITE", "TEKLITE2", "TEKWALL1",
    "TEKWALL4", "TEKWALL6", "WOOD1", "WOOD10", "WOOD12", "WOOD3", "WOOD4", "WOOD5", "WOOD6",
    "WOOD7", "WOOD8", "WOOD9", "WOODGARG", "WOODMET1", "WOODMET2", "WOODMET3", "WOODMET4",
    "WOODVERT", "ZDOORB1", "ZDOORF1", "ZELDOOR", "ZIMMER1", "ZIMMER2", "ZIMMER3", "ZIMMER4",
    "ZIMMER5", "ZIMMER7", "ZIMMER8", "ZZWOLF1", "ZZWOLF10", "ZZWOLF11", "ZZWOLF12", "ZZWOLF13",
    "ZZWOLF2", "ZZWOLF3", "ZZWOLF4", "ZZWOLF5", "ZZWOLF6", "ZZWOLF7", "ZZWOLF9", "ZZZFACE1",
    "ZZZFACE2", "ZZZFACE3", "ZZZFACE4", "ZZZFACE5", "ZZZFACE6", "ZZZFACE7", "ZZZFACE8", "ZZZFACE9",
];
