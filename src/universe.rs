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
