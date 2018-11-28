pub mod wad;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Namespace {
    Unknown,
    Map,
    Sprite,
    Flat,
    //Colormap,
    //ACS,
}

impl Namespace {
}


// TODO do...  these
pub trait Archive {
    // TODO needs a feature atm -- const supports_duplicate_names: bool;
}
