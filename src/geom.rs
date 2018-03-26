use euclid::TypedPoint2D;
use euclid::TypedRect;
use euclid::TypedSize2D;

pub struct MapSpace;
pub type Coord = f64;
pub type Point = TypedPoint2D<Coord, MapSpace>;
pub type Rect = TypedRect<Coord, MapSpace>;
pub type Size = TypedSize2D<Coord, MapSpace>;
