// hey what's up, this is a catastrophic mess ported from a paper:
// A simple algorithm for Boolean operations on polygons (2013), Martínez et al.
// TODO clean this up
// TODO rename it this name is bad
// TODO finish, it?
extern crate svg;
use svg::Document;
use svg::node::Node;
use svg::node::element::{Group, Line, Path, Style, Text};
use svg::node::element::path::Data;



use std::cmp;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;
use std::ops;

use std::cell::Cell;
use std::cell::RefCell;

use std::f64;


use bit_vec::BitVec;
use euclid::TypedPoint2D;
use euclid::TypedRect;
use euclid::TypedSize2D;
use typed_arena::Arena;


const SPEW: bool = false;


use super::geom::MapSpace;
pub type MapPoint = TypedPoint2D<f64, MapSpace>;
pub type MapRect = TypedRect<f64, MapSpace>;
pub type MapSize = TypedSize2D<f64, MapSpace>;
// TODO honestly a lot of this could be made generic over TypedPoint2D

trait MapRectX {
    fn touches(&self, other: &Self) -> bool;
}
impl MapRectX for MapRect {
    fn touches(&self, other: &Self) -> bool {
        self.origin.x <= other.origin.x + other.size.width &&
       other.origin.x <=  self.origin.x + self.size.width &&
        self.origin.y <= other.origin.y + other.size.height &&
       other.origin.y <=  self.origin.y + self.size.height
    }
}

fn compare_points(a: MapPoint, b: MapPoint) -> Ordering {
    a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap())
}


#[derive(Debug, PartialEq)]
struct Segment2 {
    source: MapPoint,
    target: MapPoint,
}

impl Segment2 {
    fn new(source: MapPoint, target: MapPoint) -> Self {
        Self { source, target }
    }

    fn is_vertical(&self) -> bool {
        self.source.x == self.target.x
    }
}


// -----------------------------------------------------------------------------
// utilities

// NOTE: this, and everything else ported from the paper, assumes the y axis points UP
pub fn triangle_signed_area(a: MapPoint, b: MapPoint, c: MapPoint) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y)
}

fn check_span_overlap(u0: f64, u1: f64, v0: f64, v1: f64) -> Option<(f64, f64)> {
    if u1 < v0 || u0 > v1 {
        None
    }
    else if u1 > v0 {
        if u0 < v1 {
            Some((u0.max(v0), u1.min(v1)))
        } else {
            // u0 == v1
            Some((u0, u0))
        }
    }
    else {
        // u1 == v0
        Some((u1, u1))
    }
}

enum SegmentIntersection {
    None,
    Point(MapPoint),
    // TODO one would think this would return a Segment
    Segment(MapPoint, MapPoint),
}

const SQR_EPSILON: f64 = 0.0000001;
const EPSILON: f64 = 0.000000000000001;
fn intersect_segments(seg0: &Segment2, seg1: &Segment2) -> SegmentIntersection {
    let p0 = seg0.source;
    let d0 = seg0.target - p0;
    let p1 = seg1.source;
    let d1 = seg1.target - p1;
    let sep = p1 - p0;
    let kross = d0.cross(d1);
    let d0len2 = d0.square_length();
    let d1len2 = d1.square_length();

    if kross * kross > SQR_EPSILON * d0len2 * d1len2 {
        // Lines containing these segments intersect; check whether the segments themselves do
        let s = sep.cross(d1) / kross;
        if s < 0. || s > 1. {
            return SegmentIntersection::None;
        }
        let t = sep.cross(d0) / kross;
        if t < 0. || t > 1. {
            return SegmentIntersection::None;
        }
        // intersection of lines is a point an each segment
        let mut intersection = p0 + d0 * s;
        // Avoid precision errors by rounding to the nearest segment endpoint
        for &endpoint in [seg0.source, seg0.target, seg1.source, seg1.target].iter() {
            if (intersection - endpoint).square_length() < EPSILON {
                intersection = endpoint;
            }
        }
        // And if that didn't do it, also round almost-integers to integers.  This is extremely
        // hokey, but it takes care of an edge case I ran into: a not-quite-integral intersection
        // point was eventually used to split a vertical line perfectly aligned to the grid, so the
        // resulting lines were no longer vertical, and that changed everything's sort order, and
        // it was a big huge mess.
        // TODO maybe do this below too?
        // TODO i would love a better solution for this, or at least some very simple test cases
        // (which i could then throw at the original implementation to see what it does?)
        // TODO consider replacing f64s with rats?
        if intersection.x.fract().abs() < 1e-9 {
            intersection.x = intersection.x.round();
        }
        if intersection.y.fract().abs() < 1e-9 {
            intersection.y = intersection.y.round();
        }
        return SegmentIntersection::Point(intersection);
    }

    // Segments are parallel; check if they're collinear
    let kross = sep.cross(d0);
    if kross * kross > SQR_EPSILON * d0len2 * sep.square_length() {
        // Nope, no intersection
        return SegmentIntersection::None;
    }

    // Segments are collinear; check whether their endpoints overlap
    let s0 = d0.dot(sep) / d0len2;  // so = Dot (D0, sep) * d0len2
    let s1 = s0 + d0.dot(d1) / d0len2;  // s1 = s0 + Dot (D0, D1) * d0len2
    let smin = s0.min(s1);
    let smax = s0.max(s1);
    if let Some((begin, end)) = check_span_overlap(0., 1., smin, smax) {
        // XXX shouldn't the intersection point always just be one of the endpoints??
        let mut intersection = p0 + d0 * begin;
        // Avoid precision errors by rounding to the nearest segment endpoint
        for &endpoint in [seg0.source, seg0.target, seg1.source, seg1.target].iter() {
            if (intersection - endpoint).square_length() < EPSILON {
                intersection = endpoint;
            }
        }
        if begin == end {
            return SegmentIntersection::Point(intersection);
        }
        else {
            // NOTE: this value is never actually used by the caller, but the fact that it's a
            // segment intersection is
            let intersection2 = p0 + d0 * end;
            return SegmentIntersection::Segment(intersection, intersection2);
        }
    }

    SegmentIntersection::None
}


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BooleanOpType {
    Intersection,
    Union,
    Difference,
    ExclusiveOr,
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum EdgeType {
    Normal,
    NonContributing,
    SameTransition,
    DifferentTransition,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SegmentEnd {
    Left,
    Right,
}


#[derive(Debug)]
struct SweepSegment<T> {
    left_point: MapPoint,
    right_point: MapPoint,
    faces_outwards: bool,
    index: usize,
    order: usize,
    data: T,
}

impl<T> SweepSegment<T> {
    fn new(point0: MapPoint, point1: MapPoint, index: usize, order: usize, data: T) -> SweepSegment<T> {
        if point0 == point1 {
            panic!("Can't create a zero-length segment");
        }

        let (left_point, right_point, faces_outwards) = if point0.x < point1.x || (point0.x == point1.x && point0.y < point1.y) {
            (point0, point1, false)
        }
        else {
            (point1, point0, true)
        };

        SweepSegment{
            left_point,
            right_point,
            // TODO hang on, this is only even used in one place?  in polygon::compute_holes??
            // that seems weird?
            faces_outwards,
            index,
            order,
            data,
        }
    }

    /// Is the line segment (left_point, right_point) below point p
    fn below(&self, p: MapPoint) -> bool {
        triangle_signed_area(self.left_point, self.right_point, p) > 0.0000001
    }

    /// Is the line segment (point, other_point) above point p
    fn above(&self, p: MapPoint) -> bool {
        ! self.below(p)
    }

    /// Is the line segment (point, other_point) a vertical line segment
    fn vertical(&self) -> bool {
        self.left_point.x == self.right_point.x
    }

    /// Return the line segment
    fn segment(&self) -> Segment2 {
        Segment2::new(self.left_point, self.right_point)
    }
}

impl<T> cmp::PartialEq for SweepSegment<T> {
    fn eq(&self, other: &SweepSegment<T>) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<T> cmp::Eq for SweepSegment<T> { }

impl<T> cmp::PartialOrd for SweepSegment<T> {
    fn partial_cmp(&self, other: &SweepSegment<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// This is used for the binary tree ordering in the sweep line algorithm, keeping sweep events in
// order by where on the y-axis they would cross the sweep line
impl<T> cmp::Ord for SweepSegment<T> {
    fn cmp(&self, other: &SweepSegment<T>) -> Ordering {
        if self as *const _ == other as *const _ {
            return Ordering::Equal;
        }

        if triangle_signed_area(self.left_point, self.right_point, other.left_point).abs() < 0.0000001 &&
            triangle_signed_area(self.left_point, self.right_point, other.right_point).abs() < 0.0000001
        {
            // Segments are collinear.  Sort by some arbitrary consistent criteria
            // XXX note that this needs to match the order used in the sweep line!!  i think.
            // XXX this used to be pol, which would cause far more ties...  is this ok?  does
            // "consistent" mean it actually needs to use the sweep comparison?
            return self.order.cmp(&other.order)
                .then_with(||
                    if self.left_point == other.left_point {
                        self.index.cmp(&other.index)
                    }
                    else {
                        Ordering::Equal
                    }
                )
                .then_with(||
                    SweepEndpoint(other, SegmentEnd::Left).cmp(&SweepEndpoint(self, SegmentEnd::Left))
                );
                //.then_with(|| compare_by_sweep(self, other));
            // FIXME this used to do this, not sure what le1 < le2 does though...
            // if (le1->pol != le2->pol)
            //     return le1->pol < le2->pol;
            // // Just a consistent criterion is used
            // if (le1->point == le2->point)
            //     return le1 < le2;
            // SweepEventComp comp;
            // return comp (le1, le2);
        }

        if self.left_point == other.left_point {
            // Both segments have the same left endpoint.  Sort on the right endpoint
            // TODO self.below() just checks triangle_signed_area again, same as above
            if self.below(other.right_point) {
                return Ordering::Less;
            }
            else {
                return Ordering::Greater;
            }
        }

        // has the segment associated to e1 been sorted in evp before the segment associated to e2?
        if SweepEndpoint(self, SegmentEnd::Left) < SweepEndpoint(other, SegmentEnd::Left) {
            if self.below(other.left_point) {
                return Ordering::Less;
            }
            else {
                return Ordering::Greater;
            }
        }
        else {
            // The segment associated to e2 has been sorted in evp before the segment associated to e1
            if other.above(self.left_point) {
                return Ordering::Less;
            }
            else {
                return Ordering::Greater;
            }
        }
    }
}




#[derive(Debug)]
struct SweepEndpoint<'a, T: 'a>(&'a SweepSegment<T>, SegmentEnd);

impl<'a, T: 'a> SweepEndpoint<'a, T> {
    fn point(&self) -> MapPoint {
        match self.1 {
            SegmentEnd::Left => self.0.left_point,
            SegmentEnd::Right => self.0.right_point,
        }
    }
    fn other_point(&self) -> MapPoint {
        match self.1 {
            SegmentEnd::Left => self.0.right_point,
            SegmentEnd::Right => self.0.left_point,
        }
    }
}
impl<'a, T: 'a> cmp::PartialEq for SweepEndpoint<'a, T> {
    fn eq(&self, other: &SweepEndpoint<'a, T>) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<'a, T: 'a> cmp::Eq for SweepEndpoint<'a, T> { }

impl<'a, T: 'a> cmp::PartialOrd for SweepEndpoint<'a, T> {
    fn partial_cmp(&self, other: &SweepEndpoint<'a, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T: 'a> cmp::Ord for SweepEndpoint<'a, T> {
    fn cmp(&self, other: &SweepEndpoint<'a, T>) -> Ordering {
        if self as *const _ == other as *const _ {
            return Ordering::Equal;
        }

        self.point().x.partial_cmp(&other.point().x).unwrap()
            .then(self.point().y.partial_cmp(&other.point().y).unwrap())
            .then_with(|| {
                // If the points coincide, a right endpoint takes priority
                if self.1 == other.1 {
                    Ordering::Equal
                }
                else if self.1 == SegmentEnd::Right {
                    Ordering::Less
                }
                else {
                    Ordering::Greater
                }
            })
            .then_with(
                // Same point, same end of their respective segments.  Use triangle area to give
                // priority to the bottom point
                || triangle_signed_area(other.point(), other.other_point(), self.other_point()).partial_cmp(&0.).unwrap()
            )
            .then_with(
                // Collinear!  Kind of arbitrary what we do here, but it should be consistent.
                // NOTE this used to be pol, unsure if this makes any real difference
                // FIXME currently have the problem where there's a collinear split and one of the
                // pieces ends up changing order in the list, and then we call compute_fields with
                // the same two pieces in both orders, which is nonsense, so...  make sure this
                // doesn't change the ordering after a split!  that could be really hard since this
                // algorithm was designed around mutate-splitting the original...
                //|| other.0.index.cmp(&self.0.index)
                || self.0.order.cmp(&other.0.order)
            )
    }
}

// -----------------------------------------------------------------------------
// polygon

#[derive(Clone)]
pub struct Contour {
    /// Set of points conforming the external contour
    pub points: Vec<MapPoint>,
    /// Holes of the contour. They are stored as the indexes of the holes in a polygon class
    pub holes: Vec<usize>,
    pub from_polygons: BitVec,
    // XXX this is maybe an odd way to go about this
    pub neighbors: BitVec,
    // is the contour an external contour? (i.e., is it not a hole?)
    _external: bool,
    _is_clockwise: Cell<Option<bool>>,
}

impl Contour {
    pub fn new() -> Self {
        Contour{
            points: Vec::new(),
            holes: Vec::new(),
            from_polygons: BitVec::new(),
            neighbors: BitVec::new(),
            _external: true,
            _is_clockwise: Cell::new(None),
        }
    }

    pub fn bbox(&self) -> MapRect {
        if self.points.is_empty() {
            return MapRect::new(MapPoint::zero(), MapSize::zero());
        }
        let mut min_x = self.points[0].x;
        let mut max_x = min_x;
        let mut min_y = self.points[0].y;
        let mut max_y = min_y;
        for point in self.points.iter().skip(1) {
            min_x = f64::min(min_x, point.x);
            max_x = f64::max(max_x, point.x);
            min_y = f64::min(min_y, point.y);
            max_y = f64::max(max_y, point.y);
        }
        MapRect::new(MapPoint::new(min_x, min_y), MapSize::new(max_x - min_x, max_y - min_y))
    }

    // FIXME should this be hidden in a RefCell since it's a cache?  but i want to actively avoid
    // copying.  hm
    pub fn clockwise(&self) -> bool {
        if let Some(ret) = self._is_clockwise.get() {
            return ret;
        }

        let mut area = self.points.last().unwrap().to_vector().cross(self.points[0].to_vector());
        for (vertex0, vertex1) in self.points.iter().zip(self.points.iter().skip(1)) {
            area += vertex0.to_vector().cross(vertex1.to_vector());
        }
        let is_clockwise = area < 0.;
        self._is_clockwise.set(Some(is_clockwise));
        is_clockwise
    }
    pub fn counterclockwise(&self) -> bool {
        ! self.clockwise()
    }

    fn _move_by(&mut self, dx: f64, dy: f64) {
        for point in self.points.iter_mut() {
            *point = MapPoint::new(point.x + dx, point.y + dy);
        }
    }

    /// Get the p-th vertex of the external contour
    fn _vertex(&self, p: usize) -> MapPoint { self.points[p] }

    #[cfg(test)]
    fn segment(&self, p: usize) -> Segment2 {
        if p == self.points.len() - 1 {
            Segment2::new(*self.points.last().unwrap(), self.points[0])
        }
        else {
            Segment2::new(self.points[p], self.points[p + 1])
        }
    }

    fn iter_segments(&self) -> ContourSegments {
        ContourSegments {
            contour: self,
            index: 0,
        }
    }

    pub fn change_orientation(&mut self) {
        self.points.reverse();
        if let Some(cc) = self._is_clockwise.get() {
            self._is_clockwise.set(Some(! cc));
        }
    }
    pub fn set_clockwise(&mut self) {
        if self.counterclockwise() {
            self.change_orientation();
        }
    }
    pub fn set_counterclockwise(&mut self) {
        if self.clockwise() {
            self.change_orientation();
        }
    }

    pub fn add(&mut self, s: MapPoint) {
        self.points.push(s);
    }
    fn _erase(&mut self, i: usize) {
        self.points.remove(i);
    }
    fn _clear(&mut self) {
        self.points.clear();
        self.holes.clear();
    }
    fn _last(&self) -> MapPoint {
        *self.points.last().unwrap()
    }
    fn add_hole(&mut self, ind: usize) {
        self.holes.push(ind);
    }
    pub fn external(&self) -> bool {
        self._external
    }
    fn set_external(&mut self, e: bool) {
        self._external = e;
    }
}

struct ContourSegments<'a> {
    contour: &'a Contour,
    index: usize,
}

impl<'a> Iterator for ContourSegments<'a> {
    type Item = Segment2;

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.contour.points.len();

        if len == 0 {
            return None;
        }

        let i = self.index;

        self.index += 1;

        if i < len - 1 {
            Some(Segment2::new(self.contour.points[i], self.contour.points[i + 1]))
        } else if i == len - 1 {
            Some(Segment2::new(self.contour.points[i], self.contour.points[0]))
        } else {
            None
        }
    }
}

struct IndexComparator<'a, T: 'a + Ord>(usize, &'a Vec<T>);
impl<'a, T: 'a + Ord> cmp::PartialEq for IndexComparator<'a, T> {
    fn eq(&self, other: &IndexComparator<'a, T>) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<'a, T: 'a + Ord> cmp::Eq for IndexComparator<'a, T> { }
impl<'a, T: 'a + Ord> cmp::PartialOrd for IndexComparator<'a, T> {
    fn partial_cmp(&self, other: &IndexComparator<'a, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<'a, T: 'a + Ord> cmp::Ord for IndexComparator<'a, T> {
    fn cmp(&self, other: &IndexComparator<'a, T>) -> Ordering {
        self.1[self.0].cmp(&self.1[other.0])
    }
}


#[derive(Clone)]
pub struct Polygon {
    /// Set of contours conforming the polygon
    pub contours: Vec<Contour>,
}

impl Polygon {
    pub fn new() -> Self {
        Polygon{ contours: Vec::new() }
    }

    /// Get the p-th contour
    fn _contour(&self, p: usize) -> &Contour {
        &self.contours[p]
    }

    fn _join(&mut self, mut pol: Polygon) {
        let size = self.contours.len();
        for mut contour in pol.contours.drain(..) {
            for mut hole in &mut contour.holes {
                *hole += size;
            }
            self.contours.push(contour);
        }
    }

    fn nvertices(&self) -> usize {
        self.contours.iter().map(|c| c.points.len()).sum()
    }

    pub fn bbox(&self) -> MapRect {
        if self.contours.len() == 0 {
            return MapRect::new(MapPoint::origin(), MapSize::zero());
        }
        let mut bbox = self.contours[0].bbox();
        for contour in self.contours.iter().skip(1) {
            bbox = bbox.union(&contour.bbox());
        }
        bbox
    }

    fn _move_by(&mut self, dx: f64, dy: f64) {
        for contour in self.contours.iter_mut() {
            contour._move_by(dx, dy);
        }
    }

    pub fn compute_holes(&mut self) {
        if self.contours.len() < 2 {
            if self.contours.len() == 1 && self.contours[0].clockwise() {
                self.contours[0].change_orientation();
            }
            return;
        }

        let mut segments_mut = Vec::with_capacity(self.nvertices());
        for (contour_id, contour) in self.contours.iter_mut().enumerate() {
            // Initialize every contour to ccw; we'll fix them all in a moment
            contour.set_counterclockwise();

            for (point_id, segment) in contour.iter_segments().enumerate() {
                // vertical segments are not processed
                if segment.is_vertical() {
                    continue;
                }

                let index = segments_mut.len();
                segments_mut.push(SweepSegment::new(
                    segment.source, segment.target, index, 0, (contour_id, point_id)));
            }
        }

        // OK, now we can grab and sort the endpoints themselves in sweep order, which is x-wards
        let segments = segments_mut;  // kill mutability so refs stay valid
        let mut endpoints = Vec::with_capacity(segments.len() * 2);
        for segment in &segments {
            endpoints.push(SweepEndpoint(segment, SegmentEnd::Left));
            endpoints.push(SweepEndpoint(segment, SegmentEnd::Right));
        }
        endpoints.sort();

        // Use a sweep line to detect which contours are holes.  Consider:
        //     +--------+
        //    / +----+ /
        //   / /    / /
        //  / +----+ /
        // +--------+
        // When the line touches the left end of a segment, that segment becomes "active"; when it
        // touches the right end, that segment becomes inactive.  At any point, the set of active
        // segments is those that would be touched by a vertical line at the current x coordinate.
        // A contour is therefore a hole if the number of active segments below (or above) some
        // point within it is even -- or, equivalently, if the contour on the other side of the
        // nearest segment is /not/ a hole.
        let mut active_segments = BTreeSet::new();
        let capacity = self.contours.len();
        let mut processed = Vec::with_capacity(capacity);
        processed.resize(capacity, false);
        let mut hole_of = Vec::with_capacity(capacity);
        hole_of.resize(capacity, None);
        let mut nprocessed = 0;
        for &SweepEndpoint(segment, end) in &endpoints {
            // Stop if we've seen every contour
            if nprocessed >= self.contours.len() {
                break;
            }

            if end == SegmentEnd::Right {
                // This is a RIGHT endpoint; this segment is no longer active
                active_segments.remove(&segment);
                continue;
            }

            // This is a LEFT endpoint; add this as a new active segment
            active_segments.insert(segment);

            let (contour_id, _segment_id) = segment.data;
            if processed[contour_id] {
                continue;
            }
            processed[contour_id] = true;
            nprocessed += 1;

            // Find the previous active segment, which should be the nearest one below us
            // NOTE: The turbofish fixes an ambiguity in type inference introduced in 1.28, which
            // added a RangeBounds impl for RangeTo<&T> as well as RangeTo<T>
            let prev_segment = match active_segments.range::<&_, _>(..segment).last() {
                Some(segment) => { segment }
                None => {
                    // We're on the outside, so set us ccw and continue
                    self.contours[contour_id].set_counterclockwise();
                    continue;
                }
            };
            let (prev_contour_id, _prev_segment_id) = prev_segment.data;

            if ! prev_segment.faces_outwards {
                hole_of[contour_id] = Some(prev_contour_id);
                self.contours[contour_id].set_external(false);
                self.contours[prev_contour_id].add_hole(contour_id);
                if self.contours[prev_contour_id].counterclockwise() {
                    self.contours[contour_id].set_clockwise();
                }
                else {
                    self.contours[contour_id].set_counterclockwise();
                }
            }
            else if let Some(parent) = hole_of[prev_contour_id] {
                hole_of[contour_id] = Some(parent);
                self.contours[contour_id].set_external(false);
                self.contours[parent].add_hole(contour_id);
                if self.contours[parent].counterclockwise() {
                    self.contours[contour_id].set_clockwise();
                }
                else {
                    self.contours[contour_id].set_counterclockwise();
                }
            }
            else {
                self.contours[contour_id].set_counterclockwise();
            }
        }
    }
}

impl ops::Index<usize> for Polygon {
    type Output = Contour;

    fn index(&self, index: usize) -> &Self::Output {
        &self.contours[index]
    }
}
impl ops::IndexMut<usize> for Polygon {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.contours[index]
    }
}


// -----------------------------------------------------------------------------
// booleanop

type PolygonIndex = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolygonMode {
    Normal,
    RemoveEdges,
    RemoveInterior,
}

#[derive(Clone, Debug)]
struct SegmentPacket {
    polygon_index: PolygonIndex,
    mode: PolygonMode,
    edge_type: EdgeType,
    up_faces_outwards: bool,
    is_outside_other_poly: bool,
    // Which original polygons contain this segment?  (The polygon this segment came from is always
    // assumed to contain it; which side faces outwards is determined by up_faces_outwards.)
    left_faces_polygons: BitVec,
    right_faces_polygons: BitVec,

    is_in_result: bool,
    // Index of the segment below this one that's actually included in the result
    below_in_result: Option<usize>,

    // Used in connectEdges
    left_processed: bool,
    right_processed: bool,
    contour_id: usize,
    left_contour_id: Option<usize>,
    right_contour_id: Option<usize>,
    result_in_out: bool,
    left_index: usize,
    right_index: usize,
}

impl SegmentPacket {
    fn new(polygon_index: usize, mode: PolygonMode, npolygons: usize) -> Self {
        SegmentPacket{
            polygon_index,
            mode,
            edge_type: EdgeType::Normal,
            up_faces_outwards: false,
            is_outside_other_poly: false,
            left_faces_polygons: BitVec::from_elem(npolygons, false),
            right_faces_polygons: BitVec::from_elem(npolygons, false),

            is_in_result: false,
            below_in_result: None,

            left_processed: false,
            right_processed: false,
            // Slightly hokey, but means everything defaults to a single outermost shell of the
            // poly, which isn't unreasonable really
            contour_id: 0,
            left_contour_id: None,
            right_contour_id: None,
            result_in_out: false,
            // Definitely hokey
            left_index: 0,
            right_index: 0,
        }
    }
}

impl<'a> SweepEndpoint<'a, RefCell<SegmentPacket>> {
    fn is_processed(&self) -> bool {
        let packet = self.0.data.borrow();
        match self.1 {
            SegmentEnd::Left => packet.left_processed,
            SegmentEnd::Right => packet.right_processed,
        }
    }

    fn mark_processed(&self) {
        let mut packet = self.0.data.borrow_mut();
        match self.1 {
            SegmentEnd::Left => { packet.left_processed = true; }
            SegmentEnd::Right => { packet.right_processed = true; }
        }
    }

    fn _is_one_sided(&self) -> bool {
        self.0._is_one_sided()
    }

    fn _faces_void(&self) -> bool {
        let packet = self.0.data.borrow();
        // We face into the void iff we're in no other polygons, and we're the side of the segment
        // facing outwards from our original polygon.
        self._is_one_sided() &&
            (self.1 == SegmentEnd::Left) == packet.up_faces_outwards
    }
}

type BoolSweepSegment = SweepSegment<RefCell<SegmentPacket>>;
impl BoolSweepSegment {
    fn _is_one_sided(&self) -> bool {
        let packet = self.data.borrow();
        packet.left_faces_polygons.none() || packet.right_faces_polygons.none()
    }
}

/// @brief compute several fields of left event le
fn compute_fields(segment: &BoolSweepSegment, maybe_below: Option<&BoolSweepSegment>) {
    // anon scope so the packet goes away at the end and we can reborrow to call is_in_result
    {
        let mut packet = segment.data.borrow_mut();
        let polygon_index = packet.polygon_index;

        // Compute up_faces_outwards and is_outside_other_poly, the fields that tell us which
        // segments to include in the final polygon
        if let Some(below) = maybe_below {
            let below_packet = below.data.borrow();
            if polygon_index == below_packet.polygon_index {
                // below line segment in sl belongs to the same polygon that "se" belongs to
                packet.up_faces_outwards = ! below_packet.up_faces_outwards;
                packet.is_outside_other_poly = below_packet.is_outside_other_poly;
            }
            else {
                // below line segment in sl belongs to a different polygon that "se" belongs to
                packet.up_faces_outwards = below_packet.left_faces_polygons[polygon_index];
                if below.vertical() {
                    packet.is_outside_other_poly = ! below_packet.up_faces_outwards;
                }
                else {
                    packet.is_outside_other_poly = below_packet.up_faces_outwards;
                }
            }

            // Our lower (right) side faces the same space as the below line's upper (left) side,
            // UNLESS the below line coincides with us, in which case we face the same space as its
            // lower side.
            packet.left_faces_polygons.clear();
            packet.right_faces_polygons.clear();
            if below_packet.edge_type == EdgeType::NonContributing {
                packet.left_faces_polygons.union(&below_packet.left_faces_polygons);
                packet.right_faces_polygons.union(&below_packet.right_faces_polygons);
            }
            else {
                packet.left_faces_polygons.union(&below_packet.left_faces_polygons);
                packet.right_faces_polygons.union(&below_packet.left_faces_polygons);
            }
            // Left/up is the same, except that it adds or removes this poly
            let up_faces_outwards = packet.up_faces_outwards;
            packet.left_faces_polygons.set(polygon_index, ! up_faces_outwards);

            // compute below_in_result field
            if below.vertical() /* || ! is_in_result(operation, below) */ || below_packet.edge_type == EdgeType::NonContributing {
                packet.below_in_result = below_packet.below_in_result;
            }
            else {
                packet.below_in_result = Some(below.index);
            }
        }
        else {
            packet.up_faces_outwards = false;
            packet.is_outside_other_poly = true;

            // Down/right faces nothing
            packet.right_faces_polygons.clear();
            // Left/up faces only this polygon
            packet.left_faces_polygons.clear();
            packet.left_faces_polygons.set(polygon_index, true);
        }
    }

    let mut left_polys = Vec::new();
    let mut right_polys = Vec::new();
    for i in 0..segment.data.borrow().left_faces_polygons.len() {
        if segment.data.borrow().left_faces_polygons[i] {
            left_polys.push(i);
        }
        if segment.data.borrow().right_faces_polygons[i] {
            right_polys.push(i);
        }
    }
    if SPEW {
        println!("@@@ computing fields for #{}, with below {:?} -> {:?} {:?}", segment.index, maybe_below.map(|x| x.index), left_polys, right_polys);
    }

    let is_in_result = is_in_result(segment);
    segment.data.borrow_mut().is_in_result = is_in_result;
}

/* Check whether a segment should be included in the final polygon */
fn is_in_result(segment: &BoolSweepSegment) -> bool {
    let packet = segment.data.borrow();
    /*
    match packet.edge_type {
        EdgeType::Normal => match operation {
            BooleanOpType::Intersection => ! packet.is_outside_other_poly,
            BooleanOpType::Union => packet.is_outside_other_poly,
            // TODO this will, of course, need to become a bit more specific
            BooleanOpType::Difference => (packet.polygon_index == 0 && packet.is_outside_other_poly) || (packet.polygon_index == 1 && ! packet.is_outside_other_poly),
            BooleanOpType::ExclusiveOr => true,
        }
        EdgeType::SameTransition => operation == BooleanOpType::Intersection || operation == BooleanOpType::Union,
        EdgeType::DifferentTransition => operation == BooleanOpType::Difference,
        EdgeType::NonContributing => false,
    }
    */
    if packet.edge_type == EdgeType::NonContributing {
        return false;
    }
    if packet.mode == PolygonMode::RemoveEdges {
        return false;
    }
    /*
    if packet.mode == PolygonMode::RemoveInterior {
        // TODO how...  do i...  also remove any line on the inside?
        return false;
    }
    */
    return true;
}

/* Check for and handle an intersection between two adjacent segments */
fn handle_intersections<'a>(maybe_seg1: Option<&'a BoolSweepSegment>, maybe_seg2: Option<&'a BoolSweepSegment>) -> (usize, Option<MapPoint>, Option<MapPoint>) {
    let seg1 = match maybe_seg1 {
        Some(val) => val,
        None => return (0, None, None),
    };
    let seg2 = match maybe_seg2 {
        Some(val) => val,
        None => return (0, None, None),
    };
//  if (e1->pol == e2->pol) // you can uncomment these two lines if self-intersecting polygons are not allowed
//      return 0;

    match intersect_segments(&seg1.segment(), &seg2.segment()) {
        SegmentIntersection::None => {
            (0, None, None)
        }
        SegmentIntersection::Point(intersection) => {
            if seg1.left_point == seg2.left_point || seg1.right_point == seg2.right_point {
                // the line segments intersect at an endpoint of both line segments
                return (0, None, None);
            }

            // The line segments associated to le1 and le2 intersect
            let pt1 = if seg1.left_point != intersection && seg1.right_point != intersection {
                // the intersection point is not an endpoint of le1->segment ()
                Some(intersection)
            }
            else {
                None
            };
            let pt2 = if seg2.left_point != intersection && seg2.right_point != intersection {
                // the intersection point is not an endpoint of le2->segment ()
                Some(intersection)
            }
            else {
                None
            };

            (1, pt1, pt2)
        }
        SegmentIntersection::Segment(a, b) => {
            if seg1.data.borrow().polygon_index == seg2.data.borrow().polygon_index {
                // the line segments overlap, but they belong to the same polygon
                // FIXME would be nice to bubble up a Result i guess
                println!("hm, what, is happening here...");
                println!("seg1: {:?}", seg1);
                println!("seg2: {:?}", seg2);
                println!("overlap: {}, {}", a, b);
                panic!("Sorry, edges of the same polygon overlap");
            }

            // The line segments associated to le1 and le2 overlap
            let left_coincide = seg1.left_point == seg2.left_point;
            // let right_coincide = seg1.right_point == seg2.right_point;
            let left_cmp = compare_points(seg1.left_point, seg2.left_point);
            let right_cmp = compare_points(seg1.right_point, seg2.right_point);

            if left_coincide {
                // Segments share a left endpoint, and may even be congruent.  After this split
                // they'll definitely coincide, so mark the bottom one as extraneous.
                // (Note that changes we make to the bottom segment here are inherited into the
                // left coincident part, not the right part.)
                let edge_type1 = EdgeType::NonContributing;
                let edge_type2 = if
                    seg1.data.borrow().up_faces_outwards ==
                    seg2.data.borrow().up_faces_outwards
                    { EdgeType::SameTransition }
                else { EdgeType::DifferentTransition };

                {
                    if SPEW {
                        println!("due to split, setting #{} to {:?} and #{} to {:?}", seg1.index, edge_type1, seg2.index, edge_type2);
                    }
                    seg1.data.borrow_mut().edge_type = edge_type1;
                    if seg2.data.borrow().edge_type != EdgeType::NonContributing {
                        seg2.data.borrow_mut().edge_type = edge_type2;
                    }
                }

                // If the right endpoints don't coincide, then one lies on the other segment
                // and needs to split it
                match right_cmp {
                    Ordering::Less =>    return (2, None, Some(seg1.right_point)),
                    Ordering::Greater => return (2, Some(seg2.right_point), None),
                    Ordering::Equal =>   return (2, None, None),
                }
            }
            else {
                // The segments overlap in some fashion, but not at their left endpoints.  That
                // leaves several possible configurations, but all of them ultimately require
                // the left endpoint of one to split the other; we'll let the other split happen
                // when we reach the split point
                match left_cmp {
                    Ordering::Less =>    return (3, Some(seg2.left_point), None),
                    Ordering::Greater => return (3, None, Some(seg1.left_point)),
                    Ordering::Equal =>   unreachable!(),
                }
            }
            // FIXME investigate whether restoring all this (and restoring the double-split case
            // above) would avoid my goofy near-integer problem?
            /*
            else if right_coincide {
                // Segments share a right endpoint (and are not congruent), so the left end of
                // one splits the other
                match left_cmp {
                    Ordering::Less =>    return (3, Some(seg1.split(seg2.left_point)))
                    Ordering::Greater => return (3, Some(seg2.split(seg1.left_point)))
                    Ordering::Equal =>   unreachable!(),
                }
            }
            else if left_cmp == right_cmp {
                // Segments overlap, but neither subsumes the other
                // NOTE: original code did two splits here; i'm doing one and assuming the
                // other intersection will be picked up during a later check!
                // FIXME observe that this code is now identical to the case above
                match left_cmp {
                    Ordering::Less => {
                        return (3, Some(seg1.split(seg2.left_point)));
                        // self.split_segment(seg1, seg2.left_point);
                        // self.split_segment(seg2, seg1.right_point);
                    }
                    Ordering::Greater => {
                        return (3, Some(seg2.split(seg1.left_point)));
                        // self.split_segment(seg1, seg2.right_point);
                        // self.split_segment(seg2, seg1.left_point);
                    }
                    Ordering::Equal => unreachable!(),
                }
            }
            else {
                // One segment includes the other one
                // NOTE: original code did a cute subtle .otherEvent trick to take care of the
                // problem of the segments being mutated out from under us; i just changed the
                // call order, which i sure hope is right!
                // NOTE: original code did two splits here; i'm doing one and assuming the
                // other intersection will be picked up during a later check!
                // FIXME observe that this code is now identical to the case above
                match left_cmp {
                    Ordering::Less => {
                        return (3, Some(seg1.split(seg2.left_point)));
                        // self.split_segment(seg1, seg2.right_point);
                        // self.split_segment(seg1, seg2.left_point);
                    }
                    Ordering::Greater => {
                        return (3, Some(seg2.split(seg1.left_point)));
                        // self.split_segment(seg2, seg1.right_point);
                        // self.split_segment(seg2, seg1.left_point);
                    }
                    Ordering::Equal => unreachable!(),
                }
            }
            */
        }
    }
}

type BoolSweepEndpoint<'a> = SweepEndpoint<'a, RefCell<SegmentPacket>>;
fn find_next_segment<'a>(current_endpoint: &'a BoolSweepEndpoint<'a>, included_endpoints: &'a Vec<BoolSweepEndpoint>) -> &'a BoolSweepEndpoint<'a> {
    // TODO it does slightly bug me that this is slightly inefficient but, eh? i GUESS i could
    // track the endpoints everywhere, or even just pass both pairs of points around??
    let next_point = current_endpoint.other_point();
    let mut start_index = if current_endpoint.1 == SegmentEnd::Left {
        current_endpoint.0.data.borrow().right_index
    }
    else {
        current_endpoint.0.data.borrow().left_index
    };

    while start_index > 0 && included_endpoints[start_index - 1].point() == next_point {
        start_index -= 1;
    }

    // FIXME dammit, the above doesn't work if we omit one of the points from included_endpoints
    // entirely!  its _index will just be the default of zero.  ass.  fuck.
    start_index = 0;
    while included_endpoints[start_index].point() != next_point {
        start_index += 1;
    }

    // Find the closest angle.  That means the biggest dot product, or the smallest, maybe.
    // TODO should i just use signed triangle area here?
    let mut closest_dot = f64::NAN;
    let mut closest_endpoint = None;
    let mut seen_ccw = false;
    let current_vec = next_point - current_endpoint.point();
    // Ascend the list of endpoints until we find a match that isn't part of an already
    // processed segment
    for i in start_index .. included_endpoints.len() {
        let endpoint = &included_endpoints[i];
        if endpoint.is_processed() {
            continue;
        }
        else if endpoint == current_endpoint {
            // FIXME is this necessary?  i.e., is the passed-in endpoint already marked processed
            // (yes)
            continue;
        }
        else if endpoint.0 == current_endpoint.0 {
            // DEFINITELY do not backtrack along the same line holy jesus
            // FIXME the problem here is that the other edge of this line is a hole, and i only
            // proactively mark the outside as already processed...
            continue;
        }
        else if next_point == endpoint.point() {
            let vec = endpoint.other_point() - endpoint.point();
            let dot = current_vec.dot(vec) / vec.length();

            let this_ccw = current_vec.cross(vec) > 0.;
            if this_ccw {
                // This angle is counterclockwise; if all we've seen so far is clockwise then it
                // wins by default
                if ! seen_ccw {
                    seen_ccw = true;
                    closest_dot = dot;
                    closest_endpoint = Some(endpoint);
                    continue;
                }
            }
            else {
                // This angle is clockwise; only consider it at all if we haven't seen a ccw angle
                if seen_ccw {
                    continue;
                }
            }

            if closest_endpoint.is_none() || (this_ccw && dot < closest_dot) || (!this_ccw && dot > closest_dot) {
                closest_dot = dot;
                closest_endpoint = Some(endpoint);
            }
        }
        else {
            break;
        }
    }

    if let Some(endpoint) = closest_endpoint {
        return endpoint;
    }
    else {
        panic!("ran out of endpoints");
    }
}

pub fn compute(polygons: &[(Polygon, PolygonMode)], operation: BooleanOpType) -> Polygon {
    // ---------------------------------------------------------------------------------------------
    // Detect trivial cases that can be answered without doing any work

    /*
     * TODO restore these...  not entirely clear how/whether they'd apply in a world with n input
     * polygons and no fixed operations
    let subjectBB = subject.bbox();     // for optimizations 1 and 2
    let clippingBB = clipping.bbox();   // for optimizations 1 and 2
    let MINMAXX = f64::min(subjectBB.max_x(), clippingBB.max_x()); // for optimization 2

    if subject.contours.is_empty() || clipping.contours.is_empty() {
        // At least one of the polygons is empty
        if operation == BooleanOpType::Difference {
            return subject.clone();
        }
        if operation == BooleanOpType::Union || operation == BooleanOpType::ExclusiveOr {
            if subject.contours.is_empty() {
                return clipping.clone();
            }
            else {
                return subject.clone();
            }
        }
        // XXX or...  what else here?  also use a match block above
    }
    if ! subjectBB.touches(&clippingBB) {
        // the bounding boxes do not overlap
        if operation == BooleanOpType::Difference {
            return subject.clone();
        }
        if operation == BooleanOpType::Union || operation == BooleanOpType::ExclusiveOr {
            let mut result = subject.clone();
            result.join(clipping.clone());
            return result;
        }
        // XXX again, or what?
    }
    */

    // ---------------------------------------------------------------------------------------------
    // Build a list of all the segments in the original polygons

    // This is tricky!  We need multiple references to the same segment no matter what we do, and
    // Rust doesn't exactly love that.  The best solution I've found: keep all the segments
    // allocated in this typed arena (which will be thrown away all at once at the end), and
    // discard their mutability as soon as we get them.  In order to change the attached data, it's
    // stored in a RefCell.  Phew.  No Rc required, though!
    let arena = Arena::new();
    let mut endpoint_queue = BinaryHeap::new();
    let mut segment_id = 0;
    let mut svg_orig_group = Group::new();
    // TODO could reserve space here and elsewhere
    let mut segment_order = Vec::new();
    for (i, &(ref polygon, mode)) in polygons.iter().enumerate() {
        let mut data = Data::new();
        for contour in &polygon.contours {
            for seg in contour.iter_segments() {
            /*  if (s.degenerate ()) // if the two edge endpoints are equal the segment is dicarded
                    return;          // This can be done as preprocessing to avoid "polygons" with less than 3 edges */
                let segment: &_ = arena.alloc(SweepSegment::new(
                    seg.source, seg.target, segment_id, i, RefCell::new(SegmentPacket::new(i, mode, polygons.len()))));
                segment_id += 1;
                segment_order.push(segment);
                endpoint_queue.push(Reverse(SweepEndpoint(segment, SegmentEnd::Left)));
                endpoint_queue.push(Reverse(SweepEndpoint(segment, SegmentEnd::Right)));
            }

            let point = contour.points.last().unwrap();
            data = data.move_to((point.x, -point.y));
            for point in &contour.points {
                data = data.line_to((point.x, -point.y));
            }
        }
        svg_orig_group.append(Path::new().set("d", data).set("fill", "#ffcc44").set("fill-opacity", 0.25).set("data-poly-index", i));
    }

    // ---------------------------------------------------------------------------------------------
    // Perform a sweep

    // segments intersecting the sweep line
    // XXX need to wrap these in Reverse anyway
    // XXX why do i need the explicit type anno here
    let mut active_segments: BTreeSet<&BoolSweepSegment> = BTreeSet::new();
    let mut swept_segments = Vec::new();

    macro_rules! _split_segment (
        ($seg:expr, $pt:expr) => (
            {
                let pt = $pt;
                let seg = $seg;
                if SPEW {
                    println!("ah!  splitting #{} at {:?}, into #{} and #{}", seg.index, pt, segment_id, segment_id + 1);
                }
                // It's not obvious at a glance, but in the original algorithm, the left end of a
                // split inherits the original segment's data, and the right end gets data fresh.
                // This is important since handle_intersections assigns the whole segment's
                // edge_type before splitting (and, TODO, maybe it shouldn't) but the right end
                // isn't meant to inherit that!
                let polygon_index = seg.data.borrow().polygon_index;
                let left: &_ = arena.alloc(SweepSegment::new(
                    seg.left_point, pt, segment_id, polygon_index, seg.data.clone()));
                segment_id += 1;
                let right: &_ = arena.alloc(SweepSegment::new(
                    pt, seg.right_point, segment_id, polygon_index, RefCell::new(SegmentPacket::new(polygon_index, seg.data.borrow().mode, polygons.len()))));
                {
                    // TODO ugly ass copy
                    let mut packet = right.data.borrow_mut();
                    packet.left_faces_polygons.clear();
                    packet.left_faces_polygons.union(&left.data.borrow().left_faces_polygons);
                    packet.right_faces_polygons.clear();
                    packet.right_faces_polygons.union(&left.data.borrow().right_faces_polygons);
                }
                segment_id += 1;

                // XXX this is pretty ugly, but it's necessary -- when two segments coincide, the
                // lower one is marked NonContributing, and that one MUST be sorted lower even
                // after a split.  currently we break order ties by segment index, so, we gotta
                // keep the index right
                //segment_order[seg.index] = left;
                //segment_order.push(seg);
                segment_order.push(left);
                segment_order.push(right);
                // We split this segment in half, so replace the existing segment with its left end
                // and give it a faux right endpoint
                // TODO maybe worth asserting it was actually removed here
                if ! active_segments.remove(&seg) {
                    println!("!!! oh no, can't find segment #{}", seg.index);
                    for s in &active_segments {
                        println!("  vs #{}: {:?}, {}, {} | rev {:?}, {}, {}", s.index, seg.cmp(&s),triangle_signed_area(seg.left_point, seg.right_point, s.left_point), triangle_signed_area(seg.left_point, seg.right_point, s.right_point), s.cmp(&seg), triangle_signed_area(s.left_point, s.right_point, seg.left_point), triangle_signed_area(s.left_point, s.right_point, seg.right_point));
                    }
                    for s in &active_segments {
                        println!("{:?}", s);
                    }
                    panic!("couldn't find a segment that ought to exist");
                }
                active_segments.insert(left);
                endpoint_queue.push(Reverse(SweepEndpoint(left, SegmentEnd::Right)));
                endpoint_queue.push(Reverse(SweepEndpoint(right, SegmentEnd::Left)));
                endpoint_queue.push(Reverse(SweepEndpoint(right, SegmentEnd::Right)));

                // XXX does /this/ handle my goofy vertical case...?
                /*
                    // FIXME i don't quite understand what this is trying to do and i think it would be better
                    // solved by switching the points AND ALSO how is a rounding error possible here?  they
                    // should have exactly the same points!
                    if right.left < left.right {
                        // avoid a rounding error. The left event would be processed after the right event
                        //std::cout << "Oops" << std::endl;
                        //le.otherEvent.left = true;
                        //l.left = false;
                    }
                    if left.left < left.right {
                        // avoid a rounding error. The left event would be processed after the right event
                        //std::cout << "Oops2" << std::endl;
                    }
                */

                // TODO this is a quick hack to fix an awkward problem: segments get "pointers" to
                // the next segment below them that's in the final result, but since we split
                // segments by creating two new ones rather than mutating the original, a pointer
                // to a segment that later gets split becomes useless.  this checks for any such
                // pointers and adjusts them.  a more robust fix would be nice, but it might
                // involve doing a bit more work when assembling the final polygon?  or something
                for other_seg in &segment_order {
                    if other_seg.data.borrow().below_in_result == Some(seg.index) {
                        other_seg.data.borrow_mut().below_in_result = Some(left.index);
                    }
                }
                left
            }
        );
    );


    // Grab the next endpoint, or bail if we've run out
    while let Some(Reverse(SweepEndpoint(mut segment, end))) = endpoint_queue.pop() {
        if SPEW {
            println!("");
            println!("LOOP ITERATION: {:?} of #{:?}[{}] {:?} -> {:?}", end, segment.index, segment.order, segment.left_point, segment.right_point);
            for seg in &active_segments {
                println!("  {} #{}[{}] {:?} -> {:?} | {} {}", if seg < &segment { "<" } else if seg > &segment { ">" } else { "=" }, seg.index, seg.order, seg.left_point, seg.right_point, triangle_signed_area(seg.left_point, seg.right_point, segment.left_point), triangle_signed_area(seg.left_point, seg.right_point, segment.right_point));
            }
        }
        // let endpoint = match end {
        //     SegmentEnd::Left => segment.left_point,
        //     SegmentEnd::Right => segment.right_point,
        // };
        /* TODO restore these
        // optimization 2
        if operation == BooleanOpType::Intersection && endpoint.x > MINMAXX {
            break;
        }
        if operation == BooleanOpType::Difference && endpoint.x > subjectBB.max_x() {
            break;
        }
        */

        if end == SegmentEnd::Right {
            // delete line segment associated to "event" from sl and check for intersection between the neighbors of "event" in sl
            // NOTE the original code stored an iterator ref in posSL; not clear if there's an
            // equivalent here, though obviously i /can/ get a two-way iterator
            // FIXME this is especially inefficient since i'm looking up the same value /three/
            // times
            if active_segments.remove(&segment) {
                swept_segments.push(segment);

                let maybe_below = active_segments.range::<&_, _>(..segment).last().map(|v| *v);
                let maybe_above = active_segments.range::<&_, _>(segment..).next().map(|v| *v);
                let cross = handle_intersections(maybe_below, maybe_above);

                if let Some(pt) = cross.1 {
                    _split_segment!(maybe_below.unwrap(), pt);
                }
                if let Some(pt) = cross.2 {
                    _split_segment!(maybe_above.unwrap(), pt);
                }
            }

            continue;
        }

        // the line segment must be inserted into sweep_line
        let mut maybe_below = active_segments.range::<&_, _>(..segment).last().map(|v| *v);
        let mut maybe_above = active_segments.range::<&_, _>(segment..).next().map(|v| *v);
        active_segments.insert(segment);
        compute_fields(segment, maybe_below);
        // Check for intersections with the segment above
        let cross = handle_intersections(Some(segment), maybe_above);
        if let Some(pt) = cross.1 {
            segment = _split_segment!(segment, pt);
        }
        if let Some(pt) = cross.2 {
            maybe_above = Some(_split_segment!(maybe_above.unwrap(), pt));
        }
        if cross.0 == 2 {
            // NOTE: this seems super duper goofy to me; why call compute_fields a second time
            // with the same args in particular?
            // NOTE: answer is: because returning 2 means we changed the segments' edge types, so
            // is_in_result might change!
            compute_fields(segment, maybe_below);
            compute_fields(maybe_above.unwrap(), Some(segment));
        }
        // Check for intersections with the segment below
        let cross = handle_intersections(maybe_below, Some(segment));
        if let Some(pt) = cross.1 {
            maybe_below = Some(_split_segment!(maybe_below.unwrap(), pt));
        }
        if let Some(pt) = cross.2 {
            segment = _split_segment!(segment, pt);
        }
        if cross.0 == 2 {
            // XXX might want to enforce that these aren't the same pair twice, since that makes
            // things...  confusing.  artifact of how we sort and split; see comment in PartialOrd
            compute_fields(maybe_below.unwrap(), active_segments.range::<&_, _>(..maybe_below.unwrap()).last().map(|v| *v));
            compute_fields(segment, maybe_below);
        }
    }

    if SPEW {
        println!("");
        println!("---MAIN LOOP DONE ---");
        println!("");
    }

    {
    let mut svg_swept_group = Group::new();
    for segment in &swept_segments {
        if ! segment.data.borrow().is_in_result {
            continue;
        }
        let mut skip_left = false;
        for (polygon_index, flag) in segment.data.borrow().left_faces_polygons.iter().enumerate() {
            if flag && polygons[polygon_index].1 == PolygonMode::RemoveInterior {
                skip_left = true;
                break;
            }
        }
        if segment.data.borrow().left_faces_polygons.none() {
            skip_left = true;
        }
        let mut skip_right = false;
        for (polygon_index, flag) in segment.data.borrow().right_faces_polygons.iter().enumerate() {
            if flag && polygons[polygon_index].1 == PolygonMode::RemoveInterior {
                skip_right = true;
                break;
            }
        }
        if segment.data.borrow().right_faces_polygons.none() {
            skip_right = true;
        }

        if skip_left && skip_right {
            continue;
        }
        svg_swept_group.append(
            Line::new()
            .set("x1", segment.left_point.x)
            .set("y1", -segment.left_point.y)
            .set("x2", segment.right_point.x)
            .set("y2", -segment.right_point.y)
            .set("stroke", "green")
            .set("stroke-width", 1)
        );
        let mut left_polys = Vec::new();
        let mut right_polys = Vec::new();
        for i in 0..polygons.len() {
            if segment.data.borrow().left_faces_polygons[i] {
                left_polys.push(i);
            }
            if segment.data.borrow().right_faces_polygons[i] {
                right_polys.push(i);
            }
        }
        svg_swept_group.append(
            Text::new()
            .add(svg::node::Text::new(format!("{} {:?}{:?} {} {}", segment.index, left_polys, right_polys, if skip_left { "L" } else { "" }, if skip_right { "R" } else { "" })))
            .set("x", (segment.left_point.x + segment.right_point.x) / 2.0)
            .set("y", -(segment.left_point.y + segment.right_point.y) / 2.0)
            .set("fill", if segment.data.borrow().edge_type == EdgeType::NonContributing { "lightgreen" } else {"darkgreen" })
            .set("text-anchor", "middle")
            .set("alignment-baseline", "central")
            .set("font-size", 4)
        );
    }
    let mut svg_active_group = Group::new();
    for seg in &active_segments {
        svg_active_group.append(
            Line::new()
            .set("x1", seg.left_point.x)
            .set("y1", -seg.left_point.y)
            .set("x2", seg.right_point.x)
            .set("y2", -seg.right_point.y)
            .set("stroke", "red")
            .set("stroke-width", 1)
        );
    }
    let doc = Document::new()
        .set("viewBox", (-16, -112, 128, 128))
        .add(Style::new("line:hover { stroke: gold; }"))
        .add(svg_orig_group)
        .add(svg_swept_group)
        .add(svg_active_group)
    ;
    svg::save("idchoppers-shapeops-debug.svg", &doc)
        .expect("could not save idchoppers-shapeops-debug.svg");
    }

    // Finally, trace the output polygons from the final set of segments we got
    let count = swept_segments.len();
    let mut included_segments: Vec<&BoolSweepSegment> = Vec::with_capacity(count);
    let mut included_endpoints = Vec::with_capacity(count * 2);
    for segment in swept_segments.into_iter() {
        if ! segment.data.borrow().is_in_result {
            continue;
        }
        // FIXME this is nnnnot gonna fly
        let mut skip_left = false;
        for (polygon_index, flag) in segment.data.borrow().left_faces_polygons.iter().enumerate() {
            if flag && polygons[polygon_index].1 == PolygonMode::RemoveInterior {
                skip_left = true;
                break;
            }
        }
        if segment.data.borrow().left_faces_polygons.none() {
            skip_left = true;
        }
        let mut skip_right = false;
        for (polygon_index, flag) in segment.data.borrow().right_faces_polygons.iter().enumerate() {
            if flag && polygons[polygon_index].1 == PolygonMode::RemoveInterior {
                skip_right = true;
                break;
            }
        }
        if segment.data.borrow().right_faces_polygons.none() {
            skip_right = true;
        }

        if skip_left && skip_right {
            continue;
        }

        included_segments.push(segment);
        if ! skip_left {
            included_endpoints.push(SweepEndpoint(segment, SegmentEnd::Left));
        }
        if ! skip_right {
            included_endpoints.push(SweepEndpoint(segment, SegmentEnd::Right));
        }
    }
    included_segments.sort();
    included_endpoints.sort();

    if SPEW {
        println!();
        println!("-- segments --");
        for seg in &included_segments {
            println!("{:?}", seg);
        }
        println!();
        println!("-- endpoints --");
        for ep in &included_endpoints {
            println!("{:?} of #{} {:?} -> {:?}", ep.1, ep.0.index, ep.0.left_point, ep.0.right_point);
        }
    }

    for (i, &SweepEndpoint(segment, end)) in included_endpoints.iter().enumerate() {
        let mut packet = segment.data.borrow_mut();
        match end {
            SegmentEnd::Left => { packet.left_index = i; }
            SegmentEnd::Right => { packet.right_index = i; }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Construct the final polygon by grouping the segments together into contours

    let mut final_polygon = Polygon::new();
    for endpoint in included_endpoints.iter() {
        let &SweepEndpoint(segment, end) = endpoint;
        if endpoint.is_processed() {
            continue;
        }

        // FIXME maybe do this in previous loop
        if end == SegmentEnd::Left && segment.data.borrow().left_faces_polygons.none() {
            segment.data.borrow_mut().left_processed = true;
            continue;
        }
        else if end == SegmentEnd::Right && segment.data.borrow().right_faces_polygons.none() {
            segment.data.borrow_mut().right_processed = true;
            continue;
        }

        // Walk around looking for a polygon until we come back to the starting point
        let mut contour = Contour::new();
        let contour_id = final_polygon.contours.len();
        let starting_point = segment.left_point;
        contour.from_polygons = match end {
            SegmentEnd::Left => segment.data.borrow().left_faces_polygons.clone(),
            SegmentEnd::Right => segment.data.borrow().right_faces_polygons.clone(),
        };
        contour.add(starting_point);
        let mut current_endpoint = &included_endpoints[segment.data.borrow().left_index];
        if SPEW {
            println!("building contour {} from #{} {:?} {:?}", contour_id, segment.index, end, starting_point);
        }
        loop {
            current_endpoint.mark_processed();
            let current_segment = current_endpoint.0;
            {
                let mut packet = current_segment.data.borrow_mut();
                if current_endpoint.1 == SegmentEnd::Left {
                    packet.left_contour_id = Some(contour_id);
                    contour.from_polygons.union(&packet.left_faces_polygons);
                }
                else {
                    packet.right_contour_id = Some(contour_id);
                    contour.from_polygons.union(&packet.right_faces_polygons);
                }
                packet.result_in_out = current_endpoint.1 == SegmentEnd::Right;
            }

            let point = current_endpoint.other_point();
            if point == starting_point {
                break;
            }
            contour.add(point);

            current_endpoint = find_next_segment(current_endpoint, &included_endpoints);
            if SPEW {
                println!("... #{} {:?}", current_endpoint.0.index, current_endpoint.point());
            }
        }

        final_polygon.contours.push(contour);

        // FIXME if we go *clockwise* here (because we're tracing a hole), then the segment
        // below us is the first segment of the equivalent *counterclockwise* shape (which
        // we'll never trace because we eliminated it), so we need to actually use the segment
        // below *that* to figure out what we're inside.  also, the winding will be wrong,
        // because we call change_orientation below assuming that everything is ccw!
        // TODO maybe everything should just be counterclockwise?  but then how do i track
        // whether i'm "inside" or "outside"?
        // XXX can this happen for two-sided lines as well?
        if final_polygon[contour_id].clockwise() {
            let &SweepEndpoint(segment, _end) = current_endpoint;

            if let Some(below_segment_id) = segment.data.borrow().below_in_result {
                // TODO this is the ONLY PLACE that uses segment_order, or segment index at all!
                let below_segment = &segment_order[below_segment_id];
                let parent_contour_id = below_segment.data.borrow().left_contour_id.unwrap();
                if SPEW {
                    println!("this contour is clockwise, and the segment below is #{}, so i think it's a hole in {}", below_segment_id, parent_contour_id);
                }
                final_polygon[parent_contour_id].add_hole(contour_id);
                final_polygon[contour_id].set_external(false);
            }
            else {
                println!("!!! can't find the counter i'm a hole of");
            }
        }
    }

    // Assign contour neighbors
    // XXX is there a better way to do this...?
    let ncontours = final_polygon.contours.len();
    for contour in final_polygon.contours.iter_mut() {
        contour.neighbors.grow(ncontours, false);
    }
    for endpoint in &included_endpoints {
        let &SweepEndpoint(segment, _) = endpoint;
        if let Some(right_contour_id) = segment.data.borrow().right_contour_id {
            if let Some(left_contour_id) = segment.data.borrow().left_contour_id {
                final_polygon[left_contour_id].neighbors.set(right_contour_id, true);
                final_polygon[right_contour_id].neighbors.set(left_contour_id, true);
            }
        }
    }

    final_polygon
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_points(points: &[(f64, f64)]) -> Vec<MapPoint> {
        points.iter().map(|&(x, y)| MapPoint::new(x, y)).collect()
    }

    fn make_rect(x: f64, y: f64, width: f64, height: f64) -> Contour {
        let mut contour = Contour::new();
        contour.add(MapPoint::new(x, y));
        contour.add(MapPoint::new(x + width, y));
        contour.add(MapPoint::new(x + width, y + height));
        contour.add(MapPoint::new(x, y + height));
        return contour;
    }

    #[test]
    fn test_contour_empty_segments() {
        let contour = Contour::new();
        let iter = contour.iter_segments();

        assert_eq!(iter.count(), 0);
    }

    #[test]
    fn test_contour_many_segments() {
        let mut contour = Contour::new();
        contour.add(MapPoint::new(0.0, 0.0));
        contour.add(MapPoint::new(1.0, 1.0));
        contour.add(MapPoint::new(2.0, 2.0));

        for (i, p) in contour.iter_segments().enumerate() {
            assert_eq!(p, contour.segment(i));
        }
    }

    #[test]
    fn test_no_overlap() {
        let mut poly1 = Polygon::new();
        poly1.contours.push(make_rect(0., 0., 10., 10.));
        let mut poly2 = Polygon::new();
        poly2.contours.push(make_rect(20., 20., 10., 10.));
        let result = compute(&[(poly1.clone(), PolygonMode::Normal), (poly2.clone(), PolygonMode::Normal)], BooleanOpType::Intersection);
        assert_eq!(result.contours.len(), 2);
        assert_eq!(result.contours[0].points, poly1.contours[0].points);
        assert_eq!(result.contours[1].points, poly2.contours[0].points);
    }

    #[test]
    fn test_corner_overlap() {
        let mut poly1 = Polygon::new();
        poly1.contours.push(make_rect(0., 0., 10., 10.));
        let mut poly2 = Polygon::new();
        poly2.contours.push(make_rect(5., 5., 10., 10.));
        let result = compute(&[(poly1.clone(), PolygonMode::Normal), (poly2.clone(), PolygonMode::Normal)], BooleanOpType::Intersection);
        assert_eq!(result.contours.len(), 3);
        assert_eq!(result.contours[0].points, make_points(&[(0., 0.), (10., 0.), (10., 5.), (5., 5.), (5., 10.), (0., 10.)]));
        assert_eq!(result.contours[1].points, make_points(&[(5., 5.), (10., 5.), (10., 10.), (5., 10.)]));
        assert_eq!(result.contours[2].points, make_points(&[(5., 10.), (10., 10.), (10., 5.), (15., 5.), (15., 15.), (5., 15.)]));
    }

    #[test]
    fn test_shared_edge() {
        let mut poly1 = Polygon::new();
        poly1.contours.push(make_rect(0., 0., 10., 10.));
        let mut poly2 = Polygon::new();
        poly2.contours.push(make_rect(10., 10., 10., 10.));
        let result = compute(&[(poly1.clone(), PolygonMode::Normal), (poly2.clone(), PolygonMode::Normal)], BooleanOpType::Intersection);
        assert_eq!(result.contours.len(), 2);
        assert_eq!(result.contours[0].points, poly1.contours[0].points);
        assert_eq!(result.contours[1].points, poly2.contours[0].points);
    }
}
