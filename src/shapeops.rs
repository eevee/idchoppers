// hey what's up, this is a catastrophic mess ported from a paper:
// A simple algorithm for Boolean operations on polygons (2013), Martínez et al.
// TODO clean this up
// TODO rename it this name is bad
// TODO finish, it?
extern crate svg;
use svg::Document;
use svg::node::Node;
use svg::node::element::{Group, Line, Path, Rectangle, Style, Text};
use svg::node::element::path::Data;



use std::cmp;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;
use std::collections::LinkedList;
use std::ops;
use std::mem::transmute;

use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;

use std::f32;


use bit_vec::BitVec;
use euclid::TypedPoint2D;
use euclid::TypedRect;
use euclid::TypedSize2D;
use typed_arena::Arena;


pub struct MapSpace;
pub type MapPoint = TypedPoint2D<f32, MapSpace>;
pub type MapRect = TypedRect<f32, MapSpace>;
pub type MapSize = TypedSize2D<f32, MapSpace>;
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

#[derive(Debug)]
struct Segment2 {
    source: MapPoint,
    target: MapPoint,
}

impl Segment2 {
    fn new(source: MapPoint, target: MapPoint) -> Self {
        return Self{ source, target };
    }

    fn is_vertical(&self) -> bool {
        return self.source.x == self.target.x;
    }
}


// -----------------------------------------------------------------------------
// utilities

// NOTE: this, and everything else ported from the paper, assumes the y axis points UP
pub fn triangle_signed_area(a: MapPoint, b: MapPoint, c: MapPoint) -> f32 {
    return (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y);
}

/** Sign of triangle (p1, p2, o) */
/*
fn triangle_sign(p1: MapPoint, p2: MapPoint, o: MapPoint) -> i8 {
    let det = (p1.x - o.x) * (p2.y - o.y) - (p2.x - o.x) * (p1.y - o.y);
    if det < 0. {
        return -1;
    }
    else if det > 0. {
        return 1;
    }
    else {
        return 0;
    }
}

fn triangle_contains_point(s: Segment2, o: MapPoint, p: MapPoint) -> bool {
    let x = triangle_sign(s.source, s.target, p);
    return (x == triangle_sign(s.target, o, p)) && (x == triangle_sign(o, s.source, p));
}
*/

fn check_span_overlap(u0: f32, u1: f32, v0: f32, v1: f32) -> Option<(f32, f32)> {
    if u1 < v0 || u0 > v1 {
        return None;
    }
    if u1 > v0 {
        if u0 < v1 {
            return Some((u0.max(v0), u1.min(v1)));
        } else {
            // u0 == v1
            return Some((u0, u0));
        }
    }
    else {
        // u1 == v0
        return Some((u1, u1));
    }
}

enum SegmentIntersection {
    None,
    Point(MapPoint),
    // TODO one would think this would return a Segment
    Segment(MapPoint, MapPoint),
}

const sqrEpsilon: f32 = 0.0000001;
const EPSILON: f32 = 0.000000000000001;
fn intersect_segments(seg0: &Segment2, seg1: &Segment2) -> SegmentIntersection {
    let p0 = seg0.source;
    let d0 = seg0.target - p0;
    let p1 = seg1.source;
    let d1 = seg1.target - p1;
    let E = p1 - p0;
    let kross = d0.cross(d1);
    let sqrLen0 = d0.square_length();
    let sqrLen1 = d1.square_length();

    if kross * kross > sqrEpsilon * sqrLen0 * sqrLen1 {
        // Lines containing these segments intersect; check whether the segments themselves do
        let s = E.cross(d1) / kross;
        if s < 0. || s > 1. {
            return SegmentIntersection::None;
        }
        let t = E.cross(d0) / kross;
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
        return SegmentIntersection::Point(intersection);
    }

    // Segments are parallel; check if they're collinear
    let sqrLenE = E.square_length();
    let kross = E.cross(d0);
    if kross * kross > sqrEpsilon * sqrLen0 * sqrLenE {
        // Nope, no intersection
        return SegmentIntersection::None;
    }

    // Segments are collinear; check whether their endpoints overlap
    let s0 = d0.dot(E) / sqrLen0;  // so = Dot (D0, E) * sqrLen0
    let s1 = s0 + d0.dot(d1) / sqrLen0;  // s1 = s0 + Dot (D0, D1) * sqrLen0
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

    return SegmentIntersection::None;
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
    data: T,
}

impl<T> SweepSegment<T> {
    fn new(point0: MapPoint, point1: MapPoint, index: usize, data: T) -> SweepSegment<T> {
        let (left_point, right_point, faces_outwards) = if point0.x < point1.x || (point0.x == point1.x && point0.y < point1.y) {
            (point0, point1, false)
        }
        else {
            (point1, point0, true)
        };

        return SweepSegment{
            left_point,
            right_point,
            // TODO hang on, this is only even used in one place?  in polygon::compute_holes??
            // that seems weird?
            faces_outwards,
            index,
            data,
        };
    }

    /** Is the line segment (left_point, right_point) below point p */
    fn below(&self, p: MapPoint) -> bool {
        return triangle_signed_area(self.left_point, self.right_point, p) > 0.;
    }

    /** Is the line segment (point, other_poin) above point p */
    fn above(&self, p: MapPoint) -> bool {
        return ! self.below(p);
    }

    /** Is the line segment (point, other_poin) a vertical line segment */
    fn vertical(&self) -> bool {
        return self.left_point.x == self.right_point.x;
    }

    /** Return the line segment */
    fn segment(&self) -> Segment2 {
        return Segment2::new(self.left_point, self.right_point);
    }
}

impl<T: Clone> SweepSegment<T> {
    fn split(&self, midpoint: MapPoint, index1: usize, index2: usize) -> (Self, Self) {
        return (
            Self::new(self.left_point, midpoint, index1, self.data.clone()),
            Self::new(midpoint, self.right_point, index2, self.data.clone()),
        );
    }

}

impl<T> cmp::PartialEq for SweepSegment<T> {
    fn eq(&self, other: &SweepSegment<T>) -> bool {
        return self.cmp(other) == Ordering::Equal;
    }
}
impl<T> cmp::Eq for SweepSegment<T> { }

impl<T> cmp::PartialOrd for SweepSegment<T> {
    fn partial_cmp(&self, other: &SweepSegment<T>) -> Option<Ordering> {
        return Some(self.cmp(other));
    }
}

// This is used for the binary tree ordering in the sweep line algorithm, keeping sweep events in
// order by where on the y-axis they would cross the sweep line
impl<T> cmp::Ord for SweepSegment<T> {
    fn cmp(&self, other: &SweepSegment<T>) -> Ordering {
        if self as *const _ == other as *const _ {
            return Ordering::Equal;
        }

        if triangle_signed_area(self.left_point, self.right_point, other.left_point) == 0. &&
            triangle_signed_area(self.left_point, self.right_point, other.right_point) == 0.
        {
            // Segments are collinear.  Sort by some arbitrary consistent criteria
            // XXX this used to be pol, which would cause far more ties...  is this ok?  does
            // "consistent" mean it actually needs to use the sweep comparison?
            return self.index.cmp(&other.index);
                //.then_with(|| compare_by_sweep(self, other));
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
        return self.cmp(other) == Ordering::Equal;
    }
}
impl<'a, T: 'a> cmp::Eq for SweepEndpoint<'a, T> { }

impl<'a, T: 'a> cmp::PartialOrd for SweepEndpoint<'a, T> {
    fn partial_cmp(&self, other: &SweepEndpoint<'a, T>) -> Option<Ordering> {
        return Some(self.cmp(other));
    }
}

impl<'a, T: 'a> cmp::Ord for SweepEndpoint<'a, T> {
    fn cmp(&self, other: &SweepEndpoint<'a, T>) -> Ordering {
        if self as *const _ == other as *const _ {
            return Ordering::Equal;
        }

        return self.point().x.partial_cmp(&other.point().x).unwrap()
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
                // Collinear!  Fall back to something totally arbitrary
                // NOTE this used to be pol, unsure if this makes any real difference
                // || self.segment_index.cmp(&other.segment_index)
                // FIXME oh this is bad
                || Ordering::Equal
            );
    }
}

// -----------------------------------------------------------------------------
// polygon

#[derive(Clone)]
pub struct Contour {
    /** Set of points conforming the external contour */
    pub points: Vec<MapPoint>,
    /** Holes of the contour. They are stored as the indexes of the holes in a polygon class */
    pub holes: Vec<usize>,
    // is the contour an external contour? (i.e., is it not a hole?)
    _external: bool,
    _is_clockwise: Cell<Option<bool>>,
}

impl Contour {
    pub fn new() -> Self {
        return Contour{
            points: Vec::new(),
            holes: Vec::new(),
            _external: true,
            _is_clockwise: Cell::new(None),
        };
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
            min_x = f32::min(min_x, point.x);
            max_x = f32::max(max_x, point.x);
            min_y = f32::min(min_y, point.y);
            max_y = f32::max(max_y, point.y);
        }
        return MapRect::new(MapPoint::new(min_x, min_y), MapSize::new(max_x - min_x, max_y - min_y));
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
        return is_clockwise;
    }
    pub fn counterclockwise(&self) -> bool {
        return ! self.clockwise();
    }

    fn move_by(&mut self, dx: f32, dy: f32) {
        for point in self.points.iter_mut() {
            *point = MapPoint::new(point.x + dx, point.y + dy);
        }
    }

    /** Get the p-th vertex of the external contour */
    fn vertex(&self, p: usize) -> MapPoint { return self.points[p]; }
    fn segment(&self, p: usize) -> Segment2 {
        if p == self.points.len() - 1 {
            return Segment2::new(*self.points.last().unwrap(), self.points[0]);
        }
        else {
            return Segment2::new(self.points[p], self.points[p + 1]);
        }
    }

    // TODO this could be an actual iterator y'know
    fn iter_segments(&self) -> Vec<Segment2> {
        let mut ret = Vec::new();
        for i in 0 .. self.points.len() {
            ret.push(self.segment(i));
        }
        return ret;
    }

    pub fn changeOrientation(&mut self) {
        self.points.reverse();
        if let Some(cc) = self._is_clockwise.get() {
            self._is_clockwise.set(Some(! cc));
        }
    }
    pub fn setClockwise(&mut self) {
        if self.counterclockwise() {
            self.changeOrientation();
        }
    }
    pub fn setCounterClockwise(&mut self) {
        if self.clockwise() {
            self.changeOrientation();
        }
    }

    pub fn add(&mut self, s: MapPoint) {
        self.points.push(s);
    }
    fn erase(&mut self, i: usize) {
        self.points.remove(i);
    }
    fn clear(&mut self) {
        self.points.clear();
        self.holes.clear();
    }
    fn clearHoles(&mut self) {
        self.holes.clear();
    }
    fn last(&self) -> MapPoint {
        return *self.points.last().unwrap();
    }
    fn addHole(&mut self, ind: usize) {
        self.holes.push(ind);
    }
    fn hole(&self, p: usize) -> usize {
        return self.holes[p];
    }
    pub fn external(&self) -> bool {
        return self._external;
    }
    fn setExternal(&mut self, e: bool) {
        self._external = e;
    }
}

struct IndexComparator<'a, T: 'a + Ord>(usize, &'a Vec<T>);
impl<'a, T: 'a + Ord> cmp::PartialEq for IndexComparator<'a, T> {
    fn eq(&self, other: &IndexComparator<'a, T>) -> bool {
        return self.cmp(other) == Ordering::Equal;
    }
}
impl<'a, T: 'a + Ord> cmp::Eq for IndexComparator<'a, T> { }
impl<'a, T: 'a + Ord> cmp::PartialOrd for IndexComparator<'a, T> {
    fn partial_cmp(&self, other: &IndexComparator<'a, T>) -> Option<Ordering> {
        return Some(self.cmp(other));
    }
}
impl<'a, T: 'a + Ord> cmp::Ord for IndexComparator<'a, T> {
    fn cmp(&self, other: &IndexComparator<'a, T>) -> Ordering {
        return self.1[self.0].cmp(&self.1[other.0]);
    }
}


#[derive(Clone)]
pub struct Polygon {
    /** Set of contours conforming the polygon */
    pub contours: Vec<Contour>,
}

impl Polygon {
    pub fn new() -> Self {
        return Polygon{ contours: Vec::new() };
    }

    /** Get the p-th contour */
    fn contour(&self, p: usize) -> &Contour {
        return &self.contours[p];
    }

    fn join(&mut self, mut pol: Polygon) {
        let size = self.contours.len();
        for mut contour in pol.contours.drain(..) {
            for mut hole in &mut contour.holes {
                *hole += size;
            }
            self.contours.push(contour);
        }
    }

    fn nvertices(&self) -> usize {
        return self.contours.iter().map(|c| c.points.len()).sum();
    }

    pub fn bbox(&self) -> MapRect {
        if self.contours.len() == 0 {
            return MapRect::new(MapPoint::origin(), MapSize::zero());
        }
        let mut bbox = self.contours[0].bbox();
        for contour in self.contours.iter().skip(1) {
            bbox = bbox.union(&contour.bbox());
        }
        return bbox;
    }

    fn move_by(&mut self, dx: f32, dy: f32) {
        for contour in self.contours.iter_mut() {
            contour.move_by(dx, dy);
        }
    }

    pub fn compute_holes(&mut self) {
        if self.contours.len() < 2 {
            if self.contours.len() == 1 && self.contours[0].clockwise() {
                self.contours[0].changeOrientation();
            }
            return;
        }

        let mut segments_mut = Vec::with_capacity(self.nvertices());
        for (contour_id, contour) in self.contours.iter_mut().enumerate() {
            // Initialize every contour to ccw; we'll fix them all in a moment
            contour.setCounterClockwise();

            for (point_id, segment) in contour.iter_segments().iter().enumerate() {
                // vertical segments are not processed
                if segment.is_vertical() {
                    continue;
                }

                let index = segments_mut.len();
                segments_mut.push(SweepSegment::new(
                    segment.source, segment.target, index, (contour_id, point_id)));
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
        let mut holeOf = Vec::with_capacity(capacity);
        holeOf.resize(capacity, None);
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

            let (contour_id, segment_id) = segment.data;
            if processed[contour_id] {
                continue;
            }
            processed[contour_id] = true;
            nprocessed += 1;

            // Find the previous active segment, which should be the nearest one below us
            let prev_segment = match active_segments.range(..segment).last() {
                Some(segment) => { segment }
                None => {
                    // We're on the outside, so set us ccw and continue
                    self.contours[contour_id].setCounterClockwise();
                    continue;
                }
            };
            let (prev_contour_id, prev_segment_id) = prev_segment.data;

            if ! prev_segment.faces_outwards {
                holeOf[contour_id] = Some(prev_contour_id);
                self.contours[contour_id].setExternal(false);
                self.contours[prev_contour_id].addHole(contour_id);
                if self.contours[prev_contour_id].counterclockwise() {
                    self.contours[contour_id].setClockwise();
                }
                else {
                    self.contours[contour_id].setCounterClockwise();
                }
            }
            else if let Some(parent) = holeOf[prev_contour_id] {
                holeOf[contour_id] = Some(parent);
                self.contours[contour_id].setExternal(false);
                self.contours[parent].addHole(contour_id);
                if self.contours[parent].counterclockwise() {
                    self.contours[contour_id].setClockwise();
                }
                else {
                    self.contours[contour_id].setCounterClockwise();
                }
            }
            else {
                self.contours[contour_id].setCounterClockwise();
            }
        }
    }
}

impl ops::Index<usize> for Polygon {
    type Output = Contour;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.contours[index];
    }
}
impl ops::IndexMut<usize> for Polygon {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.contours[index];
    }
}


// -----------------------------------------------------------------------------
// booleanop

type PolygonIndex = usize;

#[derive(Clone, Debug)]
struct SegmentPacket {
    polygon_index: PolygonIndex,
    edge_type: EdgeType,
    up_faces_outwards: bool,
    is_outside_other_poly: bool,
    // Which original polygons contain this segment?  (The polygon this segment came from is always
    // assumed to contain it; which side faces outwards is determined by up_faces_outwards.)
    in_polygons: BitVec,

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
    fn new(polygon_index: usize, npolygons: usize) -> Self {
        return SegmentPacket{
            polygon_index,
            edge_type: EdgeType::Normal,
            up_faces_outwards: false,
            is_outside_other_poly: false,
            in_polygons: BitVec::from_elem(npolygons, false),

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
        return match self.1 {
            SegmentEnd::Left => packet.left_processed,
            SegmentEnd::Right => packet.right_processed,
        };
    }

    fn mark_processed(&self) {
        let mut packet = self.0.data.borrow_mut();
        match self.1 {
            SegmentEnd::Left => { packet.left_processed = true; }
            SegmentEnd::Right => { packet.right_processed = true; }
        }
    }

    fn is_one_sided(&self) -> bool {
        return self.0.is_one_sided();
    }

    fn faces_void(&self) -> bool {
        let packet = self.0.data.borrow();
        // We face into the void iff we're in no other polygons, and we're the side of the segment
        // facing outwards from our original polygon.
        return self.is_one_sided() &&
            (self.1 == SegmentEnd::Left) == packet.up_faces_outwards;
    }
}

type BoolSweepSegment = SweepSegment<RefCell<SegmentPacket>>;
impl BoolSweepSegment {
    fn is_one_sided(&self) -> bool {
        return self.data.borrow().in_polygons.none();
    }
}

/** @brief compute several fields of left event le */
fn compute_fields(operation: BooleanOpType, segment: &BoolSweepSegment, maybe_below: Option<&BoolSweepSegment>) {
    // anon scope so the packet goes away at the end and we can reborrow to call is_in_result
    {
        let mut packet = segment.data.borrow_mut();

        // Compute up_faces_outwards and is_outside_other_poly, the fields that tell us which
        // segments to include in the final polygon
        if let Some(below) = maybe_below {
            let below_packet = below.data.borrow();
            if packet.polygon_index == below_packet.polygon_index {
                // below line segment in sl belongs to the same polygon that "se" belongs to
                packet.up_faces_outwards = ! below_packet.up_faces_outwards;
                packet.is_outside_other_poly = below_packet.is_outside_other_poly;

                // This doesn't change containment at all
                // TODO hmm well this is ugly, and /possibly/ unnecessary
                packet.in_polygons.clear();
                packet.in_polygons.union(&below_packet.in_polygons);
            }
            else {
                // below line segment in sl belongs to a different polygon that "se" belongs to
                packet.up_faces_outwards = ! below_packet.is_outside_other_poly;
                if below.vertical() {
                    packet.is_outside_other_poly = ! below_packet.up_faces_outwards;
                }
                else {
                    packet.is_outside_other_poly = below_packet.up_faces_outwards;
                }

                // The other polygon should either be added or removed, depending on which way the
                // below segment faces
                packet.in_polygons.clear();
                packet.in_polygons.union(&below_packet.in_polygons);
                let is_outside = packet.is_outside_other_poly;
                packet.in_polygons.set(below_packet.polygon_index, ! is_outside);
                let polygon_index = packet.polygon_index;
                packet.in_polygons.set(polygon_index, false);
            }

            // compute below_in_result field
            if below.vertical() || ! is_in_result(operation, below) {
                packet.below_in_result = below_packet.below_in_result;
            }
            else {
                packet.below_in_result = Some(below.index);
            }
        }
        else {
            packet.up_faces_outwards = false;
            packet.is_outside_other_poly = true;

            packet.in_polygons.clear();
            let polygon_index = packet.polygon_index;
            let is_outside = packet.is_outside_other_poly;
            //packet.in_polygons.set(polygon_index, is_outside);
        }
    }

    let is_in_result = is_in_result(operation, segment);
    segment.data.borrow_mut().is_in_result = is_in_result;
}

/* Check whether a segment should be included in the final polygon */
fn is_in_result(operation: BooleanOpType, segment: &BoolSweepSegment) -> bool {
    let packet = segment.data.borrow();
    return match packet.edge_type {
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
            return (0, None, None);
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

            return (1, pt1, pt2);
        }
        SegmentIntersection::Segment(a, b) => {
            if seg1.data.borrow().polygon_index == seg2.data.borrow().polygon_index {
                // the line segments overlap, but they belong to the same polygon
                // FIXME would be nice to bubble up a Result i guess
                println!("hm, what, is happening here...");
                println!("seg1: {:?}", seg1);
                println!("seg2: {:?}", seg2);
                println!("overlap: {}, {}", a, b);
                //panic!("Sorry, edges of the same polygon overlap");
            }

            // The line segments associated to le1 and le2 overlap
            let left_coincide = seg1.left_point == seg2.left_point;
            let right_coincide = seg1.right_point == seg2.right_point;
            let left_cmp = Ord::cmp(
                &SweepEndpoint(seg1, SegmentEnd::Left),
                &SweepEndpoint(seg2, SegmentEnd::Left));
            let right_cmp = Ord::cmp(
                &SweepEndpoint(seg1, SegmentEnd::Right),
                &SweepEndpoint(seg2, SegmentEnd::Right));

            if left_coincide {
                // Segments share a left endpoint, and may even be congruent
                let edge_type1 = EdgeType::NonContributing;
                let edge_type2 = if
                    seg1.data.borrow().up_faces_outwards ==
                    seg2.data.borrow().up_faces_outwards
                    { EdgeType::SameTransition }
                else { EdgeType::DifferentTransition };

                {
                    println!("due to split, setting #{} to {:?} and #{} to {:?}", seg1.index, edge_type1, seg2.index, edge_type2);
                    seg1.data.borrow_mut().edge_type = edge_type1;
                    seg2.data.borrow_mut().edge_type = edge_type2;
                }

                // If the right endpoints don't coincide, then one lies on the other segment
                // and needs to split it
                match right_cmp {
                    Ordering::Less =>    return (2, None, Some(seg1.right_point)),
                    Ordering::Greater => return (2, Some(seg2.right_point), None),
                    Ordering::Equal =>   unreachable!(),
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
    let mut start_index;
    if current_endpoint.1 == SegmentEnd::Left {
        start_index = current_endpoint.0.data.borrow().right_index;
    }
    else {
        start_index = current_endpoint.0.data.borrow().left_index;
    };

    while start_index > 0 && included_endpoints[start_index - 1].point() == next_point {
        start_index -= 1;
    }

    // Find the closest angle.  That means the biggest dot product, or the smallest, maybe.
    let mut closest_dot = f32::NAN;
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

            if closest_endpoint.is_none() || (this_ccw && dot > closest_dot) || (!this_ccw && dot < closest_dot) {
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

    // Hm, well, we didn't find one, so...  go backwards to the next unprocessed segment
    // period?  This doesn't make a lot of sense to me; are we just assuming we'll hit one that
    // shares a point?  TODO explicitly check for that perhaps?
    // XXX i can see it especially failing for degenerate cases where there's only one segment
    // in a polygon, oof.  though then there should still be two segments actually...?
    // XXX also this will panic on underflow (good, but maybe needs more explicit error handling)
    /*
    let mut i = start_index - 1;
    while included_endpoints[i].0.data.borrow().processed {
        i -= 1;
    }
    // TODO this might return a bogus point...!  need to check if endpoint.point() == next_point.
    // it SHOULD work, but i don't know that i can absolutely guarantee it if the input is garbage
    return &included_endpoints[i];
    */
}

// TODO what if i have a slice of polygons, or a vec of &Polygons, or...?
pub fn compute(polygons: &Vec<Polygon>, operation: BooleanOpType) -> Polygon {
    // ---------------------------------------------------------------------------------------------
    // Detect trivial cases that can be answered without doing any work

    /*
     * TODO restore these...  not entirely clear how/whether they'd apply in a world with n input
     * polygons and no fixed operations
    let subjectBB = subject.bbox();     // for optimizations 1 and 2
    let clippingBB = clipping.bbox();   // for optimizations 1 and 2
    let MINMAXX = f32::min(subjectBB.max_x(), clippingBB.max_x()); // for optimization 2

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
    for (i, polygon) in polygons.iter().enumerate() {
        for contour in &polygon.contours {
            for seg in contour.iter_segments() {
            /*  if (s.degenerate ()) // if the two edge endpoints are equal the segment is dicarded
                    return;          // This can be done as preprocessing to avoid "polygons" with less than 3 edges */
                let segment: &_ = arena.alloc(SweepSegment::new(
                    seg.source, seg.target, segment_id, RefCell::new(SegmentPacket::new(i, polygons.len()))));
                segment_id += 1;
                segment_order.push(segment);
                endpoint_queue.push(Reverse(SweepEndpoint(segment, SegmentEnd::Left)));
                endpoint_queue.push(Reverse(SweepEndpoint(segment, SegmentEnd::Right)));
                svg_orig_group.append(
                    Line::new()
                    .set("x1", seg.source.x)
                    .set("y1", -seg.source.y)
                    .set("x2", seg.target.x)
                    .set("y2", -seg.target.y)
                    .set("stroke", if i == 0 { "#aaa" } else { "#ddd" })
                    .set("stroke-width", 1)
                );
                svg_orig_group.append(
                    Text::new()
                    .add(svg::node::Text::new(format!("{}", segment_id - 1)))
                    .set("x", (seg.source.x + seg.target.x) / 2.0)
                    .set("y", -(seg.source.y + seg.target.y) / 2.0)
                    .set("text-anchor", "middle")
                    .set("alignment-baseline", "central")
                    .set("font-size", 8)
                );
            }
        }
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
                println!("ah!  splitting #{} at {:?}, into #{} and #{}", seg.index, pt, segment_id, segment_id + 1);
                // It's not obvious at a glance, but in the original algorithm, the left end of a
                // split inherits the original segment's data, and the right end gets data fresh.
                // This is important since handle_intersections assigns the whole segment's
                // edge_type before splitting (and, TODO, maybe it shouldn't) but the right end
                // isn't meant to inherit that!
                let left: &_ = arena.alloc(SweepSegment::new(
                    seg.left_point, pt, segment_id, seg.data.clone()));
                segment_id += 1;
                let right: &_ = arena.alloc(SweepSegment::new(
                    pt, seg.right_point, segment_id, RefCell::new(SegmentPacket::new(seg.data.borrow().polygon_index, polygons.len()))));
                {
                    // TODO ugly ass copy
                    let mut packet = right.data.borrow_mut();
                    packet.in_polygons.clear();
                    packet.in_polygons.union(&left.data.borrow().in_polygons);
                }
                segment_id += 1;

                segment_order.push(left);
                segment_order.push(right);
                // We split this segment in half, so replace the existing segment with its left end
                // and give it a faux right endpoint
                active_segments.remove(&seg);
                active_segments.insert(left);
                endpoint_queue.push(Reverse(SweepEndpoint(left, SegmentEnd::Right)));
                endpoint_queue.push(Reverse(SweepEndpoint(right, SegmentEnd::Left)));
                endpoint_queue.push(Reverse(SweepEndpoint(right, SegmentEnd::Right)));

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


    loop {
        // Grab the next endpoint, or bail if we've run out
        let Reverse(SweepEndpoint(mut segment, end)) = match endpoint_queue.pop() {
            Some(item) => item,
            None => break,
        };
        println!("");
        println!("LOOP ITERATION: {:?} of #{:?} {:?} -> {:?}", end, segment.index, segment.left_point, segment.right_point);
        for seg in &active_segments {
            println!("  {} #{} {:?} -> {:?}", if seg < &segment { "<" } else if seg > &segment { ">" } else { "=" }, seg.index, seg.left_point, seg.right_point);
        }
        let endpoint = match end {
            SegmentEnd::Left => segment.left_point,
            SegmentEnd::Right => segment.right_point,
        };
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

                let maybe_below = active_segments.range(..segment).last().map(|v| *v);
                let maybe_above = active_segments.range(segment..).next().map(|v| *v);
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
        let mut maybe_below = active_segments.range(..segment).last().map(|v| *v);
        let mut maybe_above = active_segments.range(segment..).next().map(|v| *v);
        active_segments.insert(segment);
        compute_fields(operation, segment, maybe_below);
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
            compute_fields(operation, segment, maybe_below);
            compute_fields(operation, maybe_above.unwrap(), Some(segment));
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
            compute_fields(operation, maybe_below.unwrap(), active_segments.range(..maybe_below.unwrap()).last().map(|v| *v));
            compute_fields(operation, segment, maybe_below);
        }
    }

    println!("");
    println!("---MAIN LOOP DONE ---");
    println!("");

    {
    let mut svg_swept_group = Group::new();
    for segment in &swept_segments {
        svg_swept_group.append(
            Line::new()
            .set("x1", segment.left_point.x)
            .set("y1", -segment.left_point.y)
            .set("x2", segment.right_point.x)
            .set("y2", -segment.right_point.y)
            .set("stroke", "green")
            .set("stroke-width", 1)
        );
        let mut in_polys = Vec::new();
        for i in 0..polygons.len() {
            if segment.data.borrow().in_polygons[i] {
                in_polys.push(i);
            }
        }
        svg_swept_group.append(
            Text::new()
            .add(svg::node::Text::new(format!("{} {:?}", segment.index, in_polys)))
            .set("x", (segment.left_point.x + segment.right_point.x) / 2.0)
            .set("y", -(segment.left_point.y + segment.right_point.y) / 2.0)
            .set("fill", "darkgreen")
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
    svg::save("idchoppers-shapeops-debug.svg", &doc);
    }

    // connect the solution edges to build the result polygon
    // copy the events in the result polygon to included_points array
    // XXX since otherEvent is still kosher, i don't think this is a copy!
    let count = swept_segments.len();
    let mut included_segments = Vec::with_capacity(count);
    let mut included_endpoints = Vec::with_capacity(count * 2);
    for segment in swept_segments.into_iter() {
        if segment.data.borrow().is_in_result || true {
            included_segments.push(segment);
            included_endpoints.push(SweepEndpoint(segment, SegmentEnd::Left));
            included_endpoints.push(SweepEndpoint(segment, SegmentEnd::Right));
        }
    }
    included_segments.sort();
    included_endpoints.sort();

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
    let mut depth = Vec::new();
    let mut holeOf = Vec::new();
    for (i, endpoint) in included_endpoints.iter().enumerate() {
        let &SweepEndpoint(segment, end) = endpoint;
        if endpoint.is_processed() {
            continue;
        }

        // FIXME maybe do this in previous loop
        if segment.data.borrow().in_polygons.none() {
            if end == SegmentEnd::Left && segment.data.borrow().up_faces_outwards {
                segment.data.borrow_mut().left_processed = true;
                continue;
            }
            else if end == SegmentEnd::Right && ! segment.data.borrow().up_faces_outwards {
                segment.data.borrow_mut().right_processed = true;
                continue;
            }
        }

        // Walk around looking for a polygon until we come back to the starting point
        let mut contour = Contour::new();
        let contourId = final_polygon.contours.len();
        let starting_point = segment.left_point;
        contour.add(starting_point);
        let mut current_endpoint = &included_endpoints[segment.data.borrow().left_index];
        println!("building a contour from #{} {:?} {:?}", segment.index, end, starting_point);
        loop {
            current_endpoint.mark_processed();
            let current_segment = current_endpoint.0;
            println!(" -> from polygon {}, others {:?}", current_segment.data.borrow().polygon_index, current_segment.data.borrow().in_polygons);
            {
                let mut packet = current_segment.data.borrow_mut();
                if current_endpoint.1 == SegmentEnd::Left {
                    packet.left_contour_id = Some(contourId);
                }
                else {
                    packet.right_contour_id = Some(contourId);
                }
                packet.result_in_out = current_endpoint.1 == SegmentEnd::Right;
            }

            let point = current_endpoint.other_point();
            if point == starting_point {
                break;
            }
            contour.add(point);

            current_endpoint = find_next_segment(current_endpoint, &included_endpoints);
            println!("... #{} {:?}", current_endpoint.0.index, current_endpoint.point());
        }

        final_polygon.contours.push(contour);
        depth.push(0);
        holeOf.push(None);

        // FIXME if we go *clockwise* here (because we're tracing a hole), then the segment
        // below us is the first segment of the equivalent *counterclockwise* shape (which
        // we'll never trace because we eliminated it), so we need to actually use the segment
        // below *that* to figure out what we're inside.  also, the winding will be wrong,
        // because we call changeOrientation below assuming that everything is ccw!
        // TODO maybe everything should just be counterclockwise?  but then how do i track
        // whether i'm "inside" or "outside"?
        // XXX can this happen for two-sided lines as well?
        let &SweepEndpoint(segment, end) = if final_polygon[contourId].clockwise() {
            current_endpoint
        }
        else {
            endpoint
        };

        // TODO wait, how much does this resemble the hole-finding stuff in Polygon?
        // Check for a hole inside another contour, or an island inside another hole.
        // Note that we should've already traced the segment below this one, if any, since we're
        // tracing in order from bottom to top.
        println!("checking containment");
        if end == SegmentEnd::Left && ! segment.is_one_sided() {
            // This is the left/upper side of a two-sided segment, so it must be a neighbor of the
            // contour on our right side, which has already been traced.
            let neighbor_contour_id = segment.data.borrow().right_contour_id.unwrap();
            println!("...it's the left side of a two-sided line, so it's the neighbor of the other side, contour {:?}", neighbor_contour_id);
            holeOf[contourId] = holeOf[neighbor_contour_id];
            depth[contourId] = depth[neighbor_contour_id];
            if let Some(parent_contour_id) = holeOf[neighbor_contour_id] {
                final_polygon[parent_contour_id].addHole(contourId);
                final_polygon.contours[contourId].setExternal(false);
            }
        }
        // XXX: if this is the right side, there MUST be a below
        else if let Some(below_segment_id) = segment.data.borrow().below_in_result {
            println!("...it's got a below, #{}, so something must be going on here", below_segment_id);
            // This is either the right/lower side of a two-sided segment, or the only side of a
            // one-sided segment.  Either way, its containment is determined by the segment below.
            // TODO this is the ONLY PLACE that uses segment_order, or segment index at all!
            let below_segment = &segment_order[below_segment_id];
            
            // If this is the left side (of a one-sided segment) and the segment below it is an
            // island (or a top-level contour), then this is a neighbor.
            if end == SegmentEnd::Left && depth[below_segment.data.borrow().right_contour_id.unwrap()] % 2 == 0 {
                // FIXME uggh this is exactly copy-pasted from above
                // XXX also this was detected in the original code by looking at external()?
                let neighbor_contour_id = below_segment.data.borrow().right_contour_id.unwrap();
                println!("...ah, a neighbor of contour {:?}", neighbor_contour_id);
                holeOf[contourId] = holeOf[neighbor_contour_id];
                depth[contourId] = depth[neighbor_contour_id];
                if let Some(parent_contour_id) = holeOf[neighbor_contour_id] {
                    final_polygon[parent_contour_id].addHole(contourId);
                    final_polygon.contours[contourId].setExternal(false);
                }
            }
            // Otherwise, it's a hole/lake inside the contour below.
            else {
                let below_contour_id = below_segment.data.borrow().left_contour_id.or(below_segment.data.borrow().right_contour_id).unwrap();
                println!("...ah, it's inside contour {:?}", below_contour_id);
                final_polygon[below_contour_id].addHole(contourId);
                holeOf[contourId] = Some(below_contour_id);
                depth[contourId] = depth[below_contour_id] + 1;
                final_polygon.contours[contourId].setExternal(false);
            }
        }
        else if end == SegmentEnd::Right {
            panic!("right/lower side MUST have a lower neighbor!");
        }

        if depth[contourId] & 1 == 1 {
            final_polygon[contourId].changeOrientation();
        }
    }

    return final_polygon;
}
