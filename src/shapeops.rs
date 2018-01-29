// hey what's up, this is a catastrophic mess ported from a paper:
// A simple algorithm for Boolean operations on polygons (2013), Martínez et al.
// TODO clean this up
// TODO rename it this name is bad
// TODO finish, it?
use std::cmp;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::collections::LinkedList;
use std::ops;

use std::cell::Cell;


use euclid::TypedPoint2D;
use euclid::TypedRect;
use euclid::TypedSize2D;


pub struct MapSpace;
pub type MapPoint = TypedPoint2D<f32, MapSpace>;
pub type MapRect = TypedRect<f32, MapSpace>;
pub type MapSize = TypedSize2D<f32, MapSpace>;
// TODO honestly a lot of this could be made generic over TypedPoint2D

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


#[derive(PartialEq, Eq)]
enum BooleanOpType {
    Intersection,
    Union,
    Difference,
    ExclusiveOr,
}
#[derive(PartialEq, Eq)]
enum EdgeType {
    Normal,
    NonContributing,
    SameTransition,
    DifferentTransition,
}
enum PolygonType {
    Subject = 0,
    Clipping = 1,
}

/*************************************************************************************************************
 * The following code is necessary for implementing the computeHoles member function
 * **********************************************************************************************************/
// XXX this was copy/pasted into polygon from booleanop lol

struct SweepEvent {
    // is point the left endpoint of the edge (point, otherEvent->point)?
    left: bool,
	// point associated with the event
    point: MapPoint,
    other_point: MapPoint,
	// event associated to the other endpoint of the edge
    //otherEvent: &'a mut SweepEvent<'a, T>,
	// Polygon to which the associated segment belongs to
    // For Polygon::findHoles or whatever, this is a contour index (?!); for the main logic, this
    // is actually PolygonType (0 for subject, 1 for clipping)
    pol: usize,
    edge_type: EdgeType,

    // Arbitrary index for linking this to the segment it came from
    segment_index: usize,

	//The following fields are only used in "left" events
	/**  Does segment (point, otherEvent->p) represent an inside-outside transition in the polygon for a vertical ray from (p.x, -infinite)? */
	down_faces_outwards: bool,
    // down_faces_outwards transition for the segment from the other polygon preceding this segment in sl
	otherInOut: bool,
    // Position of the event (line segment) in sl
    // XXX oh no oh no
    posSL: usize,
    // previous segment in sl belonging to the result of the boolean operation
	//prevInResult: Option<&'a SweepEvent<'a>>,
	inResult: bool,
    // XXX only used in connectEdges
	pos: usize,
}

impl SweepEvent {
    fn new(left: bool, point: MapPoint, other_point: MapPoint, segment_index: usize) -> Self {
        return SweepEvent{
            left,
            point,
            other_point,
            segment_index,
            pol: 0,
            edge_type: EdgeType::Normal,
            down_faces_outwards: false,
            otherInOut: false,
            posSL: 0,  // FIXME almost certainly wrong
            inResult: false,
            pos: 0,
        };
    }

	/** Is the line segment (point, other_poin) below point p */
	fn below(&self, p: MapPoint) -> bool {
        if self.left {
            return triangle_signed_area(self.point, self.other_point, p) > 0.;
        }
        else {
            return triangle_signed_area(self.other_point, self.point, p) > 0.;
        }
    }

	/** Is the line segment (point, other_poin) above point p */
	fn above(&self, p: MapPoint) -> bool {
        return ! self.below(p);
    }

	/** Is the line segment (point, other_poin) a vertical line segment */
	fn vertical(&self) -> bool {
        return self.point.x == self.other_point.x;
    }

	/** Return the line segment associated to the SweepEvent */
	fn segment(&self) -> Segment2 {
        return Segment2::new(self.point, self.other_point);
    }
}

// Compare two sweep events
// Return true means that e1 is placed at the event queue after e2, i.e,, e1 is processed by the algorithm after e2
// FIXME this was defined backwards i'm pretty sure?  very confusing; i think it's because this is
// used with a binary heap which puts the greatest thing at the top!  but in computeHoles it's
// backwards since that's a regular set, so the sweep line actually moves leftwards...
fn compare_by_sweep(e1: &SweepEvent, e2: &SweepEvent) -> Ordering {
    return e2.point.x.partial_cmp(&e1.point.x).unwrap()
        .then(e2.point.y.partial_cmp(&e1.point.y).unwrap())
        // Same point, but one is a left endpoint and the other a right endpoint. The right endpoint is processed first
        .then_with(|| if ! e1.left && e2.left { Ordering::Greater } else if ! e1.left && e2.left { Ordering::Less } else { Ordering::Equal })
        // Same point, both events are left endpoints or both are right endpoints.
        .then_with(||
            if triangle_signed_area(e1.point, e1.other_point, e2.other_point) != 0. {
                // not collinear; the event associate to the bottom segment is processed first
                if e1.above(e2.other_point) {
                    return Ordering::Less;
                }
                else {
                    return Ordering::Greater;
                }
            }
            else {
            	return e2.pol.cmp(&e1.pol);
            }
        );
}

// This is used for the binary tree ordering in the sweep line algorithm, keeping sweep events in
// order by where on the y-axis they would cross the sweep line
fn compare_by_segment(le1: &SweepEvent, le2: &SweepEvent) -> Ordering {
	if le1 == le2 {
		return Ordering::Equal;
    }
	if triangle_signed_area(le1.point, le1.other_point, le2.point) == 0. &&
		triangle_signed_area(le1.point, le1.other_point, le2.other_point) == 0.
    {
        // Segments are collinear.  Sort by some arbitrary consistent criteria
        return le1.pol.cmp(&le2.pol)
            .then_with(|| compare_by_sweep(le1, le2));
    }

    if le1.point == le2.point {
        // Both segments have the same left endpoint.  Sort on the right endpoint
        // TODO le1.below() just checks triangle_signed_area again, same as above
        if le1.below(le2.other_point) {
            return Ordering::Less;
        }
        else {
            return Ordering::Greater;
        }
    }

    // has the segment associated to e1 been sorted in evp before the segment associated to e2?
    if compare_by_sweep(le1, le2) == Ordering::Less {
        if le1.below(le2.point) {
            return Ordering::Less;
        }
        else {
            return Ordering::Greater;
        }
    }
    // The segment associated to e2 has been sorted in evp before the segment associated to e1
    if le2.above(le1.point) {
        return Ordering::Less;
    }
    else {
        return Ordering::Greater;
    }
}

impl cmp::PartialEq for SweepEvent {
    fn eq(&self, other: &SweepEvent) -> bool {
        return self.cmp(other) == Ordering::Equal;
    }
}
impl cmp::Eq for SweepEvent { }

impl cmp::PartialOrd for SweepEvent {
    fn partial_cmp(&self, other: &SweepEvent) -> Option<Ordering> {
        return Some(self.cmp(other));
    }
}

impl cmp::Ord for SweepEvent {
    fn cmp(&self, other: &SweepEvent) -> Ordering {
        return compare_by_sweep(self, other);
    }
}


#[derive(Debug, PartialEq, Eq)]
enum SegmentEnd {
    Left,
    Right,
}

#[derive(Debug)]
struct SweepEndpoint {
    end: SegmentEnd,
    point: MapPoint,
    other_point: MapPoint,
    index: usize,
    segment_index: usize,
}
//
// Compare two sweep events
impl cmp::PartialEq for SweepEndpoint {
    fn eq(&self, other: &SweepEndpoint) -> bool {
        return self.cmp(other) == Ordering::Equal;
    }
}
impl cmp::Eq for SweepEndpoint { }

impl cmp::PartialOrd for SweepEndpoint {
    fn partial_cmp(&self, other: &SweepEndpoint) -> Option<Ordering> {
        return Some(self.cmp(other));
    }
}

impl cmp::Ord for SweepEndpoint {
    fn cmp(&self, other: &SweepEndpoint) -> Ordering {
        if self as *const _ == other as *const _ {
            return Ordering::Equal;
        }

        return self.point.x.partial_cmp(&other.point.x).unwrap()
            .then(self.point.y.partial_cmp(&other.point.y).unwrap())
            .then_with(|| {
                // If the points coincide, a right endpoint takes priority
                if self.end == other.end {
                    Ordering::Equal
                }
                else if self.end == SegmentEnd::Right {
                    Ordering::Less
                }
                else {
                    Ordering::Greater
                }
            })
            .then_with(
                // Same point, same end of their respective segments.  Use triangle area to give
                // priority to the bottom point
                || triangle_signed_area(other.point, other.other_point, self.other_point).partial_cmp(&0.).unwrap()
            )
            .then_with(
                // Collinear!  Fall back to something totally arbitrary
                // NOTE this used to be pol, unsure if this makes any real difference
                || self.segment_index.cmp(&other.segment_index)
            );
    }
}


#[derive(Debug)]
struct SweepSegment<T> {
    left: SweepEndpoint,
    right: SweepEndpoint,
    faces_outwards: bool,
    index: usize,
    data: T,
}

impl<T> SweepSegment<T> {
    fn new(point0: MapPoint, point1: MapPoint, index: usize, data: T) -> SweepSegment<T> {
        let (left, right, faces_outwards) = if point0.x < point1.x {
            (point0, point1, false)
        }
        else {
            (point1, point0, true)
        };

        return SweepSegment{
            left: SweepEndpoint{
                end: SegmentEnd::Left,
                point: left,
                other_point: right,
                index: 0,
                segment_index: index,
            },
            right: SweepEndpoint{
                end: SegmentEnd::Right,
                point: right,
                other_point: left,
                index: 0,
                segment_index: index,
            },
            // XXX this feels backwards to me; down_faces_outwards is only true if this is the left
            // endpoint of a left-pointing segment, which for a CCW contour is actually the
            // border between outside and inside!  maybe this thinks negative infinity is
            // upwards?
            faces_outwards,
            index,
            data,
        };
    }

	/** Is the line segment (point, other_poin) below point p */
	fn below(&self, p: MapPoint) -> bool {
        return triangle_signed_area(self.left.point, self.right.point, p) > 0.;
    }

	/** Is the line segment (point, other_poin) above point p */
	fn above(&self, p: MapPoint) -> bool {
        return ! self.below(p);
    }

	/** Is the line segment (point, other_poin) a vertical line segment */
	fn vertical(&self) -> bool {
        return self.left.point.x == self.right.point.x;
    }

	/** Return the line segment */
	fn segment(&self) -> Segment2 {
        return Segment2::new(self.left.point, self.right.point);
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

        if triangle_signed_area(self.left.point, self.right.point, other.left.point) == 0. &&
            triangle_signed_area(self.left.point, self.right.point, other.right.point) == 0.
        {
            // Segments are collinear.  Sort by some arbitrary consistent criteria
            // XXX this used to be pol, which would cause far more ties...  is this ok?  does
            // "consistent" mean it actually needs to use the sweep comparison?
            return self.index.cmp(&other.index);
                //.then_with(|| compare_by_sweep(self, other));
        }

        if self.left.point == other.left.point {
            // Both segments have the same left endpoint.  Sort on the right endpoint
            // TODO self.below() just checks triangle_signed_area again, same as above
            if self.below(other.right.point) {
                return Ordering::Less;
            }
            else {
                return Ordering::Greater;
            }
        }

        // has the segment associated to e1 been sorted in evp before the segment associated to e2?
        if self.left < other.left {
            if self.below(other.left.point) {
                return Ordering::Less;
            }
            else {
                return Ordering::Greater;
            }
        }
        else {
            // The segment associated to e2 has been sorted in evp before the segment associated to e1
            if other.above(self.left.point) {
                return Ordering::Less;
            }
            else {
                return Ordering::Greater;
            }
        }
    }
}

struct Sweep<'a, T> {
    segments: Vec<SweepSegment<T>>,
    endpoints: Vec<&'a SweepEndpoint>,
}

impl<'a, T> Sweep<'a, T> {
    fn add_segment(&mut self, point0: MapPoint, point1: MapPoint) {

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
        let mut bbox = MapRect::new(self.points[0], MapSize::zero());
        // FIXME skip the first one.  actually this could probably be written better somehow.  reduce?
        for &vertex in self.points.iter() {
            bbox = bbox.union(&MapRect::new(vertex, MapSize::zero()));
        }
        return bbox;
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

struct SegmentHandle<'a>(usize, &'a Vec<usize>);
impl<'a> cmp::PartialEq for SegmentHandle<'a> {
    fn eq(&self, other: &SegmentHandle<'a>) -> bool {
        return self.cmp(other) == Ordering::Equal;
    }
}
impl<'a> cmp::Eq for SegmentHandle<'a> { }
impl<'a> cmp::PartialOrd for SegmentHandle<'a> {
    fn partial_cmp(&self, other: &SegmentHandle<'a>) -> Option<Ordering> {
        return Some(self.cmp(other));
    }
}
impl<'a> cmp::Ord for SegmentHandle<'a> {
    fn cmp(&self, other: &SegmentHandle<'a>) -> Ordering {
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

    fn bbox(&self) -> MapRect {
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

    pub fn computeHoles(&mut self) {
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

        // OK, now we can grab and sort the events themselves in sweep order, which is x-wards
        let segments = segments_mut;  // kill mutability so refs stay valid
        let mut events = Vec::with_capacity(segments.len() * 2);
        for segment in &segments {
            events.push(&segment.left);
            events.push(&segment.right);
        }
        events.sort();

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
        for event in &events {
            // Stop if we've seen every contour
            if nprocessed >= self.contours.len() {
                break;
            }

            let segment = &segments[event.segment_index];
            if event.end == SegmentEnd::Right {
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

