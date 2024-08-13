#ifndef MESHTRIM1_H
#define MESHTRIM1_H
#include <Eigen/Dense>
#include <iostream>
#define _USE_MATH_DEFINES
#include < math.h >
// -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Geometry/EigenVectorT.hh>

#include <map>
#include <unordered_map> 
#include <assert.h>
#include <chrono>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#include <nanoflann/nanoflann.hpp>
#include "nanoflann/utils.h"

namespace MeshTrim {
    using namespace std::chrono;
    using namespace Eigen;
    using namespace nanoflann;
    namespace fs = std::experimental::filesystem;

	typedef KDTreeSingleIndexDynamicAdaptor<
		L2_Simple_Adaptor<double, PointCloud<double> >,
		PointCloud<double>,
		3 /* dim */
	> my_kd_tree_t;
	namespace fs = std::experimental::filesystem;

	struct EigenTraits : OpenMesh::DefaultTraits {
		using Point = Eigen::Vector3f;
		using TexCoord2D = Eigen::Vector2f;
	};
	using EigenTriMesh = OpenMesh::TriMesh_ArrayKernelT<EigenTraits>;

	class FaceInter {
	public:
		int type = 0; // [0,1,2,3,4,5,6,7] see details in figure
		int fid;
		Vector3f v1, v2, v3;
		Vector2f vt1, vt2, vt3;
		int v1_id, v2_id, v3_id;
		bool v1_inside, v2_inside, v3_inside;
		std::vector<Vector3f> e12_pts,e23_pts,e31_pts,inside_pts;
	};

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// Two line segments intersect /////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// Function to calculate the orientation of three points (x1, y1), (x2, y2), (x3, y3)
	int orientation(float x1, float y1, float x2, float y2, float x3, float y3) {
		float val = (y2 - y1) * (x3 - x2) - (x2 - x1) * (y3 - y2);
		if (val == 0) return 0; // Collinear
		return (val > 0) ? 1 : 2; // Clockwise or counterclockwise
	}

	// Function to check if point (x2, y2) lies on segment (x1, y1), (x3, y3)
	bool onSegment(float x1, float y1, float x2, float y2, float x3, float y3) {
		return (x2 <= std::max(x1, x3) && x2 >= std::min(x1, x3) &&
			y2 <= std::max(y1, y3) && y2 >= std::min(y1, y3));
	}

	// Function to calculate the intersection point of two lines
	void intersectionPoint(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, float& ix, float& iy) {
		float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
		if (denom == 0) {
			ix = iy = 0; // lines are parallel
			return;
		}

		ix = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
		iy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
	}

	// Function to check if two line segments intersect
	bool doIntersect(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, float& ix, float& iy) {
		int o1 = orientation(x1, y1, x2, y2, x3, y3);
		int o2 = orientation(x1, y1, x2, y2, x4, y4);
		int o3 = orientation(x3, y3, x4, y4, x1, y1);
		int o4 = orientation(x3, y3, x4, y4, x2, y2);

		// General case
		if (o1 != o2 && o3 != o4) {
			intersectionPoint(x1, y1, x2, y2, x3, y3, x4, y4, ix, iy);
			return true;
		}

		// Special Cases
		// (x1, y1), (x2, y2) and (x3, y3) are collinear and (x3, y3) lies on segment (x1, y1), (x2, y2)
		if (o1 == 0 && onSegment(x1, y1, x3, y3, x2, y2)) {
			ix = x3;
			iy = y3;
			return true;
		}

		// (x1, y1), (x2, y2) and (x4, y4) are collinear and (x4, y4) lies on segment (x1, y1), (x2, y2)
		if (o2 == 0 && onSegment(x1, y1, x4, y4, x2, y2)) {
			ix = x4;
			iy = y4;
			return true;
		}

		// (x3, y3), (x4, y4) and (x1, y1) are collinear and (x1, y1) lies on segment (x3, y3), (x4, y4)
		if (o3 == 0 && onSegment(x3, y3, x1, y1, x4, y4)) {
			ix = x1;
			iy = y1;
			return true;
		}

		// (x3, y3), (x4, y4) and (x2, y2) are collinear and (x2, y2) lies on segment (x3, y3), (x4, y4)
		if (o4 == 0 && onSegment(x3, y3, x2, y2, x4, y4)) {
			ix = x2;
			iy = y2;
			return true;
		}

		return false; // Doesn't fall in any of the above cases
	}

	void edge_intersect_bbox1(Eigen::Vector3f v1, Eigen::Vector3f v2, std::vector<float> bbox, std::vector<Eigen::Vector3f>& edge_pts) {
		edge_pts.clear();
		float ix, iy, ratio;
		bool status;
	
		Vector3f dir = v2 - v1;
		Vector3f inter_pt;
		// check left bound
		status = doIntersect(v1[0], v1[1], v2[0], v2[1], bbox[0], bbox[2], bbox[0], bbox[3], ix, iy);
		if (status) {
			ratio = abs((ix - v1[0]) / (v2[0] - v1[0]));
			inter_pt = v1 + dir * ratio;
			edge_pts.push_back(inter_pt);
		}
		// check right bound
		status = doIntersect(v1[0], v1[1], v2[0], v2[1], bbox[1], bbox[2], bbox[1], bbox[3], ix, iy);
		if (status) {
			ratio = abs((ix - v1[0]) / (v2[0] - v1[0]));
			inter_pt = v1 + dir * ratio;
			edge_pts.push_back(inter_pt);
		}
		// check bot bound
		status = doIntersect(v1[0], v1[1], v2[0], v2[1], bbox[0], bbox[2], bbox[1], bbox[2], ix, iy);
		if (status) {
			ratio = abs((ix - v1[0]) / (v2[0] - v1[0]));
			inter_pt = v1 + dir * ratio;
			edge_pts.push_back(inter_pt);
		}
		// check top bound
		status = doIntersect(v1[0], v1[1], v2[0], v2[1], bbox[0], bbox[3], bbox[1], bbox[3], ix, iy);
		if (status) {
			ratio = abs((ix - v1[0]) / (v2[0] - v1[0]));
			inter_pt = v1 + dir * ratio;
			edge_pts.push_back(inter_pt);
		}

		// order intersect points
		if (edge_pts.size() != 0) {
			std::vector<std::pair<float, Eigen::Vector3f>> inter_dis;
			for (auto item : edge_pts) {
				float dis = (item - v1).norm();
				inter_dis.push_back(std::make_pair(dis,item));
			}
			std::sort(inter_dis.begin(), inter_dis.end(), [](auto& left, auto& right) {
				return left.first < right.first;
				});
			edge_pts.clear();
			for (auto i : inter_dis) {
				edge_pts.push_back(i.second);
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	float area_triangle(Eigen::Vector3f pt1, Eigen::Vector3f pt2, Eigen::Vector3f pt3) {
		Eigen::Vector3f cross = (pt2 - pt1).cross(pt3 - pt1);
		float area = 0.5 * cross.norm();
		return area;
	}

	// Function to calculate the area of a triangle
	float triangleArea(float x1, float y1, float x2, float y2, float x3, float y3) {
		return std::abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0);
	}

	bool isPointInsideTriangle(float px, float py, float x1, float y1, float x2, float y2, float x3, float y3) {
		// Calculate the area of the triangle
		float area = triangleArea(x1, y1, x2, y2, x3, y3);

		// Calculate the area of the three smaller triangles formed by the point and the triangle edges
		float area1 = triangleArea(px, py, x1, y1, x2, y2);
		float area2 = triangleArea(px, py, x2, y2, x3, y3);
		float area3 = triangleArea(px, py, x3, y3, x1, y1);

		// If the sum of the areas of the smaller triangles is equal to the area of the original triangle, the point is inside
		float diff = std::abs(area - (area1 + area2 + area3));
		return  diff< 1e-3;
	}

	// Function to calculate the barycentric coordinates of a point with respect to a triangle
	void barycentricCoordinates(float x, float y, float x1, float y1, float x2, float y2, float x3, float y3, float& u, float& v, float& w) {
		float denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
		u = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator;
		v = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator;
		w = 1 - u - v;
	}

	// Function to calculate the Z value of a 2D coordinate given a triangle in 3D
	float calculateZ(float x, float y, float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3) {
		float u, v, w;
		barycentricCoordinates(x, y, x1, y1, x2, y2, x3, y3, u, v, w);
		return u * z1 + v * z2 + w * z3;
	}

	// Function to calculate the Z value of a 2D coordinate given a triangle in 3D
	Eigen::Vector2f calculateUV(Eigen::Vector3f q, Eigen::Vector3f v1, Eigen::Vector2f vt1, Eigen::Vector3f v2, Eigen::Vector2f vt2, Eigen::Vector3f v3, Eigen::Vector2f vt3) {
		float u, v, w;
		barycentricCoordinates(q[0], q[1], v1[0],v1[1], v2[0], v2[1], v3[0], v3[1], u, v, w);
		return u * vt1 + v * vt2 + w * vt3;
	}

	bool edge_intersect_bbox(std::vector<Eigen::Vector3f> pts, std::vector<float> bbox) {
		int status = false;
		Eigen::Vector3f pt1 = pts[0];
		Eigen::Vector3f pt2 = pts[1];
		float edge_len = (pt1 - pt2).norm();
		Eigen::Vector3f dir = (pt2 - pt1) / edge_len;

		//xmin plane
		if ((pt1[0] <= bbox[0] && pt2[0] <= bbox[0]) || (pt1[0] >= bbox[0] && pt2[0] >= bbox[0])) {
			int a = 0; // not intersect
		}
		else {
			float step = (1 / abs(dir[0])) * std::abs(bbox[0] - pt1[0]);
			Eigen::Vector3f intersect_pt = pt1 + step * dir;
			if (intersect_pt[1] >= bbox[2] && intersect_pt[1] <= bbox[3] && intersect_pt[2] >= bbox[4] && intersect_pt[2] <= bbox[5]) {
				status = true;
				return status;
			}
		}

		//xmax plane
		if ((pt1[0] <= bbox[1] && pt2[0] <= bbox[1]) || (pt1[0] >= bbox[1] && pt2[0] >= bbox[1])) {
			int a = 0; // not intersect
		}
		else {
			float step = (1 / abs(dir[0])) * std::abs(bbox[1] - pt1[0]);
			Eigen::Vector3f intersect_pt = pt1 + step * dir;
			if (intersect_pt[1] >= bbox[2] && intersect_pt[1] <= bbox[3] && intersect_pt[2] >= bbox[4] && intersect_pt[2] <= bbox[5]) {
				status = true;
				return status;
			}
		}

		//ymin plane
		if ((pt1[1] <= bbox[2] && pt2[1] <= bbox[2]) || (pt1[1] >= bbox[2] && pt2[1] >= bbox[2])) {
			int a = 0; // not intersect
		}
		else {
			float step = (1 / abs(dir[1])) * std::abs(bbox[2] - pt1[1]);
			Eigen::Vector3f intersect_pt = pt1 + step * dir;
			if (intersect_pt[0] >= bbox[0] && intersect_pt[0] <= bbox[1] && intersect_pt[2] >= bbox[4] && intersect_pt[2] <= bbox[5]) {
				status = true;
				return status;
			}
		}

		//ymax plane
		if ((pt1[1] <= bbox[3] && pt2[1] <= bbox[3]) || (pt1[1] >= bbox[3] && pt2[1] >= bbox[3])) {
			int a = 0; // not intersect
		}
		else {
			float step = (1 / abs(dir[1])) * std::abs(bbox[3] - pt1[1]);
			Eigen::Vector3f intersect_pt = pt1 + step * dir;
			if (intersect_pt[0] >= bbox[0] && intersect_pt[0] <= bbox[1] && intersect_pt[2] >= bbox[4] && intersect_pt[2] <= bbox[5]) {
				status = true;
				return status;
			}
		}

		//zmin plane
		if ((pt1[2] <= bbox[4] && pt2[2] <= bbox[4]) || (pt1[2] >= bbox[4] && pt2[2] >= bbox[4])) {
			int a = 0; // not intersect
		}
		else {
			float step = (1 / abs(dir[2])) * std::abs(bbox[4] - pt1[2]);
			Eigen::Vector3f intersect_pt = pt1 + step * dir;
			if (intersect_pt[0] >= bbox[0] && intersect_pt[0] <= bbox[1] && intersect_pt[1] >= bbox[2] && intersect_pt[1] <= bbox[3]) {
				status = true;
				return status;
			}
		}

		//zmax plane
		if ((pt1[2] <= bbox[5] && pt2[2] <= bbox[5]) || (pt1[2] >= bbox[5] && pt2[2] >= bbox[5])) {
			int a = 0; // not intersect
		}
		else {
			float step = (1 / abs(dir[2])) * std::abs(bbox[5] - pt1[2]);
			Eigen::Vector3f intersect_pt = pt1 + step * dir;
			if (intersect_pt[0] >= bbox[0] && intersect_pt[0] <= bbox[1] && intersect_pt[1] >= bbox[2] && intersect_pt[1] <= bbox[3]) {
				status = true;
				return status;
			}
		}

		return status;

	}

	/*
	pts: three points
	bbox: xmin,xmax,ymin,ymax,zmin,zmax
	return 0: inside, 1: intersect, 2: outside
	*/
	int triangle_intersect_bbox(std::vector<Eigen::Vector3f> pts, std::vector<float> bbox, std::vector<bool>& pts_status) {
		int status = 0;
		Eigen::Vector3f pt1 = pts[0];
		Eigen::Vector3f pt2 = pts[1];
		Eigen::Vector3f pt3 = pts[2];

		bool pt1_i = false, pt2_i = false, pt3_i = false;
		bool pt1_o = false, pt2_o = false, pt3_o = false;
		if ((pt1[0] >= bbox[0]) && (pt1[0] <= bbox[1]) && (pt1[1] >= bbox[2]) && (pt1[1] <= bbox[3]) && (pt1[2] >= bbox[4]) && (pt1[2] <= bbox[5]) ) 
		{
			pt1_i=true;
		}
		if ((pt2[0] >= bbox[0]) && (pt2[0] <= bbox[1]) && (pt2[1] >= bbox[2]) && (pt2[1] <= bbox[3]) && (pt2[2] >= bbox[4]) && (pt2[2] <= bbox[5]))
		{
			pt2_i = true;
		}
		if ((pt3[0] >= bbox[0]) && (pt3[0] <= bbox[1]) && (pt3[1] >= bbox[2]) && (pt3[1] <= bbox[3]) && (pt3[2] >= bbox[4]) && (pt3[2] <= bbox[5]))
		{
			pt3_i = true;
		}
		if (((pt1[0] <= bbox[0]) || (pt1[0] >= bbox[1])) || ((pt1[1] <= bbox[2]) || (pt1[1] >= bbox[3])) || ((pt1[2] <= bbox[4]) || (pt1[2] >= bbox[5])))
		{
			pt1_o = true;
		}
		if (((pt2[0] <= bbox[0]) || (pt2[0] >= bbox[1])) || ((pt2[1] <= bbox[2]) || (pt2[1] >= bbox[3])) || ((pt2[2] <= bbox[4]) || (pt2[2] >= bbox[5])))
		{
			pt2_o = true;
		}
		if (((pt3[0] <= bbox[0]) || (pt3[0] >= bbox[1])) || ((pt3[1] <= bbox[2]) || (pt3[1] >= bbox[3])) || ((pt3[2] <= bbox[4]) || (pt3[2] >= bbox[5])))
		{
			pt3_o = true;
		}

		// First, determine the inside triangle
		if (pt1_i && pt2_i && pt3_i) {
			status = 0;
		}
		// Second, determine the outside triangle: condition 1: three vertex outside the bbox, condition2: three edge not intersect with bbox  
		else if (pt1_o && pt2_o && pt3_o) {
			bool e1 = edge_intersect_bbox({ pt1,pt2 }, bbox);
			bool e2 = edge_intersect_bbox({ pt2,pt3 }, bbox);
			bool e3 = edge_intersect_bbox({ pt1,pt3 }, bbox);
			// condition 2
			if (e1 || e2 || e3) {
				status = 1;
			}
			else {
				status = 2;
			}
		}
		else {
			status = 1;
		}

		pts_status.push_back(pt1_i);
		pts_status.push_back(pt2_i);
		pts_status.push_back(pt3_i);
		return status;

	}

	/*
pts: three points
bbox: xmin,xmax,ymin,ymax,zmin,zmax
return 0: inside, 1: intersect, 2: outside
*/
	int triangle_intersect_bbox_outside_origin(std::vector<Eigen::Vector3f> pts, std::vector<float> bbox, std::vector<bool>& pts_status) {
		int status = 0;
		Eigen::Vector3f pt1 = pts[0];
		Eigen::Vector3f pt2 = pts[1];
		Eigen::Vector3f pt3 = pts[2];

		bool pt1_i = false, pt2_i = false, pt3_i = false;
		bool pt1_o = false, pt2_o = false, pt3_o = false;
		if ((pt1[0] > bbox[0]) && (pt1[0] < bbox[1]) && (pt1[1] > bbox[2]) && (pt1[1] < bbox[3]) && (pt1[2] > bbox[4]) && (pt1[2] < bbox[5]))
		{
			pt1_i = true;
		}
		if ((pt2[0] > bbox[0]) && (pt2[0] < bbox[1]) && (pt2[1] > bbox[2]) && (pt2[1] < bbox[3]) && (pt2[2] > bbox[4]) && (pt2[2] < bbox[5]))
		{
			pt2_i = true;
		}
		if ((pt3[0] > bbox[0]) && (pt3[0] < bbox[1]) && (pt3[1] > bbox[2]) && (pt3[1] < bbox[3]) && (pt3[2] > bbox[4]) && (pt3[2] < bbox[5]))
		{
			pt3_i = true;
		}
		

		// First, determine the inside triangle
		if (pt1_i && pt2_i && pt3_i) {
			status = 0;
		}
		// Second, determine the outside triangle: condition 1: three vertex outside the bbox, condition2: three edge not intersect with bbox  
		else if (!pt1_i && !pt2_i && !pt3_i) {
			status = 2;
		}
		else {
			status = 1;
		}

		pts_status.push_back(!pt1_i);
		pts_status.push_back(!pt2_i);
		pts_status.push_back(!pt3_i);
		return status;

	}

	void step1_triangle_intersect_bbox_outside(FaceInter& faceinter, std::vector<float> bbox) {
		int status = 0;
		Eigen::Vector3f pt1 = faceinter.v1;
		Eigen::Vector3f pt2 = faceinter.v2;
		Eigen::Vector3f pt3 = faceinter.v3;

		bool pt1_i = false, pt2_i = false, pt3_i = false;
		bool pt1_o = false, pt2_o = false, pt3_o = false;
		if ((pt1[0] > bbox[0]) && (pt1[0] < bbox[1]) && (pt1[1] > bbox[2]) && (pt1[1] < bbox[3]) && (pt1[2] > bbox[4]) && (pt1[2] < bbox[5]))
		{
			pt1_i = true;
		}
		if ((pt2[0] > bbox[0]) && (pt2[0] < bbox[1]) && (pt2[1] > bbox[2]) && (pt2[1] < bbox[3]) && (pt2[2] > bbox[4]) && (pt2[2] < bbox[5]))
		{
			pt2_i = true;
		}
		if ((pt3[0] > bbox[0]) && (pt3[0] < bbox[1]) && (pt3[1] > bbox[2]) && (pt3[1] < bbox[3]) && (pt3[2] > bbox[4]) && (pt3[2] < bbox[5]))
		{
			pt3_i = true;
		}
		faceinter.v1_inside = pt1_i;
		faceinter.v2_inside = pt2_i;
		faceinter.v3_inside = pt3_i;

		// First, 3 vertex inside
		if (pt1_i && pt2_i && pt3_i) {
			faceinter.type = 0;
		}
		// Second, 3 vertex outside, maybe type=5,6,8
		else if (!pt1_i && !pt2_i && !pt3_i) {
			edge_intersect_bbox1(pt1, pt2, bbox, faceinter.e12_pts);
			edge_intersect_bbox1(pt2, pt3, bbox, faceinter.e23_pts);
			edge_intersect_bbox1(pt3, pt1, bbox, faceinter.e31_pts);
			int e12 = (faceinter.e12_pts.size() != 0) ? 1 : 0;
			int e23 = (faceinter.e23_pts.size() != 0) ? 1 : 0;
			int e31 = (faceinter.e31_pts.size() != 0) ? 1 : 0;
			if ((e12 + e23 + e31) == 0) {
				faceinter.type = 8;
			}
			else if((e12 + e23 + e31) == 1) {
				faceinter.type = 5;
				float ix= bbox[0], iy= bbox[3], iz;
				//check left-top
				if (isPointInsideTriangle(bbox[0], bbox[3], pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])) {ix = bbox[0], iy = bbox[3];}
				//check left-bot
				if (isPointInsideTriangle(bbox[0], bbox[2], pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])) { ix = bbox[0], iy = bbox[2]; }
				//check right-top
				if (isPointInsideTriangle(bbox[1], bbox[3], pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])) { ix = bbox[1], iy = bbox[3]; }
				//check right-bot
				if (isPointInsideTriangle(bbox[1], bbox[2], pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])) { ix = bbox[1], iy = bbox[2]; }

				iz=calculateZ(ix, iy, pt1[0], pt1[1], pt1[2], pt2[0], pt2[1], pt2[2], pt3[0], pt3[1], pt3[2]);
				Eigen::Vector3f inter_pt(ix, iy, iz);
				faceinter.inside_pts.push_back(inter_pt);
			}
			else {
				faceinter.type = 6;
			}
		}
		// third, mixed vertex, maybe type=1,2,3,4,7
		else {
			edge_intersect_bbox1(pt1, pt2, bbox, faceinter.e12_pts);
			edge_intersect_bbox1(pt2, pt3, bbox, faceinter.e23_pts);
			edge_intersect_bbox1(pt3, pt1, bbox, faceinter.e31_pts);

			bool inside = false;
			float ix = 0, iy = 0, iz;
			//check left-top
			if (isPointInsideTriangle(bbox[0], bbox[3], pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])) { ix = bbox[0], iy = bbox[3], inside=true; }
			//check left-bot
			if (isPointInsideTriangle(bbox[0], bbox[2], pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])) { ix = bbox[0], iy = bbox[2], inside = true; }
			//check right-top
			if (isPointInsideTriangle(bbox[1], bbox[3], pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])) { ix = bbox[1], iy = bbox[3], inside = true; }
			//check right-bot
			if (isPointInsideTriangle(bbox[1], bbox[2], pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])) { ix = bbox[1], iy = bbox[2], inside = true; }

			if (inside) {
				iz = calculateZ(ix, iy, pt1[0], pt1[1], pt1[2], pt2[0], pt2[1], pt2[2], pt3[0], pt3[1], pt3[2]);
				Eigen::Vector3f inter_pt(ix, iy, iz);
				faceinter.inside_pts.push_back(inter_pt);
			}
			int v1i = (pt1_i) ? 1 : 0;
			int v2i = (pt2_i) ? 1 : 0;
			int v3i = (pt3_i) ? 1 : 0;
			// type=1,3,7
			if ((v1i + v2i + v3i) == 1) {
				if (faceinter.e12_pts.size() && faceinter.e23_pts.size() && faceinter.e31_pts.size()) {
					faceinter.type = 7;
				}
				else {
					faceinter.type = (inside) ? 3 : 1;
				}
			}
			//type=2,4
			else {
				faceinter.type = (inside) ? 4 : 2;
			}
		}
	}

	void step2_triangulation(FaceInter& faceinter, std::vector<std::vector<Eigen::Vector3f>>& triangulation, std::vector<std::vector<Eigen::Vector2f>>& triangulation_vts) {
		if (faceinter.type == 1) {
			if (faceinter.v1_inside) {
				triangulation.push_back({ faceinter.e12_pts[0],faceinter.v2,faceinter.v3 });
				triangulation.push_back({ faceinter.e12_pts[0],faceinter.v3,faceinter.e31_pts[0]});
				Eigen::Vector2f e12 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e12,faceinter.vt2,faceinter.vt3 });
				triangulation_vts.push_back({ e12,faceinter.vt3,e31 });
			}
			else if (faceinter.v2_inside) {
				triangulation.push_back({ faceinter.e23_pts[0],faceinter.v3,faceinter.v1});
				triangulation.push_back({ faceinter.e23_pts[0],faceinter.v1,faceinter.e12_pts[0] });
				Eigen::Vector2f e23 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e12 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e23,faceinter.vt3,faceinter.vt1 });
				triangulation_vts.push_back({ e23,faceinter.vt1,e12 });
			}
			else if (faceinter.v3_inside){
				triangulation.push_back({ faceinter.e31_pts[0],faceinter.v1,faceinter.v2 });
				triangulation.push_back({ faceinter.e31_pts[0],faceinter.v2,faceinter.e23_pts[0] });
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e23 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e31,faceinter.vt1,faceinter.vt2 });
				triangulation_vts.push_back({ e31,faceinter.vt2,e23 });
			}
		}
		else if (faceinter.type == 2) {
			if (faceinter.v1_inside && faceinter.v2_inside) {
				triangulation.push_back({ faceinter.e23_pts[0],faceinter.v3,faceinter.e31_pts[0]});
				Eigen::Vector2f e23 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e23,faceinter.vt3,e31 });
			}
			else if (faceinter.v1_inside && faceinter.v3_inside) {
				triangulation.push_back({ faceinter.e12_pts[0],faceinter.v2,faceinter.e23_pts[0] });
				Eigen::Vector2f e12 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e23 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e12,faceinter.vt2,e23 });
			}
			else if (faceinter.v2_inside && faceinter.v3_inside) {
				triangulation.push_back({ faceinter.e12_pts[0],faceinter.e31_pts[0],faceinter.v1});
				Eigen::Vector2f e12 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e12,e31,faceinter.vt1 });
			}
		}
		else if (faceinter.type == 3) {
			if (faceinter.v1_inside) {
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.e12_pts[0],faceinter.v2 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v2,faceinter.v3 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v3,faceinter.e31_pts[0]});
				Eigen::Vector2f inside = calculateUV(faceinter.inside_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e12 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ inside,e12,faceinter.vt2 });
				triangulation_vts.push_back({ inside,faceinter.vt2, faceinter.vt3 });
				triangulation_vts.push_back({ inside,faceinter.vt3,e31 });
			}
			else if (faceinter.v2_inside) {
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.e23_pts[0],faceinter.v3 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v3,faceinter.v1 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v1,faceinter.e12_pts[0] });
				Eigen::Vector2f inside = calculateUV(faceinter.inside_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e23 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e12 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ inside,e23,faceinter.vt3 });
				triangulation_vts.push_back({ inside,faceinter.vt3, faceinter.vt1 });
				triangulation_vts.push_back({ inside,faceinter.vt1,e12 });
			}
			else if (faceinter.v3_inside) {
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.e31_pts[0],faceinter.v1 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v1,faceinter.v2 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v2,faceinter.e23_pts[0] });
				Eigen::Vector2f inside = calculateUV(faceinter.inside_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e23 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ inside,e31,faceinter.vt1 });
				triangulation_vts.push_back({ inside,faceinter.vt1, faceinter.vt2 });
				triangulation_vts.push_back({ inside,faceinter.vt2,e23 });
			}
		}
		else if (faceinter.type == 4) {
			if (faceinter.v1_inside && faceinter.v2_inside) {
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.e23_pts[0],faceinter.v3 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v3,faceinter.e31_pts[0]});
				Eigen::Vector2f inside = calculateUV(faceinter.inside_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e23 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ inside,e23,faceinter.vt3 });
				triangulation_vts.push_back({ inside,faceinter.vt3, e31 });
			}
			else if (faceinter.v2_inside && faceinter.v3_inside) {
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.e31_pts[0],faceinter.v1 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v1,faceinter.e12_pts[0] });
				Eigen::Vector2f inside = calculateUV(faceinter.inside_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e12 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ inside,e31,faceinter.vt1 });
				triangulation_vts.push_back({ inside,faceinter.vt1, e12 });
			}
		}
		else if (faceinter.type == 5) {
			if (faceinter.e12_pts.size()==2) {
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v1,faceinter.e12_pts[0] });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v3,faceinter.v1 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v2,faceinter.v3 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.e12_pts[1],faceinter.v2});
				Eigen::Vector2f inside = calculateUV(faceinter.inside_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e121 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e122 = calculateUV(faceinter.e12_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ inside,faceinter.vt1, e121 });
				triangulation_vts.push_back({ inside,faceinter.vt3, faceinter.vt1 });
				triangulation_vts.push_back({ inside,faceinter.vt2, faceinter.vt3 });
				triangulation_vts.push_back({ inside,e122, faceinter.vt2 });
			}
			else if (faceinter.e23_pts.size()==2) {
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v2,faceinter.e23_pts[0] });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v1,faceinter.v2 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v3,faceinter.v1 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.e23_pts[1],faceinter.v3 });
				Eigen::Vector2f inside = calculateUV(faceinter.inside_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e231 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e232 = calculateUV(faceinter.e23_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ inside,faceinter.vt2, e231 });
				triangulation_vts.push_back({ inside,faceinter.vt1, faceinter.vt2 });
				triangulation_vts.push_back({ inside,faceinter.vt3, faceinter.vt1 });
				triangulation_vts.push_back({ inside,e232, faceinter.vt3 });
			}
			else if (faceinter.e31_pts.size()==2) {
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v3,faceinter.e31_pts[0] });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v2,faceinter.v3 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.v1,faceinter.v2 });
				triangulation.push_back({ faceinter.inside_pts[0],faceinter.e31_pts[1],faceinter.v1 });
				Eigen::Vector2f inside = calculateUV(faceinter.inside_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e311 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e312 = calculateUV(faceinter.e31_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ inside,faceinter.vt3, e311 });
				triangulation_vts.push_back({ inside,faceinter.vt2, faceinter.vt3 });
				triangulation_vts.push_back({ inside,faceinter.vt1, faceinter.vt2 });
				triangulation_vts.push_back({ inside,e312, faceinter.vt1 });
			}
		}
		else if (faceinter.type == 6) {
			if (faceinter.e12_pts.size() == 0) {
				triangulation.push_back({ faceinter.e23_pts[1],faceinter.v3,faceinter.e31_pts[0] });
				triangulation.push_back({ faceinter.v1,faceinter.v2,faceinter.e23_pts[0] });
				triangulation.push_back({ faceinter.v1,faceinter.e23_pts[0],faceinter.e31_pts[1]});
				Eigen::Vector2f e231 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e232 = calculateUV(faceinter.e23_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e311 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e312 = calculateUV(faceinter.e31_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e232,faceinter.vt3, e311 });
				triangulation_vts.push_back({ faceinter.vt1, faceinter.vt2,e231 });
				triangulation_vts.push_back({ faceinter.vt1,e231,e312});
			}
			else if (faceinter.e23_pts.size() == 0) {
				triangulation.push_back({ faceinter.e31_pts[1],faceinter.v1,faceinter.e12_pts[0] });
				triangulation.push_back({ faceinter.v2,faceinter.v3,faceinter.e31_pts[0] });
				triangulation.push_back({ faceinter.v2,faceinter.e31_pts[0],faceinter.e12_pts[1] });
				Eigen::Vector2f e121 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e122 = calculateUV(faceinter.e12_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e311 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e312 = calculateUV(faceinter.e31_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e312,faceinter.vt1, e121 });
				triangulation_vts.push_back({ faceinter.vt2, faceinter.vt3,e311 });
				triangulation_vts.push_back({ faceinter.vt2,e311,e122 });
			}
			else if (faceinter.e31_pts.size() == 0) {
				triangulation.push_back({ faceinter.e12_pts[1],faceinter.v2,faceinter.e23_pts[0] });
				triangulation.push_back({ faceinter.v3,faceinter.v1,faceinter.e12_pts[0] });
				triangulation.push_back({ faceinter.v3,faceinter.e12_pts[0],faceinter.e23_pts[1] });
				Eigen::Vector2f e121 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e122 = calculateUV(faceinter.e12_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e231 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e232 = calculateUV(faceinter.e23_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e122,faceinter.vt2, e231 });
				triangulation_vts.push_back({ faceinter.vt3, faceinter.vt1,e121 });
				triangulation_vts.push_back({ faceinter.vt3,e121,e232 });
			}
		}
		else if (faceinter.type == 7) {
			if (faceinter.v1_inside) {
				triangulation.push_back({ faceinter.e12_pts[0],faceinter.v2,faceinter.e23_pts[0] });
				triangulation.push_back({ faceinter.e23_pts[1],faceinter.v3,faceinter.e31_pts[0] });
				Eigen::Vector2f e12 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e231 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e232 = calculateUV(faceinter.e23_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ e12,faceinter.vt2, e231 });
				triangulation_vts.push_back({ e232, faceinter.vt3,e31 });
			}
			else if (faceinter.v2_inside) {
				triangulation.push_back({ faceinter.v1,faceinter.e12_pts[0],faceinter.e31_pts[1]});
				triangulation.push_back({ faceinter.v3,faceinter.e31_pts[0],faceinter.e23_pts[0]});
				Eigen::Vector2f e12 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e311 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e312 = calculateUV(faceinter.e31_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e23 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({faceinter.vt1, e12, e312 });
				triangulation_vts.push_back({ faceinter.vt3, e311,e23 });
			}
			else if (faceinter.v3_inside) {
				triangulation.push_back({ faceinter.v1,faceinter.e12_pts[0],faceinter.e31_pts[0] });
				triangulation.push_back({ faceinter.v2,faceinter.e23_pts[0],faceinter.e12_pts[1] });
				Eigen::Vector2f e121 = calculateUV(faceinter.e12_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e122 = calculateUV(faceinter.e12_pts[1], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e23 = calculateUV(faceinter.e23_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				Eigen::Vector2f e31 = calculateUV(faceinter.e31_pts[0], faceinter.v1, faceinter.vt1, faceinter.v2, faceinter.vt2, faceinter.v3, faceinter.vt3);
				triangulation_vts.push_back({ faceinter.vt1, e121, e31 });
				triangulation_vts.push_back({ faceinter.vt2, e23,e122 });
			}
		}
	}

	/*
	update the vertex texture coordinates based on 
	*/
	int update_vt(std::vector<Eigen::Vector3f> pts, std::vector<Eigen::Vector2f> vts, std::vector<Eigen::Vector3f> crop_pts, std::vector<Eigen::Vector2f>& crop_vts) {
		crop_vts.clear();
		for (Eigen::Vector3f pt : crop_pts) {
			bool same = false;
			int index = 0;
			for (int i = 0; i < pts.size();++i) {
				Eigen::Vector3f pt_tmp = pts[i];
				float dis = (pt - pt_tmp).norm();
				if (dis < 1e-3) {
					same = true;
					index = i;
				}
			}
			if (same) {
				Eigen::Vector2f vt = vts[index];
				crop_vts.push_back(vt);
			}
			else {
				//use barycenteric coordinate the calculate the inside point
				float areai01 = area_triangle(pt, pts[0], pts[1]);
				float areai02 = area_triangle(pt, pts[0], pts[2]);
				float areai12 = area_triangle(pt, pts[1], pts[2]);
				float area= area_triangle(pts[0], pts[1], pts[2]);
				Eigen::Vector2f vt = vts[0] * (areai12 / area) +vts[1] * (areai02 / area) + vts[2] * (areai01 / area);
				crop_vts.push_back(vt);
			}
		}
		return 0;
	}

	int SH_TRIM_EACH_PLANE(bool min_or_max, int axis, float axis_value, std::vector<Eigen::Vector3f> fv_pts, std::vector<Eigen::Vector3f>& cropped_fv_pts) {
		std::vector<Eigen::Vector3f> out_pts;
		cropped_fv_pts.clear();
		Eigen::Vector3f pt1;
		Eigen::Vector3f pt2;
		float min_value;
		float max_value;

		if (fv_pts.size() == 0) { return 0; }

		for (int i = 0; i < fv_pts.size(); ++i) {
			if (i == fv_pts.size() - 1) {
				pt1 = fv_pts[i];
				pt2 = fv_pts[0];
			}
			else {
				pt1 = fv_pts[i];
				pt2 = fv_pts[i + 1];
			}

			min_value = std::min(pt1[axis], pt2[axis]);
			max_value = std::max(pt1[axis], pt2[axis]);
			if (min_or_max) {
				if (min_value >= axis_value) { out_pts.push_back(pt1); out_pts.push_back(pt2); }
				else if (min_value< axis_value && max_value>axis_value) {
					float len = (pt2 - pt1).norm();
					Eigen::Vector3f dir = (pt2 - pt1) / len;
					float step = abs(1 / dir[axis]) * abs(axis_value - pt1[axis]);
					Eigen::Vector3f intersect_pt = pt1 + step * dir;
					if (pt1[axis] > pt2[axis]) { out_pts.push_back(pt1); out_pts.push_back(intersect_pt); }
					else { out_pts.push_back(intersect_pt); out_pts.push_back(pt2); }
				}
				else if (min_value < axis_value && max_value == axis_value) {
					if (pt1[axis] > pt2[axis]) { out_pts.push_back(pt1); }
					else { out_pts.push_back(pt2); }
				}
			}
			else {
				if (max_value <= axis_value) { out_pts.push_back(pt1); out_pts.push_back(pt2); }
				else if (max_value > axis_value && min_value < axis_value) {
					float len = (pt2 - pt1).norm();
					Eigen::Vector3f dir = (pt2 - pt1) / len;
					float step = abs(1 / dir[axis]) * abs(axis_value - pt1[axis]);
					Eigen::Vector3f intersect_pt = pt1 + step * dir;
					if (pt1[axis] < pt2[axis]) { out_pts.push_back(pt1); out_pts.push_back(intersect_pt); }
					else { out_pts.push_back(intersect_pt); out_pts.push_back(pt2); }
				}
				else if (max_value > axis_value && min_value == axis_value) {
					if (pt1[axis] < pt2[axis]) { out_pts.push_back(pt1); }
					else { out_pts.push_back(pt2); }
				}
			}
		}


		//remove ducplicatre points
		if (out_pts.size() == 0) { return 0; }
		for (Eigen::Vector3f pt : out_pts) {
			if (cropped_fv_pts.size() == 0) {
				cropped_fv_pts.push_back(pt);
			}
			else {
				Eigen::Vector3f pt_tmp = cropped_fv_pts[cropped_fv_pts.size() - 1];
				float dis = (pt_tmp - pt).norm();
				if (dis > 1e-3) {
					cropped_fv_pts.push_back(pt);
				}
			}
		}
		Eigen::Vector3f pt_end = cropped_fv_pts[cropped_fv_pts.size() - 1];
		Eigen::Vector3f pt_start = cropped_fv_pts[0];
		float dis = (pt_end - pt_start).norm();
		if (dis < 1e-3) {
			cropped_fv_pts.pop_back();
		}
		return 0;

	}

	/*
	reference: Sutherland–Hodgman algorithm https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
	fv_pts: three points
	bbox: xmin,xmax,ymin,ymax,zmin,zmax
	cropped_fv_pts: trimmed pts
	*/
	int SH_TRIM(std::vector<Eigen::Vector3f> fv_pts, std::vector<float> bbox, std::vector<Eigen::Vector3f>& cropped_fv_pts) {
		cropped_fv_pts.clear();

		std::vector<Eigen::Vector3f> input_pts;
		std::vector<Eigen::Vector3f> output_pts;

		// xmin plane 
		input_pts = fv_pts;
		SH_TRIM_EACH_PLANE(true, 0, bbox[0], input_pts, output_pts);

		//xmax plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(false, 0, bbox[1], input_pts, output_pts);

		//ymin plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(true, 1, bbox[2], input_pts, output_pts);

		//ymax plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(false, 1, bbox[3], input_pts, output_pts);

		//zmin plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(true, 2, bbox[4], input_pts, output_pts);

		//zmax plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(false, 2, bbox[5], input_pts, output_pts);

		cropped_fv_pts = output_pts;
		return 0;
	}

	int SH_TRIM_outside(std::vector<Eigen::Vector3f> fv_pts, std::vector<float> bbox, std::vector<Eigen::Vector3f>& cropped_fv_pts) {
		cropped_fv_pts.clear();

		std::vector<Eigen::Vector3f> input_pts;
		std::vector<Eigen::Vector3f> output_pts;

		// xmin plane 
		input_pts = fv_pts;
		SH_TRIM_EACH_PLANE(true, 0, bbox[0], input_pts, output_pts);

		//xmax plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(false, 0, bbox[1], input_pts, output_pts);

		//ymin plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(true, 1, bbox[2], input_pts, output_pts);

		//ymax plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(false, 1, bbox[3], input_pts, output_pts);

		//zmin plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(true, 2, bbox[4], input_pts, output_pts);

		//zmax plane
		input_pts = output_pts;
		SH_TRIM_EACH_PLANE(false, 2, bbox[5], input_pts, output_pts);

		// calcuate the outside part based on fv_pts, output_pts
		bool v1_cover=false, v2_cover = false, v3_cover = false, e12_cover = false, e23_cover = false, e31_cover = false;
		bool inside_fv = false;
		Vector3f e12_cover_pt, e23_cover_pt, e31_cover_pt, inside_pt;
		int inside_pt_id = 0;
		for (int i = 0; i < output_pts.size(); ++i) {
			if((output_pts[i] - fv_pts[0]).norm() < 1e-5) {
				v1_cover = true;
			}
		}
		for (int i = 0; i < output_pts.size(); ++i) {
			if ((output_pts[i] - fv_pts[1]).norm() < 1e-5) {
				v2_cover = true;
			}
		}
		for (int i = 0; i < output_pts.size(); ++i) {
			if ((output_pts[i] - fv_pts[2]).norm() < 1e-5) {
				v3_cover = true;
			}
		}
		for (int i = 0; i < output_pts.size(); ++i) {
			Vector3f p1, p2, p_mid;
			p1 = fv_pts[0];
			p2 = fv_pts[1];
			p_mid = output_pts[i];
			float e1m, e2m, e12;
			e1m = (p1 - p_mid).norm();
			e2m=(p2 - p_mid).norm();
			e12 = (p1 - p2).norm();
			if ((e1m+e2m-e12)<1e-5 && e1m>1e-5 && e2m>1e-5) {
				e12_cover = true;
				e12_cover_pt = output_pts[i];
			}
		}

		for (int i = 0; i < output_pts.size(); ++i) {
			Vector3f p1, p2, p_mid;
			p1 = fv_pts[1];
			p2 = fv_pts[2];
			p_mid = output_pts[i];
			float e1m, e2m, e12;
			e1m = (p1 - p_mid).norm();
			e2m = (p2 - p_mid).norm();
			e12 = (p1 - p2).norm();
			if ((e1m + e2m - e12) < 1e-5 && e1m > 1e-5 && e2m > 1e-5) {
				e23_cover = true;
				e23_cover_pt = output_pts[i];
			}
		}
		for (int i = 0; i < output_pts.size(); ++i) {
			Vector3f p1, p2, p_mid;
			p1 = fv_pts[2];
			p2 = fv_pts[0];
			p_mid = output_pts[i];
			float e1m, e2m, e12;
			e1m = (p1 - p_mid).norm();
			e2m = (p2 - p_mid).norm();
			e12 = (p1 - p2).norm();
			if ((e1m + e2m - e12) < 1e-5 && e1m > 1e-5 && e2m > 1e-5) {
				e31_cover = true;
				e31_cover_pt = output_pts[i];
			}
		}
		for (int i = 0; i < output_pts.size(); ++i) {
			float a12m, a23m, a31m, a123;
			a12m = area_triangle(fv_pts[0], fv_pts[1], output_pts[i]);
			a23m = area_triangle(fv_pts[1], fv_pts[2], output_pts[i]);
			a31m = area_triangle(fv_pts[0], fv_pts[2], output_pts[i]);
			a123 = area_triangle(fv_pts[0], fv_pts[1], fv_pts[2]);

			if ((a12m+a23m+a31m-a123)<1e-5 && a12m>1e-5 && a23m>1e-5 && a31m>1e-5) {
				inside_fv = true;
				inside_pt = output_pts[i];
				inside_pt_id = i;
			}
		}

		std::vector<bool> inout;
		std::vector<Vector3f> pts;
		if (!v1_cover) {
			Vector3f pt = fv_pts[0];
			if ((pt[0] >= bbox[0]) && (pt[0] <= bbox[1]) && (pt[1] >= bbox[2]) && (pt[1] <= bbox[3]) && (pt[2] >= bbox[4]) && (pt[2] <= bbox[5]))
			{
				inout.push_back(true);
			}
			else {
				inout.push_back(false);
			}
			pts.push_back(fv_pts[0]);
		}
		if (e12_cover) {
			Vector3f pt = e12_cover_pt;
			if ((pt[0] >= bbox[0]) && (pt[0] <= bbox[1]) && (pt[1] >= bbox[2]) && (pt[1] <= bbox[3]) && (pt[2] >= bbox[4]) && (pt[2] <= bbox[5]))
			{
				inout.push_back(true);
			}
			else {
				inout.push_back(false);
			}
			pts.push_back(e12_cover_pt);
		}
		if (!v2_cover) {
			Vector3f pt = fv_pts[1];
			if ((pt[0] >= bbox[0]) && (pt[0] <= bbox[1]) && (pt[1] >= bbox[2]) && (pt[1] <= bbox[3]) && (pt[2] >= bbox[4]) && (pt[2] <= bbox[5]))
			{
				inout.push_back(true);
			}
			else {
				inout.push_back(false);
			}
			pts.push_back(fv_pts[1]);
		}
		if (e23_cover) {
			Vector3f pt = e23_cover_pt;
			if ((pt[0] >= bbox[0]) && (pt[0] <= bbox[1]) && (pt[1] >= bbox[2]) && (pt[1] <= bbox[3]) && (pt[2] >= bbox[4]) && (pt[2] <= bbox[5]))
			{
				inout.push_back(true);
			}
			else {
				inout.push_back(false);
			}
			pts.push_back(e23_cover_pt);
		}
		if (!v3_cover) {
			Vector3f pt = fv_pts[2];
			if ((pt[0] >= bbox[0]) && (pt[0] <= bbox[1]) && (pt[1] >= bbox[2]) && (pt[1] <= bbox[3]) && (pt[2] >= bbox[4]) && (pt[2] <= bbox[5]))
			{
				inout.push_back(true);
			}
			else {
				inout.push_back(false);
			}
			pts.push_back(fv_pts[2]);
		}
		if (e31_cover) {
			Vector3f pt = e31_cover_pt;
			if ((pt[0] >= bbox[0]) && (pt[0] <= bbox[1]) && (pt[1] >= bbox[2]) && (pt[1] <= bbox[3]) && (pt[2] >= bbox[4]) && (pt[2] <= bbox[5]))
			{
				inout.push_back(true);
			}
			else {
				inout.push_back(false);
			}
			pts.push_back(e31_cover_pt);
		}

		int num_out = 0;
		for (int i = 0; i < inout.size(); ++i) {
			if (inout[i] == false) {
				num_out++;
			}
		}
		if (num_out == 1) {
			int id = 0;
			for(id=0;id<inout.size();++id){
				if (inout[id] == false) {
					break;
				}
			}
			int inside_fv_id = 0;
			if (inside_fv) {
				for (inside_fv_id = 0; inside_fv_id < inout.size(); ++inside_fv_id) {
					int id_next = inside_fv_id + 1;
					if (id_next == inout.size()) {
						id_next = 0;
					}
					if (inout[inside_fv_id] == true && inout[id_next]==true) {
						break;
					}
				}
			}
			while (cropped_fv_pts.size() != pts.size()) {
				if (id == pts.size()) {
					id = 0;
				}
				cropped_fv_pts.push_back(pts[id]);
				if (inside_fv && id == inside_fv_id) {
					cropped_fv_pts.push_back(inside_pt);
				}
				id++;

			}
		}
		else if(num_out==2){
			int id = 0;
			for (id = 0; id < inout.size(); ++id) {
				int id_next = id + 1;
				if (id_next == inout.size()) {
					id_next = 0;
				}
				if (inout[id] == false && inout[id_next]==false) {
					break;
				}
			}
			int inside_fv_id = 0;
			if (inside_fv) {
				for (inside_fv_id = 0; inside_fv_id < inout.size(); ++inside_fv_id) {
					int id_next = inside_fv_id + 1;
					if (id_next == inout.size()) {
						id_next = 0;
					}
					if (inout[inside_fv_id] == true && inout[id_next] == true) {
						break;
					}
				}
			}
			int target_num = pts.size();
			if (inside_fv) {
				target_num++;
			}
			while (cropped_fv_pts.size() != target_num) {
				if (id == pts.size()) {
					id = 0;
				}
				cropped_fv_pts.push_back(pts[id]);
				if (inside_fv && id == inside_fv_id) {
					cropped_fv_pts.push_back(inside_pt);
				}
				id++;

			}

		}


		return 0;
	}

    class Trimming {

	public:
		std::string input_mesh_path_;
		EigenTriMesh mesh_;
		std::vector<std::vector<float>> bbox_list_; // list of cropping volume, each has xmin,xmax,ymin,ymax,zmin,zmax
		std::vector<std::string> output_list_;
		bool has_material_;
		int keep_aoi_; // 0: means keep outside part, 1: means keep inside part
		OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_;

		Trimming(std::string input_path, std::vector<std::vector<float>> bbox_list_raw, std::vector<std::string> output_list, int keep_aoi) {
			input_mesh_path_ = input_path;
			bbox_list_ = bbox_list_raw;
			output_list_ = output_list;
			keep_aoi_ = keep_aoi;

			mesh_.request_halfedge_texcoords2D();
			mesh_.request_face_texture_index();

			OpenMesh::IO::Options opt(OpenMesh::IO::Options::FaceTexCoord);
			opt += OpenMesh::IO::Options::FaceColor;
			if (!OpenMesh::IO::read_mesh(mesh_, input_path, opt))
			{
				std::cerr << "[ERROR]Cannot read mesh from " << input_path << std::endl;
			}

			if (!mesh_.get_property_handle(texindex_, "TextureMapping")){
				has_material_ = false;
				std::cout << "input mesh doesn't have material" << std::endl;
			}
			else {
				has_material_ = true;
			}		
		}

		void Start_multi_inside() {

			for (int crop_id = 0; crop_id < bbox_list_.size(); ++crop_id) {
				OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out;
				std::vector<float> bbox = bbox_list_[crop_id];
				std::string output_path = output_list_[crop_id];

				EigenTriMesh mesh_out;
				mesh_out = mesh_;
				mesh_out.request_face_status();
				mesh_out.request_edge_status();
				mesh_out.request_halfedge_status();
				mesh_out.request_vertex_status();
				mesh_out.request_halfedge_texcoords2D();
				mesh_out.request_face_texture_index();
				if (!mesh_out.get_property_handle(texindex_out, "TextureMapping")) {
					has_material_ = false;
					std::cout << "input mesh doesn't have material" << std::endl;
				}
				else {
					has_material_ = true;
				}


				// iterate all faces, delete the outside&intersect faces
				EigenTriMesh::FaceIter f_it, f_end(mesh_.faces_end());
				std::vector<int> delete_fids;
				std::vector<int> intersect_fids;
				std::set<int> kdtree_vids;
				for (f_it = mesh_.faces_begin(); f_it != f_end; ++f_it) {
					std::vector<Eigen::Vector3f> fv_pts;
					std::vector<int> fv_ids;
					std::vector<bool> pts_status;
					EigenTriMesh::HalfedgeHandle heh;
					
					for (auto fheh_it = mesh_.fh_begin(*f_it); fheh_it.is_valid(); ++fheh_it) {
						EigenTriMesh::VertexHandle fvh = mesh_.to_vertex_handle(*fheh_it);
						fv_pts.push_back(mesh_.point(fvh));
						fv_ids.push_back(fvh.idx());
					}
					int intersect_status = triangle_intersect_bbox(fv_pts, bbox, pts_status);
					if (intersect_status == 1) {
						int fid = (*f_it).idx();
						EigenTriMesh::FaceHandle fh = mesh_out.face_handle(fid);
						mesh_out.delete_face(fh);
						delete_fids.push_back((*f_it).idx());
						intersect_fids.push_back((*f_it).idx());
						if (pts_status[0]) { kdtree_vids.insert(fv_ids[0]); }
						if (pts_status[1]) { kdtree_vids.insert(fv_ids[1]); }
						if (pts_status[2]) { kdtree_vids.insert(fv_ids[2]); }
					}
					else if (intersect_status == 2) {
						int fid = (*f_it).idx();
						mesh_out.delete_face(mesh_out.face_handle(fid));
						delete_fids.push_back((*f_it).idx());
					}
				}


				// build kdtree for vertices 
				PointCloud<double> cloud;
				my_kd_tree_t kd_index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
				nanoflann::KNNResultSet<double> resultSet(1);
				std::unordered_map<int, int> kd2vid;
				for (int vid : kdtree_vids) {
					PointCloud<double>::Point p;
					p.x = mesh_.point(mesh_.vertex_handle(vid))[0];
					p.y = mesh_.point(mesh_.vertex_handle(vid))[1];
					p.z = mesh_.point(mesh_.vertex_handle(vid))[2];
					cloud.pts.push_back(p);
					kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);
					kd2vid.insert(std::make_pair(cloud.pts.size() - 1, vid));
				}

				// Crop intersection face and add to the mesh_out
				for (int fid : intersect_fids) {
					std::vector<Eigen::Vector3f> fv_pts;
					std::vector<Eigen::Vector2f> fv_vts;
					std::vector<Eigen::Vector3f> cropped_fv_pts;
					std::vector<Eigen::Vector2f> cropped_fv_vts;
					int face_texture_index = mesh_.texture_index(mesh_.face_handle(fid));
					for (auto fheh_it = mesh_.fh_begin(mesh_.face_handle(fid)); fheh_it.is_valid(); ++fheh_it) {
						EigenTriMesh::VertexHandle fvh = mesh_.to_vertex_handle(*fheh_it);
						fv_pts.push_back(mesh_.point(fvh));
						fv_vts.push_back(mesh_.texcoord2D(*fheh_it));
					}
					SH_TRIM(fv_pts, bbox, cropped_fv_pts);
					update_vt(fv_pts, fv_vts, cropped_fv_pts, cropped_fv_vts);

					// add cropped vertices to mesh_out
					std::vector<EigenTriMesh::VertexHandle> cropped_fv_handles;
					for (int tmp_id = 0; tmp_id < cropped_fv_pts.size();++tmp_id) {
						Eigen::Vector3f point = cropped_fv_pts[tmp_id];
						Eigen::Vector3f pt_tmp;
						EigenTriMesh::VertexHandle vh;
						const size_t num_results = 1;
						size_t ret_index;
						double out_dist_sqr;
						double p_tmp[3] = { point[0],point[1],point[2] };
						resultSet.init(&ret_index, &out_dist_sqr);
						kd_index.findNeighbors(resultSet, p_tmp, nanoflann::SearchParams(10));
						vh = mesh_out.vertex_handle(kd2vid[ret_index]);
						// important: when condition1: there is no close enough point, or condition2: the close enough point already been deleted
						if (abs(sqrt(out_dist_sqr)) > 1e-3 || mesh_out.status(vh).deleted()) {
							PointCloud<double>::Point p;
							p.x = point[0];
							p.y = point[1];
							p.z = point[2];
							cloud.pts.push_back(p);
							kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);
							vh = mesh_out.add_vertex(point);
							kd2vid.insert(std::make_pair(cloud.pts.size() - 1, vh.idx()));
						}
						cropped_fv_handles.push_back(vh);
					}

					// add cropped faces to mesh_out
					std::vector<EigenTriMesh::VertexHandle> fh;
					for (int j = 2; j < cropped_fv_handles.size(); ++j) {
						fh.clear();
						fh.push_back(cropped_fv_handles[0]);
						fh.push_back(cropped_fv_handles[j - 1]);
						fh.push_back(cropped_fv_handles[j]);
						EigenTriMesh::FaceHandle tmp_fh=mesh_out.add_face(fh);
						if (!tmp_fh.is_valid()) { continue; }
						EigenTriMesh::HalfedgeHandle heh=mesh_out.halfedge_handle(tmp_fh);
						mesh_out.set_texcoord2D(heh, cropped_fv_vts[0]);
						heh = mesh_out.next_halfedge_handle(heh);
						mesh_out.set_texcoord2D(heh, cropped_fv_vts[j-1]);
						heh = mesh_out.next_halfedge_handle(heh);
						mesh_out.set_texcoord2D(heh, cropped_fv_vts[j]);
						mesh_out.set_texture_index(tmp_fh, face_texture_index);
					}
				}
				mesh_out.garbage_collection();
				write_manual_tex(mesh_out, input_mesh_path_, output_path, true);
				//write_manual(mesh_out, output_path);
			}

		}

		void Start_multi_outside() {

			for (int crop_id = 0; crop_id < bbox_list_.size(); ++crop_id) {
				OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out;
				std::vector<float> bbox = bbox_list_[crop_id];
				std::string output_path = output_list_[crop_id];

				EigenTriMesh mesh_out;
				mesh_out = mesh_;
				mesh_out.request_face_status();
				mesh_out.request_edge_status();
				mesh_out.request_halfedge_status();
				mesh_out.request_vertex_status();
				mesh_out.request_halfedge_texcoords2D();
				mesh_out.request_face_texture_index();
				if (!mesh_out.get_property_handle(texindex_out, "TextureMapping")) {
					has_material_ = false;
					std::cout << "input mesh doesn't have material" << std::endl;
				}
				else {
					has_material_ = true;
				}


				// iterate all faces, delete the outside&intersect faces
				EigenTriMesh::FaceIter f_it, f_end(mesh_.faces_end());
				std::vector<int> delete_fids;
				std::vector<int> intersect_fids;
				std::vector<FaceInter> intersect_fs;
				std::set<int> kdtree_vids;
				for (f_it = mesh_.faces_begin(); f_it != f_end; ++f_it) {
					FaceInter faceinter;
					faceinter.fid = (*f_it).idx();

					if (faceinter.fid == 48) {
						int a = 1;
					}

					std::vector<Eigen::Vector3f> fv_pts;
					std::vector<Eigen::Vector2f> fv_vts;
					std::vector<int> fv_ids;
					std::vector<bool> pts_status;
					EigenTriMesh::HalfedgeHandle heh;
					if ((*f_it).idx() == 1260) {
						int a = 1;
					}
					for (auto fheh_it = mesh_.fh_begin(*f_it); fheh_it.is_valid(); ++fheh_it) {
						EigenTriMesh::VertexHandle fvh = mesh_.to_vertex_handle(*fheh_it);
						fv_pts.push_back(mesh_.point(fvh));
						fv_ids.push_back(fvh.idx());
						fv_vts.push_back(mesh_.texcoord2D(*fheh_it));
					}
					faceinter.v1 = fv_pts[0];
					faceinter.v2 = fv_pts[1];
					faceinter.v3 = fv_pts[2];
					faceinter.vt1 = fv_vts[0];
					faceinter.vt2 = fv_vts[1];
					faceinter.vt3 = fv_vts[2];
					faceinter.v1_id = fv_ids[0];
					faceinter.v2_id = fv_ids[1];
					faceinter.v3_id = fv_ids[2];
					step1_triangle_intersect_bbox_outside(faceinter, bbox);
					//int intersect_status = triangle_intersect_bbox_outside(fv_pts, bbox, pts_status); // pts_status: whether each point is outside bbox
					if (faceinter.type >0 && faceinter.type<8) {
						int fid = (*f_it).idx();
						EigenTriMesh::FaceHandle fh = mesh_out.face_handle(fid);
						mesh_out.delete_face(fh);
						delete_fids.push_back((*f_it).idx());
						intersect_fids.push_back((*f_it).idx());
						intersect_fs.push_back(faceinter);
						if (faceinter.v1_inside) { kdtree_vids.insert(fv_ids[0]); }
						if (faceinter.v2_inside) { kdtree_vids.insert(fv_ids[1]); }
						if (faceinter.v3_inside) { kdtree_vids.insert(fv_ids[2]); }
					}	
					else if (faceinter.type == 0) {
						int fid = (*f_it).idx();
						mesh_out.delete_face(mesh_out.face_handle(fid));
						delete_fids.push_back((*f_it).idx());
					}
				}

				// build kdtree for vertices 
				PointCloud<double> cloud;
				my_kd_tree_t kd_index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
				nanoflann::KNNResultSet<double> resultSet(1);
				std::unordered_map<int, int> kd2vid;
				for (int vid : kdtree_vids) {
					PointCloud<double>::Point p;
					p.x = mesh_.point(mesh_.vertex_handle(vid))[0];
					p.y = mesh_.point(mesh_.vertex_handle(vid))[1];
					p.z = mesh_.point(mesh_.vertex_handle(vid))[2];
					cloud.pts.push_back(p);
					kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);
					kd2vid.insert(std::make_pair(cloud.pts.size() - 1, vid));
				}
				// Crop intersection face and add to the mesh_out
				for (auto faceinter : intersect_fs) {
					if (faceinter.fid == 48) {
						int a = 1;
					}
					std::vector<std::vector<Eigen::Vector3f>> triangulation;
					std::vector<std::vector<Eigen::Vector2f>> triangulation_vts;
					step2_triangulation(faceinter, triangulation,triangulation_vts);
					for (int tid = 0; tid < triangulation.size();tid++) {
						// add cropped vertices to mesh_out
						std::vector<EigenTriMesh::VertexHandle> cropped_fv_handles;
						for (int pid = 0; pid < triangulation[tid].size(); ++pid) {
							Eigen::Vector3f point = triangulation[tid][pid];
							Eigen::Vector3f pt_tmp;
							EigenTriMesh::VertexHandle vh;
							const size_t num_results = 1;
							size_t ret_index;
							double out_dist_sqr;
							double p_tmp[3] = { point[0],point[1],point[2] };
							resultSet.init(&ret_index, &out_dist_sqr);
							kd_index.findNeighbors(resultSet, p_tmp, nanoflann::SearchParams(10));
							vh = mesh_out.vertex_handle(kd2vid[ret_index]);
							// important: when condition1: there is no close enough point, or condition2: the close enough point already been deleted
							if (abs(sqrt(out_dist_sqr)) > 1e-3 || mesh_out.status(vh).deleted()) {
								PointCloud<double>::Point p;
								p.x = point[0];
								p.y = point[1];
								p.z = point[2];
								cloud.pts.push_back(p);
								kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);
								vh = mesh_out.add_vertex(point);
								kd2vid.insert(std::make_pair(cloud.pts.size() - 1, vh.idx()));
							}
							cropped_fv_handles.push_back(vh);
						}
						// add cropped faces to mesh_out
						int face_texture_index = mesh_.texture_index(mesh_.face_handle(faceinter.fid));
						EigenTriMesh::FaceHandle tmp_fh = mesh_out.add_face(cropped_fv_handles);
						if (!tmp_fh.is_valid()) { continue; }
						EigenTriMesh::HalfedgeHandle heh = mesh_out.halfedge_handle(tmp_fh);
						mesh_out.set_texcoord2D(heh, triangulation_vts[tid][0]);
						heh = mesh_out.next_halfedge_handle(heh);
						mesh_out.set_texcoord2D(heh, triangulation_vts[tid][1]);
						heh = mesh_out.next_halfedge_handle(heh);
						mesh_out.set_texcoord2D(heh, triangulation_vts[tid][2]);
						mesh_out.set_texture_index(tmp_fh, face_texture_index);
					}
				}
				//for (auto faceinter: intersect_fs){
				//	std::vector<Eigen::Vector3f> fv_pts;
				//	std::vector<Eigen::Vector2f> fv_vts;
				//	std::vector<Eigen::Vector3f> cropped_fv_pts;
				//	std::vector<Eigen::Vector2f> cropped_fv_vts;
				//	int face_texture_index = mesh_.texture_index(mesh_.face_handle(faceinter.fid));
				//	for (auto fheh_it = mesh_.fh_begin(mesh_.face_handle(faceinter.fid)); fheh_it.is_valid(); ++fheh_it) {
				//		EigenTriMesh::VertexHandle fvh = mesh_.to_vertex_handle(*fheh_it);
				//		fv_pts.push_back(mesh_.point(fvh));
				//		fv_vts.push_back(mesh_.texcoord2D(*fheh_it));
				//	}

				//	SH_TRIM_outside(fv_pts, bbox, cropped_fv_pts);
				//	update_vt(fv_pts, fv_vts, cropped_fv_pts, cropped_fv_vts);

				//	// add cropped vertices to mesh_out
				//	std::vector<EigenTriMesh::VertexHandle> cropped_fv_handles;
				//	for (int tmp_id = 0; tmp_id < cropped_fv_pts.size(); ++tmp_id) {
				//		Eigen::Vector3f point = cropped_fv_pts[tmp_id];
				//		Eigen::Vector3f pt_tmp;
				//		EigenTriMesh::VertexHandle vh;
				//		const size_t num_results = 1;
				//		size_t ret_index;
				//		double out_dist_sqr;
				//		double p_tmp[3] = { point[0],point[1],point[2] };
				//		resultSet.init(&ret_index, &out_dist_sqr);
				//		kd_index.findNeighbors(resultSet, p_tmp, nanoflann::SearchParams(10));
				//		vh = mesh_out.vertex_handle(kd2vid[ret_index]);
				//		// important: when condition1: there is no close enough point, or condition2: the close enough point already been deleted
				//		if (abs(sqrt(out_dist_sqr)) > 1e-3 || mesh_out.status(vh).deleted()) {
				//			PointCloud<double>::Point p;
				//			p.x = point[0];
				//			p.y = point[1];
				//			p.z = point[2];
				//			cloud.pts.push_back(p);
				//			kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);
				//			vh = mesh_out.add_vertex(point);
				//			kd2vid.insert(std::make_pair(cloud.pts.size() - 1, vh.idx()));
				//		}
				//		cropped_fv_handles.push_back(vh);
				//	}

				//	// add cropped faces to mesh_out
				//	std::vector<EigenTriMesh::VertexHandle> fh;
				//	if (cropped_fv_handles.size() != 5) {
				//		for (int j = 2; j < cropped_fv_handles.size(); ++j) {
				//			fh.clear();
				//			fh.push_back(cropped_fv_handles[0]);
				//			fh.push_back(cropped_fv_handles[j - 1]);
				//			fh.push_back(cropped_fv_handles[j]);
				//			EigenTriMesh::FaceHandle tmp_fh = mesh_out.add_face(fh);
				//			if (!tmp_fh.is_valid()) { continue; }
				//			EigenTriMesh::HalfedgeHandle heh = mesh_out.halfedge_handle(tmp_fh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[0]);
				//			heh = mesh_out.next_halfedge_handle(heh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[j - 1]);
				//			heh = mesh_out.next_halfedge_handle(heh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[j]);
				//			mesh_out.set_texture_index(tmp_fh, face_texture_index);
				//		}
				//	}
				//	else {
				//		for (int j = 2; j < cropped_fv_handles.size(); ++j) {
				//			fh.clear();
				//			fh.push_back(cropped_fv_handles[0]);
				//			fh.push_back(cropped_fv_handles[1]);
				//			fh.push_back(cropped_fv_handles[3]);
				//			EigenTriMesh::FaceHandle tmp_fh = mesh_out.add_face(fh);
				//			if (!tmp_fh.is_valid()) { continue; }
				//			EigenTriMesh::HalfedgeHandle heh = mesh_out.halfedge_handle(tmp_fh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[0]);
				//			heh = mesh_out.next_halfedge_handle(heh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[1]);
				//			heh = mesh_out.next_halfedge_handle(heh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[3]);
				//			mesh_out.set_texture_index(tmp_fh, face_texture_index);

				//			fh.clear();
				//			fh.push_back(cropped_fv_handles[1]);
				//			fh.push_back(cropped_fv_handles[2]);
				//			fh.push_back(cropped_fv_handles[3]);
				//			tmp_fh = mesh_out.add_face(fh);
				//			if (!tmp_fh.is_valid()) { continue; }
				//			 heh = mesh_out.halfedge_handle(tmp_fh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[1]);
				//			heh = mesh_out.next_halfedge_handle(heh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[2]);
				//			heh = mesh_out.next_halfedge_handle(heh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[3]);
				//			mesh_out.set_texture_index(tmp_fh, face_texture_index);

				//			fh.clear();
				//			fh.push_back(cropped_fv_handles[0]);
				//			fh.push_back(cropped_fv_handles[3]);
				//			fh.push_back(cropped_fv_handles[4]);
				//			tmp_fh = mesh_out.add_face(fh);
				//			if (!tmp_fh.is_valid()) { continue; }
				//			heh = mesh_out.halfedge_handle(tmp_fh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[0]);
				//			heh = mesh_out.next_halfedge_handle(heh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[3]);
				//			heh = mesh_out.next_halfedge_handle(heh);
				//			mesh_out.set_texcoord2D(heh, cropped_fv_vts[4]);
				//			mesh_out.set_texture_index(tmp_fh, face_texture_index);
				//		}
				//	}

				//}
				mesh_out.garbage_collection();
				write_manual_tex(mesh_out, input_mesh_path_, output_path, true);
				//write_manual(mesh_out, output_path);
			}

		}
		void write_manual_tex(EigenTriMesh& mesh, std::string in_path, std::string path, bool copy_texture) {
			if (mesh.n_faces() == 0) {
				return;
			}
			bool useMatrial = false;
			OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out;
			std::vector<int> crop_texindex_ids;
			if (mesh.get_property_handle(texindex_out, "TextureMapping")) {
				useMatrial = true;
			}


			std::ofstream ofs(path);
			Vector3f pts;
			Vector2f tex;
			EigenTriMesh::HalfedgeHandle heh;

			//write mtllib path
			if (useMatrial) {
				std::string mtl_name = fs::v1::path(path).filename().replace_extension(".mtl").string();
				ofs << "mtllib " << mtl_name << "\n";
				ofs << std::fixed << std::setprecision(12);
			}

			//wrtie v
			for (int i = 0; i < mesh.n_vertices(); ++i) {
				pts = mesh.point(mesh.vertex_handle(i));
				ofs << "v " << pts[0] << " " << pts[1] << " " << pts[2] << "\n";
			}
			//wrtie vt
			if (useMatrial) {
				for (int i = 0; i < mesh.n_faces(); ++i) {
					heh = mesh.halfedge_handle(mesh.face_handle(i));
					tex = mesh.texcoord2D(heh);
					ofs << "vt " << tex[0] << " " << tex[1] << "\n";

					heh = mesh.next_halfedge_handle(heh);
					tex = mesh.texcoord2D(heh);
					ofs << "vt " << tex[0] << " " << tex[1] << "\n";

					heh = mesh.next_halfedge_handle(heh);
					tex = mesh.texcoord2D(heh);
					ofs << "vt " << tex[0] << " " << tex[1] << "\n";
				}
			}

			//wrtie face
			int uv = 1, v1, v2, v3;
			int current_tex_index = 0;
			int tmp_tex_index = 0;
			for (int i = 0; i < mesh.n_faces(); ++i) {
				if (useMatrial) {
					if (i == 0) {
						current_tex_index = mesh.texture_index(mesh.face_handle(i));
						ofs << "usemtl mat" << current_tex_index << '\n';
						crop_texindex_ids.push_back(current_tex_index);
					}
					else {
						tmp_tex_index = mesh.texture_index(mesh.face_handle(i));
						if (current_tex_index != tmp_tex_index) {
							current_tex_index = tmp_tex_index;
							ofs << "usemtl mat" << current_tex_index << '\n';
							crop_texindex_ids.push_back(current_tex_index);
						}
					}

				}
				auto fv_it = mesh.fv_begin(mesh.face_handle(i));
				v1 = (*fv_it).idx();
				++fv_it;
				v2 = (*fv_it).idx();
				++fv_it;
				v3 = (*fv_it).idx();
				if (useMatrial) {
					ofs << "f " << v1 + 1 << "/" << uv << " " << v2 + 1 << "/" << uv + 1 << " " << v3 + 1 << "/" << uv + 2 << std::endl;
				}
				else {
					ofs << "f " << v1 + 1 << " " << v2 + 1 << " " << v3 + 1 << std::endl;
				}
				uv += 3;
			}
			ofs.close();

			if (!useMatrial) {
				return;
			}
			// write_mtl
			std::string out_filename = fs::v1::path(path).filename().string();

			fs::v1::path mtl_in_path = fs::v1::path(in_path).replace_extension(".mtl");
			fs::v1::path mtl_out_path = fs::v1::path(path).replace_extension(".mtl");
			std::ifstream mtl_in(mtl_in_path);
			std::ofstream mtl_out(mtl_out_path);

			std::vector<std::string> texture_name;
			std::string line;
			// sort and unique the tex_index
			sort(crop_texindex_ids.begin(), crop_texindex_ids.end());
			crop_texindex_ids.erase(unique(crop_texindex_ids.begin(), crop_texindex_ids.end()), crop_texindex_ids.end());
			if (mtl_in.is_open())
			{
				for (size_t i = 0; i < crop_texindex_ids.size(); i++) {
					std::string tex_name = mesh.property(texindex_out)[crop_texindex_ids[i]];
					std::string tex_path;
					if (fs::exists(fs::path(tex_name))) {
						tex_path = tex_name;
					}
					else {
						tex_path = (fs::path(in_path).parent_path() / tex_name).string();
					}
					if (!mesh.property(texindex_out)[crop_texindex_ids[i]].empty()) {
						mtl_out << "newmtl " << "mat" << crop_texindex_ids[i] << '\n';
						mtl_out << "Ka 1.000 1.000 1.000" << '\n';
						mtl_out << "Kd 1.000 1.000 1.000" << '\n';
						mtl_out << "illum 1" << '\n';
						if (copy_texture) {
							mtl_out << "map_Kd " << tex_name << "\n";
							texture_name.push_back(tex_name);
						}
						else {
							mtl_out << "map_Kd " << tex_path << "\n";
							texture_name.push_back(tex_path);
						}

					}

				}	
				mtl_in.close();
				mtl_out.close();
			}
			else {
				std::cout << "[ERROR]can't open mtl" << std::endl;
			}
			//copy texture file
			if (copy_texture) {
				for (auto it = texture_name.begin(); it != texture_name.end(); ++it) {

					fs::v1::path tex_out_path = fs::v1::path(mtl_out_path).replace_filename(*it);
					fs::v1::path tex_in_path = fs::v1::path(mtl_in_path).replace_filename(*it);
					if (!fs::v1::exists(tex_out_path) && fs::v1::exists(tex_in_path)) {
						fs::copy_file(tex_in_path, tex_out_path);
					}
				}
			}
		}


		void write_manual(EigenTriMesh mesh,std::string path) {
			if (mesh.n_faces() == 0) {
				return;
			}
			std::ofstream ofs(path);
			Vector3f pts;
			Vector2f tex;
			EigenTriMesh::HalfedgeHandle heh;
			ofs << std::fixed << std::setprecision(12);
			//wrtie v
			for (int i = 0; i < mesh.n_vertices(); ++i) {
				pts = mesh.point(mesh.vertex_handle(i));
				ofs << "v " << pts[0] << " " << pts[1] << " " << pts[2] << "\n";
			}

			//wrtie face
			int uv = 1, v1, v2, v3;
			for (int i = 0; i < mesh.n_faces(); ++i) {
				auto fv_it = mesh.fv_begin(mesh.face_handle(i));
				v1 = (*fv_it).idx();
				++fv_it;
				v2 = (*fv_it).idx();
				++fv_it;
				v3 = (*fv_it).idx();
				ofs << "f " << v1+1 << " " << v2+1 << " " << v3+1<< "\n";
			}
			ofs.close();
		}
 };

 void MESH_TRIM(std::string input_path, std::vector<std::string> output_path, std::vector<std::vector<float>> bbox_list,int keep_aoi) {
		 MeshTrim::Trimming T(input_path, bbox_list,output_path,keep_aoi);
		 if (keep_aoi) {
			 T.Start_multi_inside();
		 }
		 else {
			 T.Start_multi_outside();
		 }
		 
	}
}

#endif MESHTRIM1_H