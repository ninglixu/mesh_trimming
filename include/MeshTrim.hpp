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
			float step = (1 / dir[0]) * std::abs(bbox[0] - pt1[0]);
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
			float step = (1 / dir[0]) * std::abs(bbox[1] - pt1[0]);
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
			float step = (1 / dir[1]) * std::abs(bbox[2] - pt1[1]);
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
			float step = (1 / dir[1]) * std::abs(bbox[3] - pt1[1]);
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
			float step = (1 / dir[2]) * std::abs(bbox[4] - pt1[2]);
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
			float step = (1 / dir[2]) * std::abs(bbox[5] - pt1[2]);
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
	reference: Sutherland�Hodgman algorithm https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
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

    class Trimming {

	public:
		std::string input_mesh_path_;
		EigenTriMesh mesh_;
		EigenTriMesh mesh_out_;
		std::vector<std::vector<float>> bbox_list_; // list of cropping volume, each has xmin,xmax,ymin,ymax,zmin,zmax
		std::vector<std::string> output_list_;

		Trimming(std::string input_path, std::vector<std::vector<float>> bbox_list_raw, std::vector<std::string> output_list) {
			input_mesh_path_ = input_path;
			bbox_list_ = bbox_list_raw;
			output_list_ = output_list;


			mesh_.request_halfedge_texcoords2D();
			mesh_.request_face_texture_index();

			OpenMesh::IO::Options opt(OpenMesh::IO::Options::FaceTexCoord);
			opt += OpenMesh::IO::Options::FaceColor;
			if (!OpenMesh::IO::read_mesh(mesh_, input_path, opt))
			{
				std::cerr << "[ERROR]Cannot read mesh from " << input_path << std::endl;
			}
			mesh_out_ = mesh_;
			mesh_out_.request_face_status();
			mesh_out_.request_edge_status();
			mesh_out_.request_halfedge_status();
			mesh_out_.request_vertex_status();
			
			int a=mesh_out_.n_faces();
			int b = 1;
		}

		void Start() {
			std::vector<float> bbox = bbox_list_[0];

			// iterate all faces, delete the outside&intersect faces
			EigenTriMesh::FaceIter f_it, f_end(mesh_.faces_end());
			std::vector<int> delete_fids;
			std::vector<int> intersect_fids;
			std::set<int> kdtree_vids;
			for (f_it = mesh_.faces_begin(); f_it != f_end; ++f_it) {
				std::vector<Eigen::Vector3f> fv_pts;
				std::vector<int> fv_ids;
				std::vector<bool> pts_status;
				for (EigenTriMesh::FaceVertexIter fv_it=mesh_.fv_begin(*f_it); fv_it.is_valid(); ++fv_it) {
					fv_pts.push_back(mesh_.point(*fv_it));
					EigenTriMesh::VertexHandle fvh = *fv_it;
					fv_ids.push_back(fvh.idx());
				}
				int intersect_status = triangle_intersect_bbox(fv_pts, bbox,pts_status);
				if (intersect_status == 1) { 
					int fid = (*f_it).idx();
					EigenTriMesh::FaceHandle fh = mesh_out_.face_handle(fid);
					mesh_out_.delete_face(fh);
					delete_fids.push_back((*f_it).idx()); 
					intersect_fids.push_back((*f_it).idx());
					if (pts_status[0]) { kdtree_vids.insert(fv_ids[0]);}
					if (pts_status[1]) { kdtree_vids.insert(fv_ids[1]); }
					if (pts_status[2]) { kdtree_vids.insert(fv_ids[2]); }
				}
				else if (intersect_status == 2) { 
					int fid = (*f_it).idx();
					mesh_out_.delete_face(mesh_out_.face_handle(fid));
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
				std::vector<Eigen::Vector3f> cropped_fv_pts;
				for (EigenTriMesh::FaceVertexIter fv_it=mesh_.fv_begin(mesh_.face_handle(fid)); fv_it.is_valid(); ++fv_it) {
					fv_pts.push_back(mesh_.point(*fv_it));
				}
				if (fid == 13429 || fid==13439) {
					int a = 1;
				}
				SH_TRIM(fv_pts, bbox, cropped_fv_pts);

				// add cropped vertices to mesh_out
				std::vector<EigenTriMesh::VertexHandle> cropped_fv_handles;
				for (Eigen::Vector3f point : cropped_fv_pts) {
					Eigen::Vector3f pt_tmp;
					EigenTriMesh::VertexHandle vh;
					const size_t num_results = 1;
					size_t ret_index;
					double out_dist_sqr;
					double p_tmp[3] = { point[0],point[1],point[2] };
					resultSet.init(&ret_index, &out_dist_sqr);
					kd_index.findNeighbors(resultSet, p_tmp, nanoflann::SearchParams(10));
					vh = mesh_out_.vertex_handle(kd2vid[ret_index]);
					// important: when condition1: there is no close enough point, or condition2: the close enough point already been deleted
					if (abs(sqrt(out_dist_sqr)) > 1e-3 || mesh_out_.status(vh).deleted()) {
						PointCloud<double>::Point p;
						p.x = point[0];
						p.y = point[1];
						p.z = point[2];
						cloud.pts.push_back(p);
						kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);

						vh = mesh_out_.add_vertex(point);
						kd2vid.insert(std::make_pair(cloud.pts.size() - 1, vh.idx()));
					}
					cropped_fv_handles.push_back(vh);
				}

				// add cropped faces to mesh_out
				std::vector<EigenTriMesh::VertexHandle> fh;
				for (int j = 2; j < cropped_fv_handles.size(); ++j) {
					fh.clear();
					fh.push_back(cropped_fv_handles[0]);
					fh.push_back(cropped_fv_handles[j-1]);
					fh.push_back(cropped_fv_handles[j]);
					mesh_out_.add_face(fh);
				}
			}
			mesh_out_.garbage_collection();
		}
   
		void Start_multi() {
			for (int crop_id = 0; crop_id < bbox_list_.size(); ++crop_id) {
				std::vector<float> bbox = bbox_list_[crop_id];
				std::string output_path = output_list_[crop_id];

				EigenTriMesh mesh_out;
				mesh_out = mesh_;
				mesh_out.request_face_status();
				mesh_out.request_edge_status();
				mesh_out.request_halfedge_status();
				mesh_out.request_vertex_status();


				// iterate all faces, delete the outside&intersect faces
				EigenTriMesh::FaceIter f_it, f_end(mesh_.faces_end());
				std::vector<int> delete_fids;
				std::vector<int> intersect_fids;
				std::set<int> kdtree_vids;
				for (f_it = mesh_.faces_begin(); f_it != f_end; ++f_it) {
					if ((*f_it).idx() == 22317) {
						int a = 1;
					}
					std::vector<Eigen::Vector3f> fv_pts;
					std::vector<int> fv_ids;
					std::vector<bool> pts_status;
					for (EigenTriMesh::FaceVertexIter fv_it = mesh_.fv_begin(*f_it); fv_it.is_valid(); ++fv_it) {
						fv_pts.push_back(mesh_.point(*fv_it));
						EigenTriMesh::VertexHandle fvh = *fv_it;
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
					std::vector<Eigen::Vector3f> cropped_fv_pts;
					for (EigenTriMesh::FaceVertexIter fv_it = mesh_.fv_begin(mesh_.face_handle(fid)); fv_it.is_valid(); ++fv_it) {
						fv_pts.push_back(mesh_.point(*fv_it));
					}
					if (fid == 13429 || fid == 13439) {
						int a = 1;
					}
					SH_TRIM(fv_pts, bbox, cropped_fv_pts);

					// add cropped vertices to mesh_out
					std::vector<EigenTriMesh::VertexHandle> cropped_fv_handles;
					for (Eigen::Vector3f point : cropped_fv_pts) {
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
						mesh_out.add_face(fh);
					}
				}
				mesh_out.garbage_collection();
				write_manual(mesh_out, output_path);
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

 void MESH_TRIM(std::string input_path, std::vector<std::string> output_path, std::vector<std::vector<float>> bbox_list) {
		 MeshTrim::Trimming T(input_path, bbox_list,output_path);
		 T.Start_multi();
	}
}

#endif MESHTRIM1_H