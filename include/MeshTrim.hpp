#ifndef MESHTRIM_H
#define MESHTRIM_H
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
#include <boost/format.hpp>

//nanoflann hpp
//#include <nanoflann/nanoflann.hpp>
//#include "nanoflann/utils.h"

#include "Tools.hpp"

namespace MeshEdit{

    using namespace std::chrono;
    using namespace Eigen;
    using namespace nanoflann;
    namespace fs = std::experimental::filesystem;

    class Trimming
    {
    public:
        std::string input_mesh_path;
        EigenTriMesh mesh;
        EigenTriMesh mesh_out;

        int added_faces = 0;
        int invalid_faces = 0;

        double threshold = 0.0001;

        //check whether two points are duplicate
        double threshold1 = 1e-3;

        // clipping bounding box [left,right,bot,top] in camera space
        Vector4f boundingbox;

        // camera matrix
        Matrix4d camera;
        Matrix4d camera_inv;

        // four planes of bounding box in camera space(homogeneous)
        Vector4f plane_left;
        Vector4f plane_right;
        Vector4f plane_bot;
        Vector4f plane_top;

        //"r"<--> right plane
        //"l"<-->left plane
        //"b"<--> bot plane
        //"t"<-->top plane
        std::map<char, Vector4f> plane_map;
        std::unordered_map<EigenTriMesh::VertexHandle, EigenTriMesh::VertexHandle> old2new;
        std::unordered_map<int, EigenTriMesh::VertexHandle> kd2new;

        //inter point check count
        int inter_error;
        std::vector<size_t> fidxs;

        // initialize kd-tree

        // construct a kd-tree index:
        
        const size_t num_results = 1;
        size_t ret_index;
        double out_dist_sqr;
        Trimming(EigenTriMesh& obj, std::string input_path) {
            input_mesh_path = input_path;
            mesh = obj;
            inter_error = 0;
            mesh_out.request_halfedge_texcoords2D();
            mesh_out.request_face_texture_index();
        }
        Trimming(EigenTriMesh& obj, Vector4f& frustum) {
            mesh = obj;
            boundingbox = frustum;

            inter_error = 0;

            //plane= [normal, -point*normal]
            plane_left << Vector3f(1, 0, 0), -(Vector3f(1, 0, 0).dot(Vector3f(boundingbox[0], boundingbox[2], 0)));
            plane_right << Vector3f(1, 0, 0), -(Vector3f(1, 0, 0).dot(Vector3f(boundingbox[1], boundingbox[2], 0)));
            plane_bot << Vector3f(0, 1, 0), -(Vector3f(0, 1, 0).dot(Vector3f(boundingbox[0], boundingbox[2], 0)));
            plane_top << Vector3f(0, 1, 0), -(Vector3f(0, 1, 0).dot(Vector3f(boundingbox[0], boundingbox[3], 0)));

            plane_map.insert(std::make_pair('l', plane_left));
            plane_map.insert(std::make_pair('r', plane_right));
            plane_map.insert(std::make_pair('t', plane_top));
            plane_map.insert(std::make_pair('b', plane_bot));

            mesh_out.request_halfedge_texcoords2D();
            mesh_out.request_face_texture_index();
        }
        ////////////////////////////////////////////////RASTER SECTION BEGIN////////////////////////////////////////
        void SH_raster() {
            if (!mesh.has_face_texture_index()) {
                mesh.request_face_texture_index();
                std::cout << "input mesh doesn't have texture index" << std::endl;
            }
            if (!mesh_out.has_face_texture_index()) {
                mesh_out.request_face_texture_index();
            }
            OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out, texindex;
            std::map<int, int> ti_map;
            if (!mesh_out.get_property_handle(texindex_out, "TextureMapping")) {
                mesh_out.add_property(texindex_out, "TextureMapping");
            }
            if (!mesh.get_property_handle(texindex, "TextureMapping")) {
                mesh.add_property(texindex, "TextureMapping");
                std::cout << "get meshA texturemapping error" << std::endl;
            }
            ///copy input tex_map to outputmesh
            for (auto it = mesh.property(texindex).begin(); it != mesh.property(texindex).end(); ++it) {
                mesh_out.property(texindex_out)[it->first] = it->second;
            }
            
            //kd-tree
            PointCloud<double> cloud;
            my_kd_tree_t kd_index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
            nanoflann::KNNResultSet<double> resultSet(1);

            EigenTriMesh::FaceIter f_it, f_end(mesh.faces_end());
            std::vector<EigenTriMesh::FaceHandle> intersect_faces;
            std::vector<EigenFace> egs;
            // add inside face first
            for (f_it = mesh.faces_begin(); f_it != f_end; ++f_it) {
                /*if ((*f_it).idx() == 28217 ) {
                    int a = 0;
                }   */   

                EigenFace eg;
                EigenTriMesh::VertexHandle vh;
                EigenTriMesh::FaceHandle fh;
                Eigen::Vector3f point;
                EigenTriMesh::HalfedgeHandle heh;
                std::vector<EigenTriMesh::VertexHandle> vh_list;
                std::vector<Eigen::Vector2f> tex_list;
                Location loc = SH_clip_raster((*f_it), eg);
                // when add inside face, only check old2new and store new vertexhandle to old2new.
                if (loc == INSIDE) {
                    vh_list.clear();
                    for (auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
                        vh_list.push_back(add_mesh_point(*fv_it, cloud, kd_index));
                    }
                    heh = mesh.halfedge_handle(*f_it);
                    tex_list.push_back(mesh.texcoord2D(heh));
                    heh = mesh.next_halfedge_handle(heh);
                    tex_list.push_back(mesh.texcoord2D(heh));
                    heh = mesh.next_halfedge_handle(heh);
                    tex_list.push_back(mesh.texcoord2D(heh));

                    //std::cout << vh_list[0].idx() << " " << vh_list[1].idx() << " " << vh_list[2].idx() << std::endl;
                    fh = mesh_out.add_face(vh_list);
                    if (fh.is_valid()) {
                        mesh_out.set_texture_index(fh,mesh.texture_index(*f_it));
                        heh = mesh_out.halfedge_handle(fh);
                        mesh_out.set_texcoord2D(heh, tex_list[0]);
                        heh = mesh_out.next_halfedge_handle(heh);
                        mesh_out.set_texcoord2D(heh, tex_list[1]);
                        heh = mesh_out.next_halfedge_handle(heh);
                        mesh_out.set_texcoord2D(heh, tex_list[2]);
                        added_faces++;
                    }
                    else {
                        invalid_faces++;
                        std::cout << invalid_faces << std::endl;
                    }
                }
                else if (loc == INTERSECT) {
                    intersect_faces.push_back((*f_it));
                    egs.push_back(eg);
                }
            }

            //add intersect face second
            for (int i = 0; i != intersect_faces.size(); ++i) {
                if (egs[i].v_list.size() > 2) {
                    if (egs[i].v_list.size() == 3) {
                        add_triangle_raster(egs[i], cloud, kd_index, resultSet);
                    }
                    else {
                        add_polygon_raster(egs[i], cloud, kd_index, resultSet);
                    }
                }
            }
            //std::cout << "inter point error: " << inter_error << std::endl;

            //completeness check
            //if (mesh_out.n_vertices() != kd2new.size() + old2new.size()) {
                //std::cout << "[ERROR] Mesh Crop: completeness check! ";
                //std::cout << " mesh out vertices: " << mesh_out.n_vertices();
                //std::cout << " kd2new+old2new: " << kd2new.size() + old2new.size() << std::endl;
            //}
            /*else {
                std::cout << "completeness check success! " << std::endl;

            }*/

        }

        void SH_raster_multi(std::vector<std::vector<double>> crop_list, std::vector<std::string> crop_out_list,bool copy_texture,double OFFX=0,double OFFY=0,double OFFZ=0) {
            if (!mesh.has_face_texture_index()) {
                mesh.request_face_texture_index();
                std::cout << "input mesh doesn't have texture index" << std::endl;
            }
            OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out, texindex;
            std::map<int, int> ti_map;
            if (!mesh.get_property_handle(texindex, "TextureMapping")) {
                mesh.add_property(texindex, "TextureMapping");
                std::cout << "get meshA texturemapping error" << std::endl;
            }

            //kd-tree
            PointCloud<double> cloud;
            my_kd_tree_t kd_index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
            nanoflann::KNNResultSet<double> resultSet(1);

            EigenTriMesh::FaceIter f_it, f_end(mesh.faces_end());


            std::vector<std::vector<EigenTriMesh::FaceHandle>> inter_or_in_faces_per_croplist(crop_list.size());
            std::vector<std::vector<EigenFace>> faces_per_croplist(crop_list.size());
            std::vector<std::vector<Location>> location_per_croplist(crop_list.size());

            // first determine each face belong to which crop_list
            for (f_it = mesh.faces_begin(); f_it != f_end; ++f_it) {
                /*if ((*f_it).idx() == 28217 ) {
                    int a = 0;
                }   */
                EigenFace eg;
                Location loc;
                for (int i = 0; i < crop_list.size(); ++i) {
                    SH_clip_raster((*f_it), crop_list[i], eg, loc);
                    if (loc == INSIDE || loc == INTERSECT) {
                        inter_or_in_faces_per_croplist[i].push_back(*f_it);
                        faces_per_croplist[i].push_back(eg);
                        location_per_croplist[i].push_back(loc);
                    }
                }
            }
            // create each crop_mesh
            for (int i = 0; i < crop_list.size(); ++i) {
                std::cout << "Start Cropping: " << crop_out_list[i] << std::endl;
                EigenTriMesh mesh_crop;
                mesh_crop.request_halfedge_texcoords2D();
                mesh_crop.request_face_texture_index();
                if (!mesh_crop.has_face_texture_index()) {
                    mesh_crop.request_face_texture_index();
                }
                OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out, texindex;
                std::map<int, int> ti_map;
                if (!mesh_crop.get_property_handle(texindex_out, "TextureMapping")) {
                    mesh_crop.add_property(texindex_out, "TextureMapping");
                }
                if (!mesh.get_property_handle(texindex, "TextureMapping")) {
                    mesh.add_property(texindex, "TextureMapping");
                    std::cout << "get meshA texturemapping error" << std::endl;
                }
                ///copy input tex_map to outputmesh
                for (auto it = mesh.property(texindex).begin(); it != mesh.property(texindex).end(); ++it) {
                    mesh_crop.property(texindex_out)[it->first] = it->second;
                }
                std::unordered_map<EigenTriMesh::VertexHandle, EigenTriMesh::VertexHandle> old2new_crop;
                std::unordered_map<int, EigenTriMesh::VertexHandle> kd2new_crop;
                PointCloud<double> cloud_crop;
                my_kd_tree_t kd_index_crop(3 /*dim*/, cloud_crop, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
                nanoflann::KNNResultSet<double> resultSet_crop(1);

                EigenFace eg;
                Location loc;
                EigenTriMesh::FaceHandle fh,fh_crop;
                EigenTriMesh::VertexHandle vh;
                Eigen::Vector3f point;
                EigenTriMesh::HalfedgeHandle heh;
                std::vector<EigenTriMesh::VertexHandle> vh_list;
                std::vector<Eigen::Vector2f> tex_list;
                std::vector<EigenTriMesh::FaceHandle> intersect_faces;
                std::vector<EigenFace> egs;
                int added_faces_crop = 0;
                int invalid_faces_crop = 0;
                for (int j = 0; j < inter_or_in_faces_per_croplist[i].size(); ++j) {

                    fh = inter_or_in_faces_per_croplist[i][j];
                    loc = location_per_croplist[i][j];
                    eg = faces_per_croplist[i][j];
                    // add inside face first
                    // when add inside face, only check old2new and store new vertexhandle to old2new.
                    if (loc == INSIDE) {
                        vh_list.clear();
                        tex_list.clear();
                        for (auto fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
                            vh_list.push_back(add_mesh_point(mesh_crop,*fv_it, cloud_crop, kd_index_crop,old2new_crop,kd2new_crop,crop_list[i]));
                        }
                        heh = mesh.halfedge_handle(fh);
                        tex_list.push_back(mesh.texcoord2D(heh));
                        heh = mesh.next_halfedge_handle(heh);
                        tex_list.push_back(mesh.texcoord2D(heh));
                        heh = mesh.next_halfedge_handle(heh);
                        tex_list.push_back(mesh.texcoord2D(heh));

                        //std::cout << vh_list[0].idx() << " " << vh_list[1].idx() << " " << vh_list[2].idx() << std::endl;
                        fh_crop = mesh_crop.add_face(vh_list);
                        if (fh_crop.is_valid()) {
                            mesh_crop.set_texture_index(fh_crop, mesh.texture_index(fh));
                            heh = mesh_crop.halfedge_handle(fh_crop);
                            mesh_crop.set_texcoord2D(heh, tex_list[0]);
                            heh = mesh_crop.next_halfedge_handle(heh);
                            mesh_crop.set_texcoord2D(heh, tex_list[1]);
                            heh = mesh_crop.next_halfedge_handle(heh);
                            mesh_crop.set_texcoord2D(heh, tex_list[2]);
                            added_faces_crop++;
                        }
                        else {
                            invalid_faces_crop++;
                            std::cout << invalid_faces_crop << std::endl;
                        }
                    }
                    else if (loc == INTERSECT) {
                        intersect_faces.push_back((fh));
                        egs.push_back(eg);
                    }
                }
                //add intersect face second
                for (int iter = 0; iter != intersect_faces.size(); ++iter) {
                    if (egs[iter].v_list.size() > 2) {
                        if (egs[iter].v_list.size() == 3) {
                            add_triangle_raster(mesh_crop, egs[iter], cloud_crop, kd_index_crop, resultSet_crop, old2new_crop, kd2new_crop, crop_list[i]);
                        }
                        else {
                            add_polygon_raster(mesh_crop, egs[iter], cloud_crop, kd_index_crop, resultSet_crop, old2new_crop, kd2new_crop, crop_list[i]);
                        }
                    }
                }
                //if (mesh_crop.n_vertices() != kd2new_crop.size() + old2new_crop.size()) {
                //    std::cout << "[ERROR]completeness check! " << std::endl;
                //    std::cout << "mesh out vertices: " << mesh_crop.n_vertices() << std::endl;
                //    std::cout << "kd2new+old2new: " << kd2new_crop.size() + old2new_crop.size() << std::endl;
                //}

                if (mesh_crop.n_vertices() == 0) {
                    //std::cout <<crop_out_list[i]<< ": No faces locaded in cropping area\n";
                    continue;
                    ////time end

                }
                //write obj
                write_manual(mesh_crop,input_mesh_path, crop_out_list[i], copy_texture);


                //write mtl
                //copy_mtl_tex(mesh_crop, input_mesh_path, crop_out_list[i], copy_texture);
            }
        }

        enum Location { INSIDE, OUTSIDE, INTERSECT };
        Location SH_clip_raster(EigenTriMesh::FaceHandle fh, EigenFace& eg) {
            //transform vertexs from world coordinates to camera coordinates, and store them in fv_coo
            std::vector<Eigen::Vector4f> fv_coo;
            std::vector<int> fv_id;
            for (auto fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
                Eigen::Vector4f pt;
                pt << mesh.point(*fv_it), 1;
                fv_coo.push_back(pt);
                fv_id.push_back((*fv_it).idx());
            }
            bool s1 = isinbbox(fv_coo[0]);
            bool s2 = isinbbox(fv_coo[1]);
            bool s3 = isinbbox(fv_coo[2]);
            if (s1 && s2 && s3) {
                return INSIDE;
            }
            // if three vertex all outside left/right/bottom/top plane
            else if ((outplane(fv_coo[0], 'l') && outplane(fv_coo[1], 'l') && outplane(fv_coo[2], 'l')) ||
                (outplane(fv_coo[0], 'r') && outplane(fv_coo[1], 'r') && outplane(fv_coo[2], 'r')) ||
                (outplane(fv_coo[0], 'b') && outplane(fv_coo[1], 'b') && outplane(fv_coo[2], 'b')) ||
                (outplane(fv_coo[0], 't') && outplane(fv_coo[1], 't') && outplane(fv_coo[2], 't'))) {
                return OUTSIDE;
            }
            else {
                std::vector<Eigen::Vector2f> tex_list;
                EigenFace_homo eg_h, after_left, after_right, after_bot, after_top;
                EigenFace eg_o;
                EigenTriMesh::HalfedgeHandle heh = mesh.halfedge_handle(fh);
                tex_list.push_back(mesh.texcoord2D(heh));
                heh = mesh.next_halfedge_handle(heh);
                tex_list.push_back(mesh.texcoord2D(heh));
                heh = mesh.next_halfedge_handle(heh);
                tex_list.push_back(mesh.texcoord2D(heh));
                eg_h.v_list = fv_coo;
                eg_h.tex_list = tex_list;

                //clip the triangular by four planes of bounding box respectively.
                after_left = SH_clip_single_raster(eg_h, 'l');
                after_bot = SH_clip_single_raster(after_left, 'b');
                after_right = SH_clip_single_raster(after_bot, 'r');
                after_top = SH_clip_single_raster(after_right, 't');

                if (after_top.v_list.size() < 3) {
                    return OUTSIDE;
                }
                else {
                    for (auto point = after_top.v_list.begin(); point != after_top.v_list.end(); ++point) {
                        Vector3f point_change_dehomo;
                        point_change_dehomo << (*point)[0] / (*point)[3], (*point)[1] / (*point)[3], (*point)[2] / (*point)[3];
                        int id = -1;
                        //after SH clip, points that are reserved should store vid, points that are newly added should store -1.
                        for (auto v_id = fv_id.begin(); v_id != fv_id.end(); ++v_id) {
                            if ((point_change_dehomo - mesh.point(mesh.vertex_handle(*v_id))).norm() < threshold1) {
                                id = *v_id;
                            }
                        }
                        eg_o.v_id.push_back(id);
                        eg_o.v_list.push_back(point_change_dehomo);
                    }
                    eg_o.tex_index = mesh.texture_index(fh);
                    eg_o.tex_list = after_top.tex_list;
                    check_vt(eg_o, eg);
                    return INTERSECT;
                }

            }
        }

        void SH_clip_raster(EigenTriMesh::FaceHandle fh, vector<double> crop_bbox, EigenFace& eg, Location& location) {
            //transform vertexs from world coordinates to camera coordinates, and store them in fv_coo
            std::vector<Eigen::Vector4f> fv_coo;
            std::vector<int> fv_id;
            for (auto fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
                Eigen::Vector4f pt;
                pt << mesh.point(*fv_it), 1;
                fv_coo.push_back(pt);
                fv_id.push_back((*fv_it).idx());
            }
            bool s1 = isinbbox(fv_coo[0],crop_bbox);
            bool s2 = isinbbox(fv_coo[1],crop_bbox);
            bool s3 = isinbbox(fv_coo[2],crop_bbox);
            if (s1 && s2 && s3) {
                location = INSIDE;
                return;
            }
            // if three vertex all outside left/right/bottom/top plane
            else if ((outplane(fv_coo[0], 'l',crop_bbox) && outplane(fv_coo[1], 'l', crop_bbox) && outplane(fv_coo[2], 'l', crop_bbox)) ||
                (outplane(fv_coo[0], 'r', crop_bbox) && outplane(fv_coo[1], 'r', crop_bbox) && outplane(fv_coo[2], 'r', crop_bbox)) ||
                (outplane(fv_coo[0], 'b', crop_bbox) && outplane(fv_coo[1], 'b', crop_bbox) && outplane(fv_coo[2], 'b', crop_bbox)) ||
                (outplane(fv_coo[0], 't', crop_bbox) && outplane(fv_coo[1], 't', crop_bbox) && outplane(fv_coo[2], 't', crop_bbox))) {
                location = OUTSIDE;
                return;
            }
            else {
                std::vector<Eigen::Vector2f> tex_list;
                EigenFace_homo eg_h, after_left, after_right, after_bot, after_top;
                EigenFace eg_o;
                EigenTriMesh::HalfedgeHandle heh = mesh.halfedge_handle(fh);
                tex_list.push_back(mesh.texcoord2D(heh));
                heh = mesh.next_halfedge_handle(heh);
                tex_list.push_back(mesh.texcoord2D(heh));
                heh = mesh.next_halfedge_handle(heh);
                tex_list.push_back(mesh.texcoord2D(heh));
                eg_h.v_list = fv_coo;
                eg_h.tex_list = tex_list;

                //clip the triangular by four planes of bounding box respectively.
                after_left = SH_clip_single_raster(eg_h, 'l', crop_bbox);
                after_bot = SH_clip_single_raster(after_left, 'b', crop_bbox);
                after_right = SH_clip_single_raster(after_bot, 'r', crop_bbox);
                after_top = SH_clip_single_raster(after_right, 't', crop_bbox);

                if (after_top.v_list.size() < 3) {
                    location = OUTSIDE;
                    return;
                }
                else {
                    for (auto point = after_top.v_list.begin(); point != after_top.v_list.end(); ++point) {
                        Vector3f point_change_dehomo;
                        point_change_dehomo << (*point)[0] / (*point)[3], (*point)[1] / (*point)[3], (*point)[2] / (*point)[3];
                        int id = -1;
                        //after SH clip, points that are reserved should store vid, points that are newly added should store -1.
                        for (auto v_id = fv_id.begin(); v_id != fv_id.end(); ++v_id) {
                            if ((point_change_dehomo - mesh.point(mesh.vertex_handle(*v_id))).norm() < threshold1) {
                                id = *v_id;
                            }
                        }
                        eg_o.v_id.push_back(id);
                        eg_o.v_list.push_back(point_change_dehomo);
                    }
                    eg_o.tex_index = mesh.texture_index(fh);
                    eg_o.tex_list = after_top.tex_list;
                    check_vt(eg_o, eg);
                    location = INTERSECT;
                    return;
                }

            }
        }


        EigenFace_homo SH_clip_single_raster(EigenFace_homo eg_h, char plane) {
            std::vector<Vector4f> output, out;
            std::vector<Eigen::Vector2f> output_tex, out_tex;
            int first, second;
            std::vector<Vector4f> fv_coo = eg_h.v_list;
            std::vector<Eigen::Vector2f> tex_list = eg_h.tex_list;

            //clip process: get three line segmentation, and clip each line segmentation
            for (int i = 0; i < fv_coo.size(); i++) {
                //first, second=a line segmentation, to preserve vertex order
                if (i == fv_coo.size() - 1) {
                    first = i, second = 0;
                }
                else {
                    first = i, second = i + 1;
                }

                //clip process: only preserve inside plane vertex.
                if (inplane(fv_coo[first], plane)) {
                    output.push_back(fv_coo[first]);
                    output_tex.push_back(tex_list[first]);
                    if (inplane(fv_coo[second], plane)) {
                        output.push_back(fv_coo[second]);
                        output_tex.push_back(tex_list[second]);
                    }
                    else {
                        Vector4f inter_point;
                        double alpha;
                        if (!inter_SH(fv_coo[first], fv_coo[second], plane, inter_point, alpha)) {
                            inter_point = fv_coo[first];
                            alpha = 0;
                        }
                        Eigen::Vector2f inter_tex = tex_list[first] + (tex_list[second] - tex_list[first]) * alpha;
                        output.push_back(inter_point);
                        output_tex.push_back(inter_tex);
                    }
                }
                else {
                    if (inplane(fv_coo[second], plane)) {
                        Vector4f inter_point;
                        double alpha;
                        if (!inter_SH(fv_coo[first], fv_coo[second], plane, inter_point, alpha)) {
                            inter_point = fv_coo[second];
                            alpha = 1;
                        }
                        output.push_back(inter_point);
                        output.push_back(fv_coo[second]);
                        Eigen::Vector2f inter_tex = tex_list[first] + (tex_list[second] - tex_list[first]) * alpha;
                        output_tex.push_back(inter_tex);
                        output_tex.push_back(tex_list[second]);
                    }
                }
            }

            //remove duplicate vertex
            for (int i = 0; i < output.size(); ++i) {
                int cnt = 0;
                if (out.size() == 0) {
                    out.push_back(output[i]);
                    out_tex.push_back(output_tex[i]);
                }
                else {
                    for (int j = 0; j != out.size(); ++j) {
                        if (((output[i]) - (out[j])).norm() < threshold1) {
                            cnt++;
                        }
                    }
                    if (cnt == 0) {
                        out.push_back(output[i]);
                        out_tex.push_back(output_tex[i]);
                    }
                }
            }

            EigenFace_homo ef;
            ef.tex_list = out_tex;
            ef.v_list = out;
            return ef;
        }
        EigenFace_homo SH_clip_single_raster(EigenFace_homo eg_h, char plane,vector<double> crop_bbox) {
            std::vector<Vector4f> output, out;
            std::vector<Eigen::Vector2f> output_tex, out_tex;
            int first, second;
            std::vector<Vector4f> fv_coo = eg_h.v_list;
            std::vector<Eigen::Vector2f> tex_list = eg_h.tex_list;

            //clip process: get three line segmentation, and clip each line segmentation
            for (int i = 0; i < fv_coo.size(); i++) {
                //first, second=a line segmentation, to preserve vertex order
                if (i == fv_coo.size() - 1) {
                    first = i, second = 0;
                }
                else {
                    first = i, second = i + 1;
                }

                //clip process: only preserve inside plane vertex.
                if (inplane(fv_coo[first], plane,crop_bbox)) {
                    output.push_back(fv_coo[first]);
                    output_tex.push_back(tex_list[first]);
                    if (inplane(fv_coo[second], plane, crop_bbox)) {
                        output.push_back(fv_coo[second]);
                        output_tex.push_back(tex_list[second]);
                    }
                    else {
                        Vector4f inter_point;
                        double alpha;
                        if (!inter_SH(fv_coo[first], fv_coo[second], plane, inter_point, alpha,crop_bbox)) {
                            inter_point = fv_coo[first];
                            alpha = 0;
                        }
                        Eigen::Vector2f inter_tex = tex_list[first] + (tex_list[second] - tex_list[first]) * alpha;
                        output.push_back(inter_point);
                        output_tex.push_back(inter_tex);
                    }
                }
                else {
                    if (inplane(fv_coo[second], plane, crop_bbox)) {
                        Vector4f inter_point;
                        double alpha;
                        if (!inter_SH(fv_coo[first], fv_coo[second], plane, inter_point, alpha, crop_bbox)) {
                            inter_point = fv_coo[second];
                            alpha = 1;
                        }
                        output.push_back(inter_point);
                        output.push_back(fv_coo[second]);
                        Eigen::Vector2f inter_tex = tex_list[first] + (tex_list[second] - tex_list[first]) * alpha;
                        output_tex.push_back(inter_tex);
                        output_tex.push_back(tex_list[second]);
                    }
                }
            }

            //remove duplicate vertex
            for (int i = 0; i < output.size(); ++i) {
                int cnt = 0;
                if (out.size() == 0) {
                    out.push_back(output[i]);
                    out_tex.push_back(output_tex[i]);
                }
                else {
                    for (int j = 0; j != out.size(); ++j) {
                        if (((output[i]) - (out[j])).norm() < threshold1) {
                            cnt++;
                        }
                    }
                    if (cnt == 0) {
                        out.push_back(output[i]);
                        out_tex.push_back(output_tex[i]);
                    }
                }
            }

            EigenFace_homo ef;
            ef.tex_list = out_tex;
            ef.v_list = out;
            return ef;
        }
        void add_triangle_raster(EigenFace eg, PointCloud<double>& cloud, my_kd_tree_t& kd_index, nanoflann::KNNResultSet<double>& resultSet) {
            assert(eg.v_list.size() == eg.tex_list.size());
            std::vector<int> vid = eg.v_id;
            std::vector<Vector3f> vlist = eg.v_list;
            std::vector<EigenTriMesh::VertexHandle> fh;
            EigenTriMesh::VertexHandle vh1, vh2, vh3;
            std::vector<int> vids;
            fh.clear();
            if (vid[0] != -1) {
                vh1 = add_mesh_point(mesh.vertex_handle(vid[0]), cloud, kd_index);
            }
            //intersection face vertex that are newly added should look up kd2new, if not found then add point and insert to kd2new
            else {
                vh1 = add_point(vlist[0], cloud, kd_index, resultSet);
            }
            if (vid[1] != -1) {
                vh2 = add_mesh_point(mesh.vertex_handle(vid[1]), cloud, kd_index);
            }
            else {
                vh2 = add_point(vlist[1], cloud, kd_index, resultSet);
            }
            if (vid[2] != -1) {
                vh3 = add_mesh_point(mesh.vertex_handle(vid[2]), cloud, kd_index);
            }
            else {
                vh3 = add_point(vlist[2], cloud, kd_index, resultSet);
            }
            if (std::find(vids.begin(), vids.end(), vh1.idx()) == vids.end()) {
                vids.push_back(vh1.idx());
                fh.push_back(vh1);
            }
            if (std::find(vids.begin(), vids.end(), vh2.idx()) == vids.end()) {
                vids.push_back(vh2.idx());
                fh.push_back(vh2);
            }
            if (std::find(vids.begin(), vids.end(), vh3.idx()) == vids.end()) {
                vids.push_back(vh3.idx());
                fh.push_back(vh3);
            }
            if (fh.size() == 3) {
                //mesh_out.add_face(fh);
                //std::cout << vids[0] <<" "<< vids[1] <<" "<< vids[2] << std::endl;
                EigenTriMesh::FaceHandle fh_result = mesh_out.add_face(fh);
                if (fh_result != EigenTriMesh::InvalidFaceHandle) {
                    mesh_out.set_texture_index(fh_result, eg.tex_index);
                    EigenTriMesh::HalfedgeHandle heh = mesh_out.halfedge_handle(fh_result);
                    mesh_out.set_texcoord2D(heh, eg.tex_list[0]);
                    heh = mesh_out.next_halfedge_handle(heh);
                    mesh_out.set_texcoord2D(heh, eg.tex_list[1]);
                    heh = mesh_out.next_halfedge_handle(heh);
                    mesh_out.set_texcoord2D(heh, eg.tex_list[2]);
                    added_faces++;
                    if (std::find(fidxs.begin(), fidxs.end(), fh_result.idx()) == fidxs.end()) {
                        fidxs.push_back(fh_result.idx());
                    }  
                }
                else {
                    invalid_faces++;
                }
            }
        }
        void add_triangle_raster(EigenTriMesh& mesh_crop,EigenFace eg, PointCloud<double>& cloud, my_kd_tree_t& kd_index, nanoflann::KNNResultSet<double>& resultSet,
            std::unordered_map<EigenTriMesh::VertexHandle, EigenTriMesh::VertexHandle>& old2new_crop,
            std::unordered_map<int, EigenTriMesh::VertexHandle>& kd2new_crop, vector<double> crop_bbox) {
            assert(eg.v_list.size() == eg.tex_list.size());
            std::vector<int> vid = eg.v_id;
            std::vector<Vector3f> vlist = eg.v_list;
            std::vector<EigenTriMesh::VertexHandle> fh;
            EigenTriMesh::VertexHandle vh1, vh2, vh3;
            std::vector<int> vids;
            fh.clear();
            if (vid[0] != -1) {
                vh1 = add_mesh_point(mesh_crop,mesh.vertex_handle(vid[0]), cloud, kd_index,old2new_crop,kd2new_crop,crop_bbox);
            }
            //intersection face vertex that are newly added should look up kd2new, if not found then add point and insert to kd2new
            else {
                vh1 = add_point(mesh_crop,vlist[0], cloud, kd_index, resultSet, old2new_crop, kd2new_crop);
            }
            if (vid[1] != -1) {
                vh2 = add_mesh_point(mesh_crop,mesh.vertex_handle(vid[1]), cloud, kd_index, old2new_crop, kd2new_crop, crop_bbox);
            }
            else {
                vh2 = add_point(mesh_crop, vlist[1], cloud, kd_index, resultSet, old2new_crop, kd2new_crop);
            }
            if (vid[2] != -1) {
                vh3 = add_mesh_point(mesh_crop,mesh.vertex_handle(vid[2]), cloud, kd_index, old2new_crop, kd2new_crop, crop_bbox);
            }
            else {
                vh3 = add_point(mesh_crop,vlist[2], cloud, kd_index, resultSet, old2new_crop, kd2new_crop);
            }
            if (std::find(vids.begin(), vids.end(), vh1.idx()) == vids.end()) {
                vids.push_back(vh1.idx());
                fh.push_back(vh1);
            }
            if (std::find(vids.begin(), vids.end(), vh2.idx()) == vids.end()) {
                vids.push_back(vh2.idx());
                fh.push_back(vh2);
            }
            if (std::find(vids.begin(), vids.end(), vh3.idx()) == vids.end()) {
                vids.push_back(vh3.idx());
                fh.push_back(vh3);
            }
            if (fh.size() == 3) {
                //mesh_crop.add_face(fh);
                //std::cout << vids[0] <<" "<< vids[1] <<" "<< vids[2] << std::endl;
                EigenTriMesh::FaceHandle fh_result = mesh_crop.add_face(fh);
                if (fh_result != EigenTriMesh::InvalidFaceHandle) {
                    mesh_crop.set_texture_index(fh_result, eg.tex_index);
                    EigenTriMesh::HalfedgeHandle heh = mesh_crop.halfedge_handle(fh_result);
                    mesh_crop.set_texcoord2D(heh, eg.tex_list[0]);
                    heh = mesh_crop.next_halfedge_handle(heh);
                    mesh_crop.set_texcoord2D(heh, eg.tex_list[1]);
                    heh = mesh_crop.next_halfedge_handle(heh);
                    mesh_crop.set_texcoord2D(heh, eg.tex_list[2]);
                    added_faces++;
                    if (std::find(fidxs.begin(), fidxs.end(), fh_result.idx()) == fidxs.end()) {
                        fidxs.push_back(fh_result.idx());
                    }
                }
                else {
                    invalid_faces++;
                }
            }
        }

        //add polygen(2 or 3 triangles) to new obj
        void add_polygon_raster(EigenFace eg, PointCloud<double>& cloud, my_kd_tree_t& kd_index, nanoflann::KNNResultSet<double>& resultSet) {
            assert(eg.v_list.size() == eg.tex_list.size());
            std::vector<int> vid = eg.v_id;
            std::vector<Vector3f> vlist = eg.v_list;
            if (eg.v_list.size() == 4) {
                EigenFace eg1, eg2;
                eg1.v_list.push_back(eg.v_list[0]);
                eg1.v_list.push_back(eg.v_list[1]);
                eg1.v_list.push_back(eg.v_list[2]);
                eg1.v_id.push_back(eg.v_id[0]);
                eg1.v_id.push_back(eg.v_id[1]);
                eg1.v_id.push_back(eg.v_id[2]);
                eg1.tex_list.push_back(eg.tex_list[0]);
                eg1.tex_list.push_back(eg.tex_list[1]);
                eg1.tex_list.push_back(eg.tex_list[2]);
                eg1.tex_index = eg.tex_index;
                eg2.v_list.push_back(eg.v_list[0]);
                eg2.v_list.push_back(eg.v_list[2]);
                eg2.v_list.push_back(eg.v_list[3]);
                eg2.v_id.push_back(eg.v_id[0]);
                eg2.v_id.push_back(eg.v_id[2]);
                eg2.v_id.push_back(eg.v_id[3]);
                eg2.tex_list.push_back(eg.tex_list[0]);
                eg2.tex_list.push_back(eg.tex_list[2]);
                eg2.tex_list.push_back(eg.tex_list[3]);
                eg2.tex_index = eg.tex_index;
                add_triangle_raster(eg1, cloud, kd_index, resultSet);
                add_triangle_raster(eg2, cloud, kd_index, resultSet);
                std::vector<EigenTriMesh::VertexHandle> fh1, fh2;
            }
            else {
                assert(eg.v_list.size() == 5);
                EigenFace eg1, eg2, eg3;
                eg1.v_list.push_back(eg.v_list[0]);
                eg1.v_list.push_back(eg.v_list[1]);
                eg1.v_list.push_back(eg.v_list[2]);
                eg1.v_id.push_back(eg.v_id[0]);
                eg1.v_id.push_back(eg.v_id[1]);
                eg1.v_id.push_back(eg.v_id[2]);
                eg1.tex_list.push_back(eg.tex_list[0]);
                eg1.tex_list.push_back(eg.tex_list[1]);
                eg1.tex_list.push_back(eg.tex_list[2]);
                eg1.tex_index = eg.tex_index;
                eg2.v_list.push_back(eg.v_list[0]);
                eg2.v_list.push_back(eg.v_list[2]);
                eg2.v_list.push_back(eg.v_list[3]);
                eg2.v_id.push_back(eg.v_id[0]);
                eg2.v_id.push_back(eg.v_id[2]);
                eg2.v_id.push_back(eg.v_id[3]);
                eg2.tex_list.push_back(eg.tex_list[0]);
                eg2.tex_list.push_back(eg.tex_list[2]);
                eg2.tex_list.push_back(eg.tex_list[3]);
                eg2.tex_index = eg.tex_index;
                eg3.v_list.push_back(eg.v_list[0]);
                eg3.v_list.push_back(eg.v_list[3]);
                eg3.v_list.push_back(eg.v_list[4]);
                eg3.v_id.push_back(eg.v_id[0]);
                eg3.v_id.push_back(eg.v_id[3]);
                eg3.v_id.push_back(eg.v_id[4]);
                eg3.tex_list.push_back(eg.tex_list[0]);
                eg3.tex_list.push_back(eg.tex_list[3]);
                eg3.tex_list.push_back(eg.tex_list[4]);
                eg3.tex_index = eg.tex_index;
                add_triangle_raster(eg1, cloud, kd_index, resultSet);
                add_triangle_raster(eg2, cloud, kd_index, resultSet);
                add_triangle_raster(eg3, cloud, kd_index, resultSet);
            }
        }
        void add_polygon_raster(EigenTriMesh& mesh_crop, EigenFace eg, PointCloud<double>& cloud, my_kd_tree_t& kd_index, nanoflann::KNNResultSet<double>& resultSet,
            std::unordered_map<EigenTriMesh::VertexHandle, EigenTriMesh::VertexHandle>& old2new_crop,
            std::unordered_map<int, EigenTriMesh::VertexHandle>& kd2new_crop, vector<double> crop_bbox) {

            assert(eg.v_list.size() == eg.tex_list.size());
            std::vector<int> vid = eg.v_id;
            std::vector<Vector3f> vlist = eg.v_list;
            if (eg.v_list.size() == 4) {
                EigenFace eg1, eg2;
                eg1.v_list.push_back(eg.v_list[0]);
                eg1.v_list.push_back(eg.v_list[1]);
                eg1.v_list.push_back(eg.v_list[2]);
                eg1.v_id.push_back(eg.v_id[0]);
                eg1.v_id.push_back(eg.v_id[1]);
                eg1.v_id.push_back(eg.v_id[2]);
                eg1.tex_list.push_back(eg.tex_list[0]);
                eg1.tex_list.push_back(eg.tex_list[1]);
                eg1.tex_list.push_back(eg.tex_list[2]);
                eg1.tex_index = eg.tex_index;
                eg2.v_list.push_back(eg.v_list[0]);
                eg2.v_list.push_back(eg.v_list[2]);
                eg2.v_list.push_back(eg.v_list[3]);
                eg2.v_id.push_back(eg.v_id[0]);
                eg2.v_id.push_back(eg.v_id[2]);
                eg2.v_id.push_back(eg.v_id[3]);
                eg2.tex_list.push_back(eg.tex_list[0]);
                eg2.tex_list.push_back(eg.tex_list[2]);
                eg2.tex_list.push_back(eg.tex_list[3]);
                eg2.tex_index = eg.tex_index;
                add_triangle_raster(mesh_crop,eg1, cloud, kd_index, resultSet,old2new_crop,kd2new_crop,crop_bbox);
                add_triangle_raster(mesh_crop,eg2, cloud, kd_index, resultSet,old2new_crop,kd2new_crop,crop_bbox);
                std::vector<EigenTriMesh::VertexHandle> fh1, fh2;
            }
            else if(eg.v_list.size() == 5) {
                EigenFace eg1, eg2, eg3;
                eg1.v_list.push_back(eg.v_list[0]);
                eg1.v_list.push_back(eg.v_list[1]);
                eg1.v_list.push_back(eg.v_list[2]);
                eg1.v_id.push_back(eg.v_id[0]);
                eg1.v_id.push_back(eg.v_id[1]);
                eg1.v_id.push_back(eg.v_id[2]);
                eg1.tex_list.push_back(eg.tex_list[0]);
                eg1.tex_list.push_back(eg.tex_list[1]);
                eg1.tex_list.push_back(eg.tex_list[2]);
                eg1.tex_index = eg.tex_index;
                eg2.v_list.push_back(eg.v_list[0]);
                eg2.v_list.push_back(eg.v_list[2]);
                eg2.v_list.push_back(eg.v_list[3]);
                eg2.v_id.push_back(eg.v_id[0]);
                eg2.v_id.push_back(eg.v_id[2]);
                eg2.v_id.push_back(eg.v_id[3]);
                eg2.tex_list.push_back(eg.tex_list[0]);
                eg2.tex_list.push_back(eg.tex_list[2]);
                eg2.tex_list.push_back(eg.tex_list[3]);
                eg2.tex_index = eg.tex_index;
                eg3.v_list.push_back(eg.v_list[0]);
                eg3.v_list.push_back(eg.v_list[3]);
                eg3.v_list.push_back(eg.v_list[4]);
                eg3.v_id.push_back(eg.v_id[0]);
                eg3.v_id.push_back(eg.v_id[3]);
                eg3.v_id.push_back(eg.v_id[4]);
                eg3.tex_list.push_back(eg.tex_list[0]);
                eg3.tex_list.push_back(eg.tex_list[3]);
                eg3.tex_list.push_back(eg.tex_list[4]);
                eg3.tex_index = eg.tex_index;
                add_triangle_raster(mesh_crop, eg1, cloud, kd_index, resultSet, old2new_crop, kd2new_crop, crop_bbox);
                add_triangle_raster(mesh_crop, eg2, cloud, kd_index, resultSet, old2new_crop, kd2new_crop, crop_bbox);
                add_triangle_raster(mesh_crop, eg3, cloud, kd_index, resultSet, old2new_crop, kd2new_crop, crop_bbox);
            }
            else if (eg.v_list.size() == 6) {
                EigenFace eg1, eg2, eg3,eg4;
                eg1.v_list.push_back(eg.v_list[0]);
                eg1.v_list.push_back(eg.v_list[1]);
                eg1.v_list.push_back(eg.v_list[2]);
                eg1.v_id.push_back(eg.v_id[0]);
                eg1.v_id.push_back(eg.v_id[1]);
                eg1.v_id.push_back(eg.v_id[2]);
                eg1.tex_list.push_back(eg.tex_list[0]);
                eg1.tex_list.push_back(eg.tex_list[1]);
                eg1.tex_list.push_back(eg.tex_list[2]);
                eg1.tex_index = eg.tex_index;
                eg2.v_list.push_back(eg.v_list[0]);
                eg2.v_list.push_back(eg.v_list[2]);
                eg2.v_list.push_back(eg.v_list[3]);
                eg2.v_id.push_back(eg.v_id[0]);
                eg2.v_id.push_back(eg.v_id[2]);
                eg2.v_id.push_back(eg.v_id[3]);
                eg2.tex_list.push_back(eg.tex_list[0]);
                eg2.tex_list.push_back(eg.tex_list[2]);
                eg2.tex_list.push_back(eg.tex_list[3]);
                eg2.tex_index = eg.tex_index;
                eg3.v_list.push_back(eg.v_list[0]);
                eg3.v_list.push_back(eg.v_list[3]);
                eg3.v_list.push_back(eg.v_list[4]);
                eg3.v_id.push_back(eg.v_id[0]);
                eg3.v_id.push_back(eg.v_id[3]);
                eg3.v_id.push_back(eg.v_id[4]);
                eg3.tex_list.push_back(eg.tex_list[0]);
                eg3.tex_list.push_back(eg.tex_list[3]);
                eg3.tex_list.push_back(eg.tex_list[4]);
                eg3.tex_index = eg.tex_index;
                eg4.v_list.push_back(eg.v_list[0]);
                eg4.v_list.push_back(eg.v_list[4]);
                eg4.v_list.push_back(eg.v_list[5]);
                eg4.v_id.push_back(eg.v_id[0]);
                eg4.v_id.push_back(eg.v_id[4]);
                eg4.v_id.push_back(eg.v_id[5]);
                eg4.tex_list.push_back(eg.tex_list[0]);
                eg4.tex_list.push_back(eg.tex_list[4]);
                eg4.tex_list.push_back(eg.tex_list[5]);
                eg4.tex_index = eg.tex_index;
                add_triangle_raster(mesh_crop, eg1, cloud, kd_index, resultSet, old2new_crop, kd2new_crop, crop_bbox);
                add_triangle_raster(mesh_crop, eg2, cloud, kd_index, resultSet, old2new_crop, kd2new_crop, crop_bbox);
                add_triangle_raster(mesh_crop, eg3, cloud, kd_index, resultSet, old2new_crop, kd2new_crop, crop_bbox);
                add_triangle_raster(mesh_crop, eg4, cloud, kd_index, resultSet, old2new_crop, kd2new_crop, crop_bbox);
            }
        }

        bool isinbbox(Eigen::Vector4f pt) {
            if (pt[0] >= boundingbox[0] && pt[0] <= boundingbox[1] && pt[1] >= boundingbox[2] && pt[1] <= boundingbox[3]) {
                return true;
            }
            else {
                return false;
            }
        }
        bool isinbbox(Eigen::Vector4f pt , vector<double> crop_bbox) {
            if (pt[0] >= crop_bbox[0] && pt[0] <= crop_bbox[1] && pt[1] >= crop_bbox[2] && pt[1] <= crop_bbox[3]) {
                return true;
            }
            else {
                return false;
            }
        }
        bool onbbox(Eigen::Vector3f pt, vector<double> crop_bbox) {
            if (abs(pt[0] - crop_bbox[0]) < threshold1
                || abs(pt[0] - crop_bbox[1]) < threshold1
                || abs(pt[1] - crop_bbox[2]) < threshold1
                || abs(pt[1] - crop_bbox[3]) < threshold1) {
                return true;
            }
            else {
                return false;
            }
        }
        bool onbbox(Eigen::Vector3f pt) {
            if (abs(pt[0] - boundingbox[0]) < threshold1
                || abs(pt[0] - boundingbox[1]) < threshold1
                || abs(pt[1] - boundingbox[2]) < threshold1
                || abs(pt[1] - boundingbox[3]) < threshold1) {
                return true;
            }
            else {
                return false;
            }
        }
        ////////////////////////////////////////////////RASTER SECTION END////////////////////////////////////////

        // tell if a vertex is out or in a plane.
        bool outplane(const Ref<Vector4f>& point, char plane) {
            switch (plane) {
            case 'l':
                if (point[0] <= boundingbox[0]) {
                    return true;
                }
                else {
                    return false;
                }
            case 'r':
                if (point[0] >= boundingbox[1]) {
                    return true;
                }
                else {
                    return false;
                }
            case 'b':
                if (point[1] <= boundingbox[2]) {
                    return true;
                }
                else {
                    return false;
                }
            case 't':
                if (point[1] >= boundingbox[3]) {
                    return true;
                }
                else {
                    return false;
                }
            }
        }
        bool outplane(const Ref<Vector4f>& point, char plane, vector<double> crop_bbox) {
            switch (plane) {
            case 'l':
                if (point[0] <= crop_bbox[0]) {
                    return true;
                }
                else {
                    return false;
                }
            case 'r':
                if (point[0] >= crop_bbox[1]) {
                    return true;
                }
                else {
                    return false;
                }
            case 'b':
                if (point[1] <= crop_bbox[2]) {
                    return true;
                }
                else {
                    return false;
                }
            case 't':
                if (point[1] >= crop_bbox[3]) {
                    return true;
                }
                else {
                    return false;
                }
            }
        }

        
        bool inplane(const Ref<Vector4f>& point, char plane) {
            switch (plane) {
            case 'l':
                if (point[0] >= boundingbox[0]) {
                    return true;
                }
                else {
                    return false;
                }
            case 'r':
                if (point[0] <= boundingbox[1]) {
                    return true;
                }
                else {
                    return false;
                }
            case 'b':
                if (point[1] >= boundingbox[2]) {
                    return true;
                }
                else {
                    return false;
                }
            case 't':
                if (point[1] <= boundingbox[3]) {
                    return true;
                }
                else {
                    return false;
                }
            }
        }

        bool inplane(const Ref<Vector4f>& point, char plane, vector<double> crop_bbox) {
            switch (plane) {
            case 'l':
                if (point[0] >= crop_bbox[0]) {
                    return true;
                }
                else {
                    return false;
                }
            case 'r':
                if (point[0] <= crop_bbox[1]) {
                    return true;
                }
                else {
                    return false;
                }
            case 'b':
                if (point[1] >= crop_bbox[2]) {
                    return true;
                }
                else {
                    return false;
                }
            case 't':
                if (point[1] <= crop_bbox[3]) {
                    return true;
                }
                else {
                    return false;
                }
            }
        }

        void check_vt(EigenFace& eg, EigenFace& eg_out) {
            eg_out.v_list = eg.v_list;
            for (auto i = eg.tex_list.begin(); i != eg.tex_list.end(); ++i) {
                //nan check
                if (isnan((*i)[0]) || (*i)[0] > 1 || (*i)[0] < 0) {
                    (*i)[0] = 0.0;
                }
                if (isnan((*i)[1]) || (*i)[1] > 1 || (*i)[1] < 0) {
                    (*i)[1] = 0.0;
                }
            }
            eg_out.tex_list = eg.tex_list;
            eg_out.v_id = eg.v_id;
            eg_out.tex_index = eg.tex_index;
        }

        EigenTriMesh::VertexHandle add_mesh_point(EigenTriMesh::VertexHandle vh_in, PointCloud<double>& cloud, my_kd_tree_t& kd_index) {
            if (old2new.find(vh_in) != old2new.end()) {
                return old2new[vh_in];
            }
            else {
                Eigen::Vector3f point;
                EigenTriMesh::VertexHandle vh;
                point = mesh.point(vh_in);
                vh = mesh_out.add_vertex(point);
                old2new.insert(std::make_pair(vh_in, vh));
                //if point on boundary, add it to kdtree
                if (onbbox(point)) {
                    PointCloud<double>::Point p;
                    p.x = point[0];
                    p.y = point[1];
                    p.z = point[2];
                    cloud.pts.push_back(p);
                    kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);
                    kd2new.insert(std::make_pair(cloud.pts.size() - 1, vh));
                }
                return vh;
            }
        }

        EigenTriMesh::VertexHandle add_mesh_point(EigenTriMesh& mesh_crop, EigenTriMesh::VertexHandle vh_in, PointCloud<double>& cloud, my_kd_tree_t& kd_index, 
            std::unordered_map<EigenTriMesh::VertexHandle, EigenTriMesh::VertexHandle>& old2new_crop,
            std::unordered_map<int, EigenTriMesh::VertexHandle>& kd2new_crop, vector<double> crop_bbox) {
            if (old2new_crop.find(vh_in) != old2new_crop.end()) {
                return old2new_crop[vh_in];
            }
            else {
                Eigen::Vector3f point;
                EigenTriMesh::VertexHandle vh;
                point = mesh.point(vh_in);
                vh = mesh_crop.add_vertex(point);
                old2new_crop.insert(std::make_pair(vh_in, vh));
                //if point on boundary, add it to kdtree
                if (onbbox(point, crop_bbox)) {
                    PointCloud<double>::Point p;
                    p.x = point[0];
                    p.y = point[1];
                    p.z = point[2];
                    cloud.pts.push_back(p);
                    kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);
                    kd2new_crop.insert(std::make_pair(cloud.pts.size() - 1, vh));
                }
                return vh;
            }
        }


        //if point is already in mesh, return its vertex_handle
        //else add point, and return its vertex_handle
        EigenTriMesh::VertexHandle add_point(Vector3f& point, PointCloud<double>& cloud, my_kd_tree_t& kd_index, nanoflann::KNNResultSet<double>& resultSet) {
            EigenTriMesh::VertexHandle vh;
            EigenTriMesh::VertexIter v_it, v_end(mesh_out.vertices_end());

            double p_tmp[3] = { point[0],point[1],point[2] };
            resultSet.init(&ret_index, &out_dist_sqr);
            kd_index.findNeighbors(resultSet, p_tmp, nanoflann::SearchParams(10));
            if (abs(sqrt(out_dist_sqr)) > threshold1) {
                PointCloud<double>::Point p;
                p.x = point[0];
                p.y = point[1];
                p.z = point[2];
                cloud.pts.push_back(p);
                kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);

                vh = mesh_out.add_vertex(point);
                kd2new.insert(std::make_pair(cloud.pts.size() - 1, vh));
            }
            else {
                vh = kd2new[ret_index];
            }

            /*int cnt = 0;
            for (v_it = mesh_out.vertices_begin(); v_it != v_end; ++v_it) {
                if (point.isApprox(mesh_out.point(*v_it))) {
                    cnt++;
                    vh = *v_it;
                    break;
                }
            }
            if (cnt == 0) {
                vh = mesh_out.add_vertex(point);
                assert(vh.is_valid());

            }*/
            return vh;
        }
        EigenTriMesh::VertexHandle add_point(EigenTriMesh& mesh_crop, Vector3f& point, PointCloud<double>& cloud, my_kd_tree_t& kd_index, nanoflann::KNNResultSet<double>& resultSet,
            std::unordered_map<EigenTriMesh::VertexHandle, EigenTriMesh::VertexHandle>& old2new_crop,
            std::unordered_map<int, EigenTriMesh::VertexHandle>& kd2new_crop) {


            EigenTriMesh::VertexHandle vh;
            EigenTriMesh::VertexIter v_it, v_end(mesh_crop.vertices_end());
            double p_tmp[3] = { point[0],point[1],point[2] };
            resultSet.init(&ret_index, &out_dist_sqr);
            kd_index.findNeighbors(resultSet, p_tmp, nanoflann::SearchParams(10));
            if (abs(sqrt(out_dist_sqr)) > threshold1) {
                PointCloud<double>::Point p;
                p.x = point[0];
                p.y = point[1];
                p.z = point[2];
                cloud.pts.push_back(p);
                kd_index.addPoints(cloud.pts.size() - 1, cloud.pts.size() - 1);

                vh = mesh_crop.add_vertex(point);
                kd2new_crop.insert(std::make_pair(cloud.pts.size() - 1, vh));
            }
            else {
                vh = kd2new_crop[ret_index];
            }

            return vh;
        }


        bool inter_SH(Eigen::Vector4f& p1, const Eigen::Vector4f& p2, char plane, Eigen::Vector4f& inter, double& d) {
            Eigen::Vector3f plane_normal, plane_pts;
            switch (plane) {
            case 'l':
                plane_normal << 1, 0, 0;
                plane_pts << boundingbox[0], boundingbox[2], 0;
                break;
            case 'r':
                plane_normal << 1, 0, 0;
                plane_pts << boundingbox[1], boundingbox[2], 0;
                break;
            case 'b':
                plane_normal << 0, 1, 0;
                plane_pts << boundingbox[0], boundingbox[2], 0;
                break;
            case 't':
                plane_normal << 0, 1, 0;
                plane_pts << boundingbox[0], boundingbox[3], 0;
                break;
            }
            Vector3f p1_3, p2_3;
            p1_3 << p1[0], p1[1], p1[2];
            p2_3 << p2[0], p2[1], p2[2];
            //parallel check
            if (plane_normal.dot(p2_3 - p1_3) == 0) {
                std::cout << "parallel" << std::endl;
                inter_error++;
                return false;
            }
            else {
                d = (plane_pts - p1_3).dot(plane_normal) / (plane_normal.dot(p2_3 - p1_3));
                if (d >= 0 && d <= 1) {
                    inter << p1_3 + d * (p2_3 - p1_3), 1;
                    return true;
                }
                else {
                    std::cout << "inter point not in line seg" << std::endl;
                    inter_error++;
                    return false;
                }
            }
        }

        bool inter_SH(Eigen::Vector4f& p1, const Eigen::Vector4f& p2, char plane, Eigen::Vector4f& inter, double& d, vector<double> crop_bbox) {
            Eigen::Vector3f plane_normal, plane_pts;
            switch (plane) {
            case 'l':
                plane_normal << 1, 0, 0;
                plane_pts << crop_bbox[0], crop_bbox[2], 0;
                break;
            case 'r':
                plane_normal << 1, 0, 0;
                plane_pts << crop_bbox[1], crop_bbox[2], 0;
                break;
            case 'b':
                plane_normal << 0, 1, 0;
                plane_pts << crop_bbox[0], crop_bbox[2], 0;
                break;
            case 't':
                plane_normal << 0, 1, 0;
                plane_pts << crop_bbox[0], crop_bbox[3], 0;
                break;
            }
            Vector3f p1_3, p2_3;
            p1_3 << p1[0], p1[1], p1[2];
            p2_3 << p2[0], p2[1], p2[2];
            //parallel check
            if (plane_normal.dot(p2_3 - p1_3) == 0) {
                std::cout << "parallel" << std::endl;
                inter_error++;
                return false;
            }
            else {
                d = (plane_pts - p1_3).dot(plane_normal) / (plane_normal.dot(p2_3 - p1_3));
                if (d >= 0 && d <= 1) {
                    inter << p1_3 + d * (p2_3 - p1_3), 1;
                    return true;
                }
                else {
                    std::cout << "inter point not in line seg" << std::endl;
                    inter_error++;
                    return false;
                }
            }
        }

        void writeobj_openmesh(char* path) {

            // write mesh to output.obj
            try
            {
                OpenMesh::IO::Options opt(OpenMesh::IO::Options::FaceTexCoord);
                if (!OpenMesh::IO::write_mesh(mesh_out, path, opt))
                {
                    std::cerr << "Cannot write mesh to file 'man.obj'" << std::endl;
                }
            }
            catch (std::exception& x)
            {
                std::cerr << x.what() << std::endl;
            }
        }

        void write_manual(std::string path) {
            std::string mtl_name= fs::v1::path(path).filename().replace_extension(".mtl").string();
            std::ofstream ofs(path);
            Vector3f pts;
            Vector2f tex;
            EigenTriMesh::HalfedgeHandle heh;
            boost::format fmt;
            ofs<< "mtllib " << mtl_name << "\n";
            ofs << std::fixed << std::setprecision(12);
            //wrtie v
            for (int i = 0; i < mesh_out.n_vertices(); ++i) {
                pts = mesh_out.point(mesh_out.vertex_handle(i));
                ofs << "v " << pts[0] << " " << pts[1] << " " << pts[2] << "\n";
                //fmt = boost::format("v %s %s %s\n") % pts[0] % pts[1] % pts[2];
                //ofs << fmt.str();
            }
            //wrtie vt
            for (int i = 0; i < mesh_out.n_faces(); ++i) {
                heh = mesh_out.halfedge_handle(mesh_out.face_handle(i));
                tex = mesh_out.texcoord2D(heh);
                ofs << "vt " << tex[0] << " " << tex[1] << "\n";
                //fmt = boost::format("vt %s %s\n") % tex[0] % tex[1];
                //ofs << fmt.str();

                heh = mesh_out.next_halfedge_handle(heh);
                tex = mesh_out.texcoord2D(heh);
                ofs << "vt " << tex[0] << " " << tex[1] << "\n";
                //fmt = boost::format("vt %s %s\n") % tex[0] % tex[1];
                //ofs << fmt.str();

                heh = mesh_out.next_halfedge_handle(heh);
                tex = mesh_out.texcoord2D(heh);
                ofs << "vt " << tex[0] << " " << tex[1] << "\n";
                //fmt = boost::format("vt %s %s\n") % tex[0] % tex[1];
                //ofs << fmt.str();
            }
            //wrtie face
            bool useMatrial = false;
            OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out;
            if (mesh_out.get_property_handle(texindex_out, "TextureMapping")) {
                useMatrial = true;
            }

            int uv = 1, v1, v2, v3;
            int current_tex_index = 1;
            int tmp_tex_index = 0;
            ofs << "usemtl mat" << current_tex_index << '\n';
            for (int i = 0; i < mesh_out.n_faces(); ++i) {
                if (useMatrial) {
                    tmp_tex_index = mesh_out.texture_index(mesh_out.face_handle(i));
                    if (current_tex_index != tmp_tex_index) {
                        current_tex_index = tmp_tex_index;
                        ofs << "usemtl mat" << current_tex_index << '\n';
                    }
                }
                auto fv_it = mesh_out.fv_begin(mesh_out.face_handle(i));
                v1 = (*fv_it).idx();
                ++fv_it;
                v2 = (*fv_it).idx();
                ++fv_it;
                v3 = (*fv_it).idx();
                fmt = boost::format("f %s/%s %s/%s %s/%s\n") % (v1 + 1) % uv % (v2 + 1) % (uv + 1) % (v3 + 1) % (uv + 2);
                ofs << fmt.str();
                uv += 3;
            }
            ofs.close();

        }
        
        void write_manual(EigenTriMesh& mesh,std::string in_path ,std::string path,bool copy_texture) {
            std::string mtl_name = fs::v1::path(path).filename().replace_extension(".mtl").string();
            std::ofstream ofs(path);
            Vector3f pts;
            Vector2f tex;
            EigenTriMesh::HalfedgeHandle heh;
            boost::format fmt;
            ofs << "mtllib " << mtl_name << "\n";
            ofs << std::fixed << std::setprecision(12);
            //wrtie v
            for (int i = 0; i < mesh.n_vertices(); ++i) {
                pts = mesh.point(mesh.vertex_handle(i));
                fmt = boost::format("v %s %s %s\n") % pts[0] % pts[1] % pts[2];
                ofs << "v " << pts[0] << " " << pts[1] << " " << pts[2] << "\n";
                //ofs << fmt.str();
            }
            //wrtie vt
            for (int i = 0; i < mesh.n_faces(); ++i) {
                heh = mesh.halfedge_handle(mesh.face_handle(i));
                tex = mesh.texcoord2D(heh);
                ofs << "vt " << tex[0] << " " << tex[1]<<"\n";
                //fmt = boost::format("vt %s %s\n") % tex[0] % tex[1];
                //ofs << fmt.str();

                heh = mesh.next_halfedge_handle(heh);
                tex = mesh.texcoord2D(heh);
                ofs << "vt " << tex[0] << " " << tex[1] << "\n";
                //fmt = boost::format("vt %s %s\n") % tex[0] % tex[1];
                //ofs << fmt.str();

                heh = mesh.next_halfedge_handle(heh);
                tex = mesh.texcoord2D(heh);
                ofs << "vt " << tex[0] << " " << tex[1] << "\n";
                //fmt = boost::format("vt %s %s\n") % tex[0] % tex[1];
                //ofs << fmt.str();
            }
            //wrtie face
            bool useMatrial = false;
            OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out;
            std::vector<int> crop_texindex_ids;
            if (mesh.get_property_handle(texindex_out, "TextureMapping")) {
                useMatrial = true;
            }

            int uv = 1, v1, v2, v3;
            int current_tex_index = 1;
            int tmp_tex_index = 1;
            for (int i = 0; i < mesh.n_faces(); ++i) {
                if (useMatrial) {
                    if (i == 0) {
                        current_tex_index= mesh.texture_index(mesh.face_handle(i));
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
                fmt = boost::format("f %s/%s %s/%s %s/%s\n") % (v1 + 1) % uv % (v2 + 1) % (uv + 1) % (v3 + 1) % (uv + 2);
                ofs << fmt.str();
                uv += 3;
            }
            ofs.close();

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
                if (useMatrial) {
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

                }
                else {
                    while (std::getline(mtl_in, line))
                    {
                        if (line.find("newmtl") != std::string::npos) {
                            mtl_out << "newmtl " + out_filename.substr(0, out_filename.find(".obj")) << "\n";
                        }
                        else if (line.find("map_K") != std::string::npos) {
                            if (line.find(" ") != std::string::npos) {
                                texture_name.push_back(line.substr(line.find(" ") + 1));
                                mtl_out << line << "\n";
                            }
                            else {
                                mtl_out << line << "\n";
                            }

                        }
                        else {
                            mtl_out << line << "\n";
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
                    //else {
                    //    std::cout << "already copied or texure image doesn't exist" << std::endl;
                    //}
                }
            }

        }



        void write_manual_with_tex(std::string path, Vector4f& tex_box) {
            std::ofstream ofs(path);
            std::string mtl_name = fs::v1::path(path).filename().replace_extension(".mtl").string();
            Vector3f pts;
            Vector2f tex;
            EigenTriMesh::HalfedgeHandle heh;
            boost::format fmt;
            double tex_box_width = abs(tex_box[1] - tex_box[0]);
            double tex_box_height = abs(tex_box[2] - tex_box[3]);
            ofs << "mtllib " << mtl_name << "\n";
            ofs << std::fixed << std::setprecision(12);
            //wrtie v
            for (int i = 0; i < mesh_out.n_vertices(); ++i) {
                pts = mesh_out.point(mesh_out.vertex_handle(i));
                ofs << "v " << pts[0] << " " << pts[1] << " " << pts[2] << "\n";
                //fmt = boost::format("v %s %s %s\n") % pts[0] % pts[1] % pts[2];
                //ofs << fmt.str();
                ofs << "vt " << ((pts[0] - tex_box[0]) / tex_box_width) << " " << ((pts[1] - tex_box[2]) / tex_box_height) << "\n";
                //fmt = boost::format("vt %s %s\n") % ((pts[0]-tex_box[0])/tex_box_width) % ((pts[1]-tex_box[2])/tex_box_height);
                //ofs << fmt.str();

            }
            //wrtie face
            int uv = 1, v1, v2, v3;
            int current_tex_index = 1;
            int tmp_tex_index = 0;
            ofs << "usemtl mat" << current_tex_index << '\n';
            for (int i = 0; i < mesh_out.n_faces(); ++i) {
                auto fv_it = mesh_out.fv_begin(mesh_out.face_handle(i));
                v1 = (*fv_it).idx();
                ++fv_it;
                v2 = (*fv_it).idx();
                ++fv_it;
                v3 = (*fv_it).idx();
                fmt = boost::format("f %s/%s %s/%s %s/%s\n") % (v1 + 1) % (v1 + 1) % (v2 + 1) % (v2 + 1) % (v3 + 1) % (v3 + 1);
                ofs << fmt.str();
                uv += 3;
            }
            ofs.close();
        }

        void copy_mtl_tex(std::string input, std::string output) {
            std::string out_filename = fs::v1::path(output).filename().string();

            fs::v1::path mtl_in_path = fs::v1::path(input).replace_extension(".mtl");
            fs::v1::path mtl_out_path = fs::v1::path(output).replace_extension(".mtl");
            std::ifstream mtl_in(mtl_in_path);
            std::ofstream mtl_out(mtl_out_path);

            std::vector<std::string> texture_name;
            std::string line;
            //copy mtl
            
            //write mtl file
            bool useMatrial = false;
            OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out;
            if (mesh_out.get_property_handle(texindex_out, "TextureMapping")) {
                useMatrial = true;
            }

            if (mtl_in.is_open())
            {
                if (useMatrial) {
                    for (size_t i = 0; i < mesh_out.property(texindex_out).size(); i++) {
                        if (!mesh_out.property(texindex_out)[i].empty()) {
                            mtl_out << "newmtl " << "mat" << i << '\n';
                            mtl_out << "Ka 1.000 1.000 1.000" << '\n';
                            mtl_out << "Kd 1.000 1.000 1.000" << '\n';
                            mtl_out << "illum 1" << '\n';
                            mtl_out << "map_Kd " << mesh_out.property(texindex_out)[i] << "\n";
                            texture_name.push_back(mesh_out.property(texindex_out)[i]);
                        }
                        
                    }

                }
                else {
                    while (std::getline(mtl_in, line))
                    {
                        if (line.find("newmtl") != std::string::npos) {
                            mtl_out << "newmtl " + out_filename.substr(0, out_filename.find(".obj")) << "\n";
                        }
                        else if (line.find("map_K") != std::string::npos) {
                            if (line.find(" ") != std::string::npos) {
                                texture_name.push_back(line.substr(line.find(" ") + 1));
                                mtl_out << line << "\n";
                            }
                            else {
                                mtl_out << line << "\n";
                            }

                        }
                        else {
                            mtl_out << line << "\n";
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
            for (auto it=texture_name.begin();it!=texture_name.end();++it){

                fs::v1::path tex_out_path = fs::v1::path(mtl_out_path).replace_filename(*it);
                fs::v1::path tex_in_path = fs::v1::path(mtl_in_path).replace_filename(*it);
                if (!fs::v1::exists(tex_out_path) && fs::v1::exists(tex_in_path)) {
                    fs::copy_file(tex_in_path, tex_out_path);
                }
                else {
                    std::cout << "already copied or texure image doesn't exist" << std::endl;
                }
            }

            ////add mtl name to the beginning of obj file
            //fs::v1::path output_tmp = fs::v1::path(output).replace_filename("tmp.obj");
            //std::ofstream outputFile(output_tmp);
            //std::ifstream inputFile(output);
            //std::string tempString;

            //outputFile << "mtllib " << fs::v1::path(mtl_out_path).filename().string() << "\n";
            //outputFile << inputFile.rdbuf();

            //inputFile.close();
            //outputFile.close();

            //std::remove(output.c_str());
            //std::rename(output_tmp.string().c_str(), output.c_str());

        }

        void copy_mtl_tex(EigenTriMesh& mesh, std::string input, std::string output, bool copy_texture) {
            std::string out_filename = fs::v1::path(output).filename().string();

            fs::v1::path mtl_in_path = fs::v1::path(input).replace_extension(".mtl");
            fs::v1::path mtl_out_path = fs::v1::path(output).replace_extension(".mtl");
            std::ifstream mtl_in(mtl_in_path);
            std::ofstream mtl_out(mtl_out_path);

            std::vector<std::string> texture_name;
            std::string line;
            //copy mtl

            //write mtl file
            bool useMatrial = false;
            OpenMesh::MPropHandleT< std::map< int, std::string > > texindex_out;
            if (mesh.get_property_handle(texindex_out, "TextureMapping")) {
                useMatrial = true;
            }

            if (mtl_in.is_open())
            {
                if (useMatrial) {
                    for (size_t i = 0; i < mesh.property(texindex_out).size(); i++) {
                        if (!mesh.property(texindex_out)[i].empty()) {
                            mtl_out << "newmtl " << "mat" << i << '\n';
                            mtl_out << "Ka 1.000 1.000 1.000" << '\n';
                            mtl_out << "Kd 1.000 1.000 1.000" << '\n';
                            mtl_out << "illum 1" << '\n';
                            mtl_out << "map_Kd " << mesh.property(texindex_out)[i] << "\n";
                            texture_name.push_back(mesh.property(texindex_out)[i]);
                        }

                    }

                }
                else {
                    while (std::getline(mtl_in, line))
                    {
                        if (line.find("newmtl") != std::string::npos) {
                            mtl_out << "newmtl " + out_filename.substr(0, out_filename.find(".obj")) << "\n";
                        }
                        else if (line.find("map_K") != std::string::npos) {
                            if (line.find(" ") != std::string::npos) {
                                texture_name.push_back(line.substr(line.find(" ") + 1));
                                mtl_out << line << "\n";
                            }
                            else {
                                mtl_out << line << "\n";
                            }

                        }
                        else {
                            mtl_out << line << "\n";
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
                    else {
                        std::cout << "already copied or texure image doesn't exist" << std::endl;
                    }
                }
            }
        }


        void write_mtl( std::string output, std::string texname) {
            fs::v1::path mtl_out_path = fs::v1::path(output).replace_extension(".mtl");
            std::ofstream mtl_out(mtl_out_path);
            mtl_out << "newmtl " << "mat1" << '\n';
            mtl_out << "Ka 1.000 1.000 1.000" << '\n';
            mtl_out << "Kd 1.000 1.000 1.000" << '\n';
            mtl_out << "illum 1" << '\n';
            mtl_out << "map_Kd " << texname << "\n";
            mtl_out.close();         
        }
        //change vt=nan or not in [0,1] to 0.0f
        void check_write_file(std::string input) {
            std::ifstream ifs(input);
            fs::v1::path output_tmp = fs::v1::path(input).replace_filename("tmp.obj");
            std::ofstream ofs(output_tmp);
            std::string line;
            while (std::getline(ifs, line)) {
                if (line[0] == 'v' && line[1] == 't' && line[2] == ' ')
                {
                    double vtx, vty;
                    if (std::sscanf(line.c_str(), "vt %f %f", &vtx, &vty) == 2)
                    {
                        if (isnan(vtx) || vtx > 1 || vtx < 0) {
                            vtx = 0.0f;
                        }
                        if (isnan(vty) || vty > 1 || vty < 0) {
                            vty = 0.0f;
                        }
                        ofs << "vt " + std::to_string(vtx) + " " + std::to_string(vty) + "\n";
                    }
                    else {
                        ofs << line + "\n";
                    }
                }
                else {
                    ofs << line + "\n";
                }


            }
            ifs.close();
            ofs.close();
            std::remove(input.c_str());
            std::rename(output_tmp.string().c_str(), input.c_str());

        }

    };
    

    class Mesh_Trim_Simple {
    public:
        EigenTriMesh mesh_;
        EigenTriMesh mesh_out_;

        int added_faces = 0;
        int invalid_faces = 0;

        double threshold = 0.0001;

        //check whether two points are duplicate
        double threshold1 = 1e-3;

        // clipping bounding box [left,right,bot,top] in top-view
        Vector4f boundingbox_;

        // clipping polygon vertex list
        std::vector<Vector2f> polygon_vertice_;
        std::vector<int> polygon_indice_;


        //"r"<--> right plane
        //"l"<-->left plane
        //"b"<--> bot plane
        //"t"<-->top plane
        std::map<char, Vector4f> plane_map;
        std::unordered_map<EigenTriMesh::VertexHandle, EigenTriMesh::VertexHandle> old2new;
        std::unordered_map<int, EigenTriMesh::VertexHandle> kd2new;

        //inter point check count
        int inter_error;
        std::vector<size_t> fidxs;

        // initialize kd-tree

        // construct a kd-tree index:

        const size_t num_results = 1;
        size_t ret_index;
        double out_dist_sqr;

        Mesh_Trim_Simple(EigenTriMesh& obj, std::vector<Vector2f>& vertice, std::vector<int>& indice) {
            mesh_ = obj;
            polygon_vertice_ = vertice;
            polygon_indice_ = indice;
        }
        Mesh_Trim_Simple(EigenTriMesh& obj, std::vector<Vector3f>& vertice, std::vector<int>& indice) {
            mesh_ = obj;
            polygon_vertice_.clear();
            for (auto& vertex : vertice) {
                Vector2f v = Vector2f(vertex[0], vertex[1]);
                polygon_vertice_.push_back(v);
            }
            polygon_indice_ = indice;
        }
        Mesh_Trim_Simple(EigenTriMesh& obj, Vector4f& bbox) {
            mesh_ = obj;
            polygon_vertice_.clear();
            polygon_vertice_.push_back(Vector2f(bbox[0], bbox[2]));
            polygon_vertice_.push_back(Vector2f(bbox[1], bbox[2]));
            polygon_vertice_.push_back(Vector2f(bbox[1], bbox[3]));
            polygon_vertice_.push_back(Vector2f(bbox[0], bbox[3]));
            polygon_indice_.clear();
            polygon_indice_.push_back(0);
            polygon_indice_.push_back(1);
            polygon_indice_.push_back(2);
            polygon_indice_.push_back(3);
        }

        bool Start() {
            EigenTriMesh::VertexIter v_it, v_end(mesh_.vertices_end());
            EigenTriMesh::VertexHandle vh;
            int a = mesh_.n_vertices();
            for (v_it = mesh_.vertices_begin(); v_it != v_end; ++v_it) {
                if (!v_it->is_valid()) { continue; }
                Vector3f pt3d = mesh_.point(*v_it);
                Vector2f pt((double)pt3d[0],(double)pt3d[1]);
                if (!Pt_in_Poly(pt, polygon_vertice_, polygon_indice_)) {
                    vh = mesh_.vertex_handle(v_it->idx());
                    mesh_.delete_vertex(vh);
                }
            }
            mesh_.garbage_collection();
            return false;
        }




    };
    
    int Mesh_Trim(char* input_path, char* output_path, double* bbox_arr, double* tex_arr,std::string texname) {
        std::cout << fs::v1::path(output_path).parent_path().string() << " Clip" << std::endl;
        EigenTriMesh mesh_origin;
        //mesh_origin.request_vertex_texcoords2D();
        mesh_origin.request_halfedge_texcoords2D();
        mesh_origin.request_face_texture_index();

        OpenMesh::IO::Options opt(OpenMesh::IO::Options::FaceTexCoord);
        opt+= OpenMesh::IO::Options::FaceColor;
        // generate vertices
        if (!OpenMesh::IO::read_mesh(mesh_origin, input_path, opt))
        {
            std::cerr << "[ERROR]Cannot read mesh from " << input_path << std::endl;
            return 1;
        }
        //OpenMesh::IO::write_mesh(mesh_origin, "D:\\xuningli\\1.obj", opt);

        //time start
        auto start = high_resolution_clock::now();
        Vector4f bbox = { float(bbox_arr[0]),float(bbox_arr[1]),float(bbox_arr[2]),float(bbox_arr[3]) };
        Vector4f tex_bbox = { float(tex_arr[0]),float(tex_arr[1]),float(tex_arr[2]),float(tex_arr[3]) };
        Trimming T(mesh_origin, bbox);

        //T.SH();
        T.SH_raster();
        //std::cout << "added faces: " << T.added_faces << std::endl;
        //std::cout << "invalid faces: " << T.invalid_faces << std::endl;

        if (T.mesh_out.n_vertices() == 0) {
            std::cout << "No faces locaded in cropping area\n";
            ////time end
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            return 0;
        }
        //write obj
        if (abs(tex_bbox[0]-0)<1e-3 && abs(tex_bbox[1]-0)<1e-3 && abs(tex_bbox[2]-0)<1e-3 && abs(tex_bbox[3]-0)<1e-3) {
            T.write_manual(output_path);
        }
        else {
            T.write_manual_with_tex(output_path,tex_bbox);
        }

        //write mtl
        if (texname.empty()) {
            T.copy_mtl_tex(input_path, output_path);          
        }
        else {
            T.write_mtl(output_path,texname);
        }

        ////time end
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        
        return 0;
    }

    int Mesh_Trim_Multi(char* input_path, std::vector<std::vector<double>> crop_list, std::vector<std::string> crop_out_list, bool copy_texture) {
        EigenTriMesh mesh_origin;
        //mesh_origin.request_vertex_texcoords2D();
        mesh_origin.request_halfedge_texcoords2D();
        mesh_origin.request_face_texture_index();

        OpenMesh::IO::Options opt(OpenMesh::IO::Options::FaceTexCoord);
        opt += OpenMesh::IO::Options::FaceColor;
        // generate vertices
        if (!OpenMesh::IO::read_mesh(mesh_origin, input_path, opt))
        {
            std::cerr << "[ERROR]Cannot read mesh from " << input_path << std::endl;
            return 1;
        }
        //OpenMesh::IO::write_mesh(mesh_origin, "D:\\xuningli\\1.obj", opt);

        //time start
        auto start = high_resolution_clock::now();
        Trimming T(mesh_origin,input_path);

        //T.SH();
        T.SH_raster_multi(crop_list,crop_out_list,copy_texture);
        //std::cout << "added faces: " << T.added_faces << std::endl;
        //std::cout << "invalid faces: " << T.invalid_faces << std::endl;

        ////time end
        //auto stop = high_resolution_clock::now();
        //auto duration = duration_cast<microseconds>(stop - start);


        return 0;
    }

    void read_obj_and_center1(std::string obj_in, std::string obj_out, double& offx, double& offy, double& offz) {
        ifstream ifs(obj_in);
        ofstream ofs(obj_out);
        char char_arr[500];
        string line;
        bool first = true;
        while (!ifs.eof()) {
            getline(ifs, line);
            const char* token = line.c_str();
            if (token[0] == 'v' && (token[1] == ' ' || token[1] == '\t')) {
                strcpy(char_arr, line.c_str());
                double x, y, z;
                sscanf(char_arr, "v %lf %lf %lf\n", &x, &y, &z);
                if (first) {
                    offx = x;
                    offy = y;
                    offz = z;
                    first = false;
                }
                x -= offx;
                y -= offy;
                z -= offz;
                ofs << "v " << x << " " << y << " " << z << "\n";
            }
            else {
                ofs << line << "\n";
            }
        }
        ifs.close();
        ofs.close();
    }


    int Mesh_Trim_Multi(std::string input_path, std::vector<std::vector<double>> crop_list, std::vector<std::string> crop_out_list, bool copy_texture) {
        //omp_lock_t lck;
        //omp_init_lock(&lck);
        EigenTriMesh mesh_origin;
        mesh_origin.request_halfedge_texcoords2D();
        mesh_origin.request_face_texture_index();

        OpenMesh::IO::Options opt(OpenMesh::IO::Options::FaceTexCoord);
        opt += OpenMesh::IO::Options::FaceColor;
        // generate vertices
        bool status;
        //double OFFX, OFFY, OFFZ;
#pragma omp critical
        {

//            string cur_dir = fs::path(input_path).parent_path().string();
//            if (!fs::exists(input_path)) {
//                return 1;
//            }
//            std::string mesh_raw_in_centered = input_path.replace(input_path.find(".obj"), sizeof(".obj") - 1, "_centered.obj");
//#pragma omp critical
//            {
//                read_obj_and_center1(input_path, mesh_raw_in_centered, OFFX, OFFY, OFFZ);
//            }

            status = OpenMesh::IO::read_mesh(mesh_origin, input_path, opt); 
            //std::cout << "Finished" << std::endl;
            if (mesh_origin.n_faces() == 0) {
                status = false;
            }
        }
        if (!status) { return 1; }

        std::string cur_dir = fs::v1::path(input_path).parent_path().string();
        OpenMesh::MPropHandleT< std::map< int, std::string > > texindex;
        mesh_origin.get_property_handle(texindex, "TextureMapping");
        for (auto it = mesh_origin.property(texindex).begin(); it != mesh_origin.property(texindex).end(); ++it) {
            string texname = it->second;
            string tex_path;

            if (fs::exists(texname)) {
                tex_path = texname;
            }
            else {
                tex_path = (fs::path(cur_dir) / texname).string();
            }
            it->second = tex_path;

        }

        Trimming T(mesh_origin, input_path);
        T.SH_raster_multi(crop_list, crop_out_list, copy_texture);
  

        return 0;
    }
};

#endif MESHTRIM_H