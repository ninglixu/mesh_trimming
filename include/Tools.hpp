#ifndef MESHTOOLS_H
#define MESHTOOLS_H
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
#include <boost/algorithm/string.hpp>
//nanoflann hpp
#include <nanoflann/nanoflann.hpp>
#include "nanoflann/utils.h"

namespace MeshEdit {
	//omp_lock_t lck_crop,lck_merge1,lck_merge2,lck_merge3;

	using namespace std::chrono;
	namespace fs = std::experimental::filesystem;
	using namespace Eigen;
	using namespace std;
	using namespace nanoflann;
	//Output LoD format
	class CLIP_INFO {
	public:
		std::vector<float> clip_bbox;
		std::vector<float> obj_bbox;
		std::string obj_path;
		std::string tex_path;
	};
	class TILE_INFO {
	public:
		std::vector<float> tile_bbox;
		std::vector<CLIP_INFO> clip;
		std::vector<CLIP_INFO> merge;
		bool up;
		int tile_col_id;
		int tile_row_id;
		string tile_name;
		TILE_INFO() {};
	};
	class LAYER_INFO {
	public:
		std::vector<std::vector<TILE_INFO>> layer;
		string layer_name;
		bool up;
		LAYER_INFO() {};
	};
	class LOD_INFO {
	public:
		std::vector<LAYER_INFO> lod_info;
		LOD_INFO() {};
	};
	//Input Mesh format
	class MESH_INFO {
	public:
		std::vector<float> bbox;
		std::string obj_path;
		int vertex_num;
		MESH_INFO() {};
	};
	class WHOLE_MESH_INFO {
	public:
		std::vector<MESH_INFO> obj_info;
		std::vector<float> whole_bbox;
		std::string utm_ref;
		float utm_x;
		float utm_y;
		float utm_z;
		WHOLE_MESH_INFO() {};

	};

	namespace BingTileSystem {
		double EarthRadius = 6378137;
		double MinLatitude = -85.05112878;
		double MaxLatitude = 85.05112878;
		double MinLongitude = -180;
		double MaxLongitude = 180;
		double PI = 3.1415926535897932;

		/// <summary>  
		/// Clips a number to the specified minimum and maximum values.  
		/// </summary>  
		/// <param name="n">The number to clip.</param>  
		/// <param name="minValue">Minimum allowable value.</param>  
		/// <param name="maxValue">Maximum allowable value.</param>  
		/// <returns>The clipped value.</returns>  
		double Clip(double n, double minValue, double maxValue)
		{
			return min(max(n, minValue), maxValue);
		}

		/// <summary>  
		/// Determines the map width and height (in pixels) at a specified level  
		/// of detail.  
		/// </summary>  
		/// <param name="levelOfDetail">Level of detail, from 1 (lowest detail)  
		/// to 23 (highest detail).</param>  
		/// <returns>The map width and height in pixels.</returns>  
		uint MapSize(int levelOfDetail)
		{
			return (uint)256 << levelOfDetail;
		}

		/// <summary>  
		/// Determines the ground resolution (in meters per pixel) at a specified  
		/// latitude and level of detail.  
		/// </summary>  
		/// <param name="latitude">Latitude (in degrees) at which to measure the  
		/// ground resolution.</param>  
		/// <param name="levelOfDetail">Level of detail, from 1 (lowest detail)  
		/// to 23 (highest detail).</param>  
		/// <returns>The ground resolution, in meters per pixel.</returns>  
		double GroundResolution(double latitude, int levelOfDetail)
		{
			latitude = Clip(latitude, MinLatitude, MaxLatitude);
			return cos(latitude * PI / 180) * 2 * PI * EarthRadius / MapSize(levelOfDetail);
		}

		/// <summary>  
		/// Determines the map scale at a specified latitude, level of detail,  
		/// and screen resolution.  
		/// </summary>  
		/// <param name="latitude">Latitude (in degrees) at which to measure the  
		/// map scale.</param>  
		/// <param name="levelOfDetail">Level of detail, from 1 (lowest detail)  
		/// to 23 (highest detail).</param>  
		/// <param name="screenDpi">Resolution of the screen, in dots per inch.</param>  
		/// <returns>The map scale, expressed as the denominator N of the ratio 1 : N.</returns>  
		double MapScale(double latitude, int levelOfDetail, int screenDpi)
		{
			return GroundResolution(latitude, levelOfDetail) * screenDpi / 0.0254;
		}

		/// <summary>  
		/// Converts a point from latitude/longitude WGS-84 coordinates (in degrees)  
		/// into pixel XY coordinates at a specified level of detail.  
		/// </summary>  
		/// <param name="latitude">Latitude of the point, in degrees.</param>  
		/// <param name="longitude">Longitude of the point, in degrees.</param>  
		/// <param name="levelOfDetail">Level of detail, from 1 (lowest detail)  
		/// to 23 (highest detail).</param>  
		/// <param name="pixelX">Output parameter receiving the X coordinate in pixels.</param>  
		/// <param name="pixelY">Output parameter receiving the Y coordinate in pixels.</param>  
		void LatLongToPixelXY(double latitude, double longitude, int levelOfDetail, int& pixelX, int& pixelY)
		{
			latitude = Clip(latitude, MinLatitude, MaxLatitude);
			longitude = Clip(longitude, MinLongitude, MaxLongitude);

			double x = (longitude + 180) / 360;
			double sinLatitude = sin(latitude * PI / 180);
			double y = 0.5 - log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * PI);

			uint mapSize = MapSize(levelOfDetail);
			pixelX = (int)Clip(x * mapSize + 0.5, 0, mapSize - 1);
			pixelY = (int)Clip(y * mapSize + 0.5, 0, mapSize - 1);
		}

		/// <summary>  
		/// Converts a pixel from pixel XY coordinates at a specified level of detail  
		/// into latitude/longitude WGS-84 coordinates (in degrees).  
		/// </summary>  
		/// <param name="pixelX">X coordinate of the point, in pixels.</param>  
		/// <param name="pixelY">Y coordinates of the point, in pixels.</param>  
		/// <param name="levelOfDetail">Level of detail, from 1 (lowest detail)  
		/// to 23 (highest detail).</param>  
		/// <param name="latitude">Output parameter receiving the latitude in degrees.</param>  
		/// <param name="longitude">Output parameter receiving the longitude in degrees.</param>  
		void PixelXYToLatLong(int pixelX, int pixelY, int levelOfDetail, double& latitude, double& longitude)
		{
			double mapSize = MapSize(levelOfDetail);
			double x = (Clip(pixelX, 0, mapSize - 1) / mapSize) - 0.5;
			double y = 0.5 - (Clip(pixelY, 0, mapSize - 1) / mapSize);

			latitude = 90 - 360 * atan(exp(-y * 2 * PI)) / PI;
			longitude = 360 * x;
		}

		/// <summary>  
		/// Converts pixel XY coordinates into tile XY coordinates of the tile containing  
		/// the specified pixel.  
		/// </summary>  
		/// <param name="pixelX">Pixel X coordinate.</param>  
		/// <param name="pixelY">Pixel Y coordinate.</param>  
		/// <param name="tileX">Output parameter receiving the tile X coordinate.</param>  
		/// <param name="tileY">Output parameter receiving the tile Y coordinate.</param>  
		void PixelXYToTileXY(int pixelX, int pixelY, int& tileX, int& tileY)
		{
			tileX = pixelX / 256;
			tileY = pixelY / 256;
		}

		/// <summary>  
		/// Converts tile XY coordinates into pixel XY coordinates of the upper-left pixel  
		/// of the specified tile.  
		/// </summary>  
		/// <param name="tileX">Tile X coordinate.</param>  
		/// <param name="tileY">Tile Y coordinate.</param>  
		/// <param name="pixelX">Output parameter receiving the pixel X coordinate.</param>  
		/// <param name="pixelY">Output parameter receiving the pixel Y coordinate.</param>  
		void TileXYToPixelXY(int tileX, int tileY, int& pixelX, int& pixelY)
		{
			pixelX = tileX * 256;
			pixelY = tileY * 256;
		}

		void TileXYToBBOXLLA(int tileX, int tileY, int layer, double& minx, double& maxx, double& miny, double& maxy) {
			int pixel_minx = tileX * 256;
			int pixel_maxx = (tileX + 1) * 256;
			int pixel_miny = tileY * 256;
			int pixel_maxy = (tileY + 1) * 256;
			PixelXYToLatLong(pixel_minx, pixel_miny, layer, maxy, minx);
			PixelXYToLatLong(pixel_maxx, pixel_maxy, layer, miny, maxx);
		}

		/// <summary>  
		/// Converts tile XY coordinates into a QuadKey at a specified level of detail.  
		/// </summary>  
		/// <param name="tileX">Tile X coordinate.</param>  
		/// <param name="tileY">Tile Y coordinate.</param>  
		/// <param name="levelOfDetail">Level of detail, from 1 (lowest detail)  
		/// to 23 (highest detail).</param>  
		/// <returns>A string containing the QuadKey.</returns>  
		string TileXYToQuadKey(int tileX, int tileY, int levelOfDetail)
		{
			string quadKey = "";
			for (int i = levelOfDetail; i > 0; i--)
			{
				char digit = '0';
				int mask = 1 << (i - 1);
				if ((tileX & mask) != 0)
				{
					digit++;
				}
				if ((tileY & mask) != 0)
				{
					digit++;
					digit++;
				}
				quadKey += digit;
			}
			return quadKey;
		}

		/// <summary>  
		/// Converts a QuadKey into tile XY coordinates.  
		/// </summary>  
		/// <param name="quadKey">QuadKey of the tile.</param>  
		/// <param name="tileX">Output parameter receiving the tile X coordinate.</param>  
		/// <param name="tileY">Output parameter receiving the tile Y coordinate.</param>  
		/// <param name="levelOfDetail">Output parameter receiving the level of detail.</param>  
		void QuadKeyToTileXY(string quadKey, int& tileX, int& tileY, int& levelOfDetail)
		{
			tileX = tileY = 0;
			levelOfDetail = quadKey.length();
			for (int i = levelOfDetail; i > 0; i--)
			{
				int mask = 1 << (i - 1);
				switch (quadKey[levelOfDetail - i])
				{
				case '0':
					break;

				case '1':
					tileX |= mask;
					break;

				case '2':
					tileY |= mask;
					break;

				case '3':
					tileX |= mask;
					tileY |= mask;
					break;

				default:
					std::cout << "Error: Invalid Sequence\n";
				}
			}
		}
	};


	typedef KDTreeSingleIndexDynamicAdaptor<
		L2_Simple_Adaptor<double, PointCloud<double> >,
		PointCloud<double>,
		3 /* dim */
	> my_kd_tree_t;
	namespace fs = std::experimental::filesystem;

	struct EigenTraits : OpenMesh::DefaultTraits {
		using Point = Eigen::Vector3f;
		//using Normal = Eigen::Vector3f;
		using TexCoord2D = Eigen::Vector2f;
		//using Color = Eigen::Vector3f;
	};

	struct EigenTraitsF : OpenMesh::DefaultTraits {
		using Point = Eigen::Vector3f;
		using Normal = Eigen::Vector3f;
		using TexCoord2D = Eigen::Vector2f;
		//using Color = Eigen::Vector3f;
	};

	using EigenTriMesh = OpenMesh::TriMesh_ArrayKernelT<EigenTraits>;
	using EigenTriMeshF = OpenMesh::TriMesh_ArrayKernelT<EigenTraitsF>;

	struct EigenFace {
		std::vector<Vector3f> v_list;
		std::vector<Eigen::Vector2f> tex_list;
		std::vector<int> v_id;
		int tex_index;
	};

	struct EigenFace_homo {
		std::vector<Vector4f> v_list;
		std::vector<Eigen::Vector2f> tex_list;
	};

	const double INF = 9999999999;
	//bool Pt_in_Bbox(Vector2f& pt, Vector4f& bbox) { return false; }
	bool Pt_in_Bbox(Vector2f& pt, Vector4f& bbox) {
		if (pt[0] >= bbox[0] && pt[0] <= bbox[1] && pt[1] >= bbox[2] && pt[1] <= bbox[1]) {
			return true;
		}
		else {
			return false;
		}
	}
	
	// Given three collinear points p, q, r, the function checks if
	// point q lies on line segment 'pr'
	//bool Pt_on_LineSeg(Vector2f& p, Vector2f& q, Vector2f& r) 
	//	{return false; }
	bool Pt_on_LineSeg(Vector2f& p, Vector2f& q, Vector2f& r) {
		if (q[0] <= max(p[0], r[0]) && q[0] >= min(p[0], r[0]) &&
			q[1] <= max(p[1], r[1]) && q[1] >= min(p[1], r[1])) {
			return true;
		}
		else {
			return false;
		}
	}

	// To find orientation of ordered triplet (p, q, r).
	// The function returns following values
	// 0 --> p, q and r are collinear
	// 1 --> Clockwise
	// 2 --> Counterclockwise
	//int Pts_orientation(Vector2f& p, Vector2f& q, Vector2f& r) { return false; }
	int Pts_orientation(Vector2f& p, Vector2f& q, Vector2f& r)
	{
		Vector2f v1 = q - p;
		Vector2f v2 = r - q;
		float val = v1[0] * v2[1] - v1[1] * v2[0];
		//int val = (q[1] - p[1]) * (r[0] - q[0]) -
		//	(q[0] - p[0]) * (r[1] - q[1]);

		if (val == 0) return 0; // collinear
		return (val > 0) ? 1 : 2; // clock or counterclock wise
	}

	
	// The function that returns true if line segment 'p1q1'
	// and 'p2q2' intersect.
	//bool LineSeg_Intersect(Vector2f& p1, Vector2f& q1, Vector2f& p2, Vector2f& q2) { return false; }
	bool LineSeg_Intersect(Vector2f& p1, Vector2f& q1, Vector2f& p2, Vector2f& q2)
	{
		// Find the four orientations needed for general and
		// special cases
		int o1 = Pts_orientation(p1, q1, p2);
		int o2 = Pts_orientation(p1, q1, q2);
		int o3 = Pts_orientation(p2, q2, p1);
		int o4 = Pts_orientation(p2, q2, q1);

		// General case
		if (o1 != o2 && o3 != o4)
			return true;

		// Special Cases
		// p1, q1 and p2 are collinear and p2 lies on segment p1q1
		if (o1 == 0 && Pt_on_LineSeg(p1, p2, q1)) return true;

		// p1, q1 and p2 are collinear and q2 lies on segment p1q1
		if (o2 == 0 && Pt_on_LineSeg(p1, q2, q1)) return true;

		// p2, q2 and p1 are collinear and p1 lies on segment p2q2
		if (o3 == 0 && Pt_on_LineSeg(p2, p1, q2)) return true;

		// p2, q2 and q1 are collinear and q1 lies on segment p2q2
		if (o4 == 0 && Pt_on_LineSeg(p2, q1, q2)) return true;

		return false; // Doesn't fall in any of the above cases
	}
	// Returns true if the point p lies inside the polygon[] with n vertices
	
	//bool Pt_in_Poly(Vector2f& pt, std::vector<Vector2f>& vertices, std::vector<int>& poly_indices) { return false; }
	bool Pt_in_Poly(Vector2f& pt, std::vector<Vector2f>& vertices, std::vector<int>& poly_indices) 
	{

		// There must be at least 3 vertices in polygon
		if (poly_indices.size() < 3) return false;

		// Create a point for line segment from pt to infinite
		Vector2f extreme = { INF, pt[1] };

		// Count intersections of the above line with sides of polygon
		int count = 0, i = 0;
		do
		{
			int next = (i + 1) % poly_indices.size();

			// Check if the line segment from 'p' to 'extreme' intersects
			// with the line segment from 'polygon[i]' to 'polygon[next]'
			if (LineSeg_Intersect(vertices[poly_indices[i]], vertices[poly_indices[next]], pt, extreme))
			{
				// If the point 'p' is collinear with line segment 'i-next',
				// then check if it lies on segment. If it lies, return true,
				// otherwise false
				if (Pts_orientation(vertices[poly_indices[i]], pt, vertices[poly_indices[next]]) == 0) {
					return Pt_on_LineSeg(vertices[poly_indices[i]], pt, vertices[poly_indices[next]]);
				}
				count++;
			}
			i = next;
		} while (i != 0);

		// Return true if count is odd, false otherwise
		return count & 1; // Same as (count%2 == 1)
	}

	bool Triangle_Area(Vector3f& p1, Vector3f& p2, Vector3f& p3, double& area) {
		area=((p1 - p2).cross(p3 - p2)).norm()/2;
		return true;
	}

	bool Load_Mesh(string path, EigenTriMesh& mesh) {
		bool OBJ_or_PLY = true;
		OpenMesh::IO::Options opt;
		if (path.find(".obj") != std::string::npos && path.find(".ply") == std::string::npos) {
			OBJ_or_PLY = true;
		}
		else if (path.find(".obj") == std::string::npos && path.find(".ply") != std::string::npos) {
			OBJ_or_PLY = false;
		}
		else {
			std::cout << "Input path is not obj or ply" << std::endl;
			return false;
		}
		if (OBJ_or_PLY) {
			mesh.request_halfedge_texcoords2D();
			mesh.request_face_texture_index();
			mesh.request_face_colors();
			mesh.request_face_status();
			mesh.request_vertex_status();
			mesh.request_vertex_colors();
			mesh.request_halfedge_colors();
			mesh.request_halfedge_status();
			opt += OpenMesh::IO::Options::FaceTexCoord;
			opt += OpenMesh::IO::Options::FaceColor;
			opt += OpenMesh::IO::Options::VertexColor;
			opt += OpenMesh::IO::Options::VertexTexCoord;
		}
		else {
			mesh.request_face_status();
			mesh.request_vertex_status();
			mesh.request_vertex_colors();
			opt += OpenMesh::IO::Options::VertexColor;
			opt += OpenMesh::IO::Options::VertexTexCoord;
			opt += OpenMesh::IO::Options::ColorFloat;

		}
		//mesh.request_vertex_texcoords2D();

		//OpenMesh::MPropHandleT< std::map< int, std::string > > texindex;
		//std::map<int, int> tiA_map, tiB_map;
		//int meshA_texture_cnt = 0;
		//if (!mesh.get_property_handle(texindex, "TextureMapping")) {
		//	mesh.add_property(texindex, "TextureMapping");
		//}

		if (!OpenMesh::IO::read_mesh(mesh, path,opt))
		{
			std::cerr << "[ERROR]Cannot read mesh from " << path << std::endl;
			return false;
		}
		else {
			return true;
		}
	}
	void copy_mtl_tex(std::string input, std::string output, EigenTriMesh& mesh_out) {
		std::string out_filename = fs::v1::path(output).filename().string();
		std::string out_file_ext = fs::v1::path(output).filename().extension().string();
		fs::v1::path mtl_in_path = fs::v1::path(input).replace_extension(".mtl");
		fs::v1::path mtl_out_path = fs::v1::path(output).replace_extension(".mtl");
		std::ifstream mtl_in(mtl_in_path);
		std::ofstream mtl_out(mtl_out_path);

		std::vector<std::string> in_texture_name;
		std::vector<std::string> out_texture_name;
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
					std::string filename = out_filename;
					if (!mesh_out.property(texindex_out)[i].empty()) {
						boost::replace_all(filename, out_file_ext, "_" + std::to_string(i)+".jpg");
						mtl_out << "newmtl " << "mat" << i << '\n';
						mtl_out << "Ka 1.000 1.000 1.000" << '\n';
						mtl_out << "Kd 1.000 1.000 1.000" << '\n';
						mtl_out << "illum 1" << '\n';
						mtl_out << "map_Kd " << filename << "\n";
						in_texture_name.push_back(mesh_out.property(texindex_out)[i]);
						out_texture_name.push_back(filename);
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
							in_texture_name.push_back(line.substr(line.find(" ") + 1));
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
		int cnt = 0;
		for (auto it = in_texture_name.begin(); it != in_texture_name.end(); ++it) {

			fs::v1::path tex_out_path = fs::v1::path(mtl_out_path).replace_filename(out_texture_name[cnt]);
			fs::v1::path tex_in_path = fs::v1::path(mtl_in_path).replace_filename(*it);
			//fs::copy_file(tex_in_path, tex_out_path);
			if (fs::v1::exists(tex_out_path)) {
				remove(tex_out_path);
				fs::copy_file(tex_in_path, tex_out_path);
			}
			else {
				fs::copy_file(tex_in_path, tex_out_path);
			}
			cnt++;
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

	bool Write_Mesh(string path, EigenTriMesh& mesh) {
		OpenMesh::MPropHandleT< std::map< int, std::string > > texindex;
		if (!mesh.get_property_handle(texindex, "TextureMapping")) {
			mesh.add_property(texindex, "TextureMapping");
		}
		for (auto it = mesh.property(texindex).begin(); it != mesh.property(texindex).end(); ++it) {
			string a = it->second;
		}

		return false;
	}

	void Write_Mesh_Manual(std::string out_path, EigenTriMesh& mesh_out,std::string in_path) {
		std::string mtl_name = fs::v1::path(out_path).filename().replace_extension(".mtl").string();
		std::ofstream ofs(out_path);
		Vector3f pts;
		Vector2f tex;
		EigenTriMesh::HalfedgeHandle heh;
		boost::format fmt;
		ofs << "mtllib " << mtl_name << "\n";
		//wrtie v
		for (int i = 0; i < mesh_out.n_vertices(); ++i) {
			pts = mesh_out.point(mesh_out.vertex_handle(i));
			fmt = boost::format("v %s %s %s\n") % pts[0] % pts[1] % pts[2];
			ofs << fmt.str();
		}
		//wrtie vt
		for (int i = 0; i < mesh_out.n_faces(); ++i) {
			heh = mesh_out.halfedge_handle(mesh_out.face_handle(i));
			tex = mesh_out.texcoord2D(heh);
			fmt = boost::format("vt %s %s\n") % tex[0] % tex[1];
			ofs << fmt.str();

			heh = mesh_out.next_halfedge_handle(heh);
			tex = mesh_out.texcoord2D(heh);
			fmt = boost::format("vt %s %s\n") % tex[0] % tex[1];
			ofs << fmt.str();

			heh = mesh_out.next_halfedge_handle(heh);
			tex = mesh_out.texcoord2D(heh);
			fmt = boost::format("vt %s %s\n") % tex[0] % tex[1];
			ofs << fmt.str();
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
			tmp_tex_index = mesh_out.texture_index(mesh_out.face_handle(i));
			if (useMatrial && !mesh_out.property(texindex_out)[tmp_tex_index].empty()) {
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

		copy_mtl_tex(in_path, out_path,mesh_out);


	}
	

	void write_mtl(std::string output, std::string texname) {
		fs::v1::path mtl_out_path = fs::v1::path(output).replace_extension(".mtl");
		std::ofstream mtl_out(mtl_out_path);
		mtl_out << "newmtl " << "mat1" << '\n';
		mtl_out << "Ka 1.000 1.000 1.000" << '\n';
		mtl_out << "Kd 1.000 1.000 1.000" << '\n';
		mtl_out << "illum 1" << '\n';
		mtl_out << "map_Kd " << texname << "\n";
		mtl_out.close();
	}

}
#endif MESHTOOLS_H