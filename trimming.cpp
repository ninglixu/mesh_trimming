
#include <iostream>
#include <MeshTrim.hpp>



int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cout << "usage : exe obj_in_path crop_info_txt" << std::endl;
        std::cout << "crop_info_txt: each line contains seven elements including output_path, xmin, xmax, ymin, ymax, zmin, zmax" << std::endl;
        return 0;
    }
    char* input = argv[1];
    char* crop_info_path = argv[2];

    std::string input_path(input);

    std::vector<std::vector<float>> crop_list;
    std::vector<std::string> crop_out_list;
    std::ifstream ifs(crop_info_path);
    std::string line;
    while (std::getline(ifs, line)) {
        size_t found = line.find(" ");
        std::string output_tmp = line.substr(0, found);
        line = line.substr(found+1);
        found = line.find(" ");
        float xmin = std::stof(line.substr(0, found));

        line = line.substr(found+1);
        found = line.find(" ");
        float xmax = std::stof(line.substr(0, found));

        line = line.substr(found+1);
        found = line.find(" ");
        float ymin = std::stof(line.substr(0, found));

        line = line.substr(found+1);
        found = line.find(" ");
        float ymax = std::stof(line.substr(0, found));

        line = line.substr(found+1);
        found = line.find(" ");
        float zmin = std::stof(line.substr(0, found));

        line = line.substr(found+1);
        found = line.find(" ");
        float zmax = std::stof(line.substr(0, found));

        std::vector<float> bbox = { xmin,xmax,ymin,ymax,zmin,zmax };
        crop_list.push_back(bbox);
        crop_out_list.push_back(output_tmp);
    }
    ifs.close();

    MeshTrim::MESH_TRIM(input_path, crop_out_list, crop_list);
    
}
