
#include <iostream>
#include <MeshTrim.hpp>
namespace ME = MeshEdit;

int main(int argc,char* argv[])
{
    //debug argc!=1
    //release argc==1
    if (argc == 1) {
        std::cout << "usage : clip_exe mode(0: single crop, 1: multi crop) (if mode==0) obj_in_path\n obj_out_path\n [clip_bbox] xmin xmax ymin ymax\n  [tex_bbox] xmin xmax ymin ymax\n texname \n (if mode=1) txt_path copy_txt_or_not(0 no or 1 yes)" << std::endl;
    }
    else {
        int mode = (int)strtof(argv[1],NULL);
        if (mode == 0) {
            ////for release
            char* input_path = argv[2];
            char* output_path = argv[3];
            //Vector4f clip_bbox;  
            double* clip_bbox = new double[4];
            clip_bbox[0] = (double)strtod(argv[4], NULL);
            clip_bbox[1] = (double)strtod(argv[5], NULL);
            clip_bbox[2] = (double)strtod(argv[6], NULL);
            clip_bbox[3] = (double)strtod(argv[7], NULL);
            double* tex_bbox = new double[4];
            tex_bbox[0] = 0;
            tex_bbox[1] = 0;
            tex_bbox[2] = 0;
            tex_bbox[3] = 0;
            if (argc > 8) {
                tex_bbox[0] = (double)strtod(argv[8], NULL);
                tex_bbox[1] = (double)strtod(argv[9], NULL);
                tex_bbox[2] = (double)strtod(argv[10], NULL);
                tex_bbox[3] = (double)strtod(argv[11], NULL);
            }
            std::string texname = "";
            if (argc > 12) {
                texname = argv[11];
            }

            ME::Mesh_Trim(input_path, output_path, clip_bbox, tex_bbox, texname);
            delete[] clip_bbox;
            delete[] tex_bbox;
            return 1;
        }
        else if(mode == 1) {
            char* input_path = argv[2];
            /* should be in format:
            * xmin
            * xmax
            * ymin
            * ymax
            * output_path
            */
            char* crop_txt_path = argv[3]; 
            int copy_text_or_not = (int)strtof(argv[4], NULL);
            std::vector<std::vector<double>> crop_list;
            std::vector<std::string> crop_out_list;
            std::ifstream ifs(crop_txt_path);
            std::string line;
            std::vector<double> tmp_crop;
            while (std::getline(ifs, line)) {
                if (tmp_crop.size() != 4) {
                    tmp_crop.push_back(stod(line));
                }
                else {
                    crop_list.push_back(tmp_crop);
                    tmp_crop.clear();
                    crop_out_list.push_back(line);
                }
            }
            if (crop_list.size() != crop_out_list.size()) {
                std::cout << "Load " << crop_txt_path << " Error\n";
                return 1;
            }
            else {
                ME::Mesh_Trim_Multi(input_path, crop_list, crop_out_list, copy_text_or_not);
                return 1;
            }
        }

    }
}