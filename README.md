# Mesh Trimming

## Introduction
We are releasing our mesh editing tool as open-source software to facilitate efficient mesh trimming. Unlike existing tools like CloudCompare and Open3D, which simply remove triangles outside or inside AOi, leading a rough boundary, our tool takes a more precise approach. It carefully splits triangles that intersect with the AOI into smaller sub-triangles, ensuring a smooth clipping boundary. 

* Support keeping the inside part or outside part mesh.
* Support multiple cropping at the same time. 
* Time complexity: O(N+k), where N is the number of triangles within the mesh and k is the number of triangles intersecting with the AOI.
* AOI format: rectangle. 

## Usage
```
Trimming.exe aoi_info.txt keep_aoi[0: keep outside part, 1:keep inside part]

crop_info_txt: each line contains seven elements including output_path, xmin, xmax, ymin, ymax, zmin, zmax
```


## Results
### Mesh Visualization
![alt text](./demo/illustration.png)

### Comparision to CloudCompare
![alt text](./demo/compare.png)

### Detailed view
![alt text](./demo/detail_compare.png)

## Process
![alt text](./demo/pipeline.png)

(a) Overall pipeline

![alt text](./demo/table.png)

(b) Intersection types summary

## Contact:
Ningli Xu: xu.3961@buckeyemail.osu.edu , [website](https://ninglixu.github.io/)

