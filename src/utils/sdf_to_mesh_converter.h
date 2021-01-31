#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>

#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

#include "visualizer/opengl/tiny_opengl3_app.h"
#include "visualizer/opengl/utils/tiny_chrome_trace_util.h"
#include "visualizer/opengl/utils/tiny_logging.h"
#include "utils/file_utils.hpp"
#include "visualizer/opengl/utils/tiny_mesh_utils.h"
#include "stb_image/stb_image.h"

struct CGALShape
{
    std::vector<GfxVertexFormat1> vertices;
    std::vector<int> indices;

    size_t num_triangles;
    size_t num_vertices;
};

template <typename K, typename Tr>
struct SdfToMeshConverter
{
    using FT = typename K::FT;
    using Point = typename K::Point_3;
    using Mesh_domain = CGAL::Labeled_mesh_domain_3<K>;
    typedef FT (Function)(const Point&);


    #ifdef CGAL_CONCURRENT_MESH_3
    using Concurrency_tag = CGAL::Parallel_tag;
    #else
    using Concurrency_tag = CGAL::Sequential_tag;
    #endif

    using C3T3 = CGAL::Mesh_complex_3_in_triangulation_3<Tr>;
    using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;
    using Vertex_handle = typename C3T3::Vertex_handle;
    using Facet = typename C3T3::Facet;

    using Surface_mesh = CGAL::Surface_mesh<Point>;
    using vertex_descriptor = typename boost::graph_traits<Surface_mesh>::vertex_descriptor;

    Mesh_domain domain;
    Mesh_criteria criteria;
    C3T3 c3t3;
    Function* mesh_function;
    const double DIFF = 1e-4;

    SdfToMeshConverter(Function* mesh_func, const Mesh_domain& dom, const Mesh_criteria& crit)
    : mesh_function(mesh_func)
    , domain(dom)
    , criteria(crit)
    {}

    const C3T3& generate_mesh()
    {
        c3t3 = CGAL::make_mesh_3<C3T3>(domain, criteria);
        return c3t3;
    }

    const Tr& get_triangulation() const
    {
        return c3t3.triangulation();
    }

    std::vector<FT> compute_normal(const Point& p)
    {
        auto x = p.x();
        auto y = p.y();
        auto z = p.z();

        Point x_upper(x + DIFF, y, z);
        Point x_lower(x - DIFF, y, z);
        auto x_diff = (mesh_function(x_upper) - mesh_function(x_lower)) / (2 * DIFF);

        Point y_upper(x, y + DIFF, z);
        Point y_lower(x, y - DIFF, z);
        auto y_diff = (mesh_function(y_upper) - mesh_function(y_lower)) / (2 * DIFF);

        Point z_upper(x, y, z + DIFF);
        Point z_lower(x, y, z - DIFF);
        auto z_diff = (mesh_function(z_upper) - mesh_function(z_lower)) / (2 * DIFF);

        // CGAL::Point_3<FT> result(x_diff, y_diff, z_diff);
        std::vector<FT> result(3);
        result[0] = x_diff;
        result[1] = y_diff;
        result[2] = z_diff;
        return result;
    }

    CGALShape convert_to_shape()
    {
        this->generate_mesh();
        const Tr& tr = this->get_triangulation();
        
        CGALShape shape;
        std::unordered_map<Vertex_handle, int> V;
        int inum = 1;

        
        // CGAL::Surface_mesh<Point> surface_mesh;
        // CGAL::facets_in_complex_3_to_triangle_mesh(c3t3, surface_mesh);

        // auto vnormals = surface_mesh.add_property_map<vertex_descriptor, K::Vector_3>("v:normals", CGAL::NULL_VECTOR).first;
        // CGAL::Polygon_mesh_processing::compute_vertex_normals(surface_mesh, vnormals);

        // Load Vertices
        auto vertices_num = tr.number_of_vertices();
        shape.num_vertices = vertices_num;

        // for (auto vit = surface_mesh.vertices_begin(); vit != surface_mesh.vertices_end(); ++vit) {
            
        // }

        for (auto vit = tr.finite_vertices_begin(); vit != tr.finite_vertices_end(); ++vit) {
            V[vit] = inum++;
            auto p = tr.point(vit);

            GfxVertexFormat1 vertex_curr;
            vertex_curr.x = CGAL::to_double(p.x());
            vertex_curr.y = CGAL::to_double(p.y());
            vertex_curr.z = CGAL::to_double(p.z());
            vertex_curr.w = 1.;
            // vertex_curr.nx = vertex_curr.ny = vertex_curr.nz = 1.;

            auto v_normal = compute_normal(p.point());
            vertex_curr.nx = CGAL::to_double(v_normal[0]);
            vertex_curr.ny = CGAL::to_double(v_normal[1]);
            vertex_curr.nz = CGAL::to_double(v_normal[2]);
            vertex_curr.u = vertex_curr.v = 0.;

            // CGAL::Polygon_mesh_processing::compute_vertex_normal(vit, c3t3);

            shape.vertices.push_back(vertex_curr);
        }

        // Load triangles
        auto triangles_num = c3t3.number_of_facets_in_complex();
        triangles_num += triangles_num; 

        shape.num_triangles = triangles_num;
    
        for (auto fit = c3t3.facets_in_complex_begin(); fit != c3t3.facets_in_complex_end(); ++fit) {
            Facet f = (*fit);

            if (f.first->subdomain_index() > f.first->neighbor(f.second)->subdomain_index()) {
                f = tr.mirror_facet(f);
            }

            Vertex_handle vh1 = f.first->vertex((f.second + 1) % 4);
            Vertex_handle vh2 = f.first->vertex((f.second + 2) % 4);
            Vertex_handle vh3 = f.first->vertex((f.second + 3) % 4);
            if (f.second % 2 != 0) {
                std::swap(vh2, vh3);
            }

            shape.indices.push_back(V[vh1]-1);
            shape.indices.push_back(V[vh2]-1);
            shape.indices.push_back(V[vh3]-1);

            shape.indices.push_back(V[vh3]-1);
            shape.indices.push_back(V[vh2]-1);
            shape.indices.push_back(V[vh1]-1);
        }

        return shape;
    }

    void output_to_medit(std::ostream& output)
    {
        c3t3.output_to_medit(output);
    }
};