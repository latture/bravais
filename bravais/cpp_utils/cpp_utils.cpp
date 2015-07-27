#include "cpp_utils.h"

#include <vector>
#include <set>
#include <numeric>
#include <functional>
#include <iostream>
#include <cmath>

namespace cpp_utils {

std::vector< std::vector < int > > delete_duplicates_int(const std::vector< std::vector < int > > &input) {

    // create set to delete duplicates
    std::set< std::vector < int > > unique_rows(input.begin(), input.end());

    // copy data back into a vector
    std::vector< std::vector < int > > output(unique_rows.begin(), unique_rows.end());

    return output;
}

std::vector< std::vector < double > > delete_duplicates_dbl(std::vector< std::vector < double > > input) {

    // round data
    if (input.size() < 100)
    {
	    unsigned int num_rows = input[0].size();
	    for (unsigned int i = 0; i < input.size(); ++i)
	    {
	        for (unsigned int j = 0; j < num_rows; ++j)
	        {
	            input[i][j] = std::round(input[i][j] * 1.0e7)/1.0e7;
	        }
	    }
    }

    else
    {
    	unsigned int i, j;
	    unsigned int num_rows = input[0].size();
	    #pragma omp parallel for collapse(2) private(i, j)
		    for (i = 0; i < input.size(); ++i)
		    {
		        for (j = 0; j < num_rows; ++j)
		        {
		            input[i][j] = std::round(input[i][j] * 1.0e7)/1.0e7;
		        }
		    }
    }

    // create set to delete duplicates
    std::set< std::vector < double > > unique_rows(input.begin(), input.end());

    // copy data back into a vector
    std::vector< std::vector < double > > output(unique_rows.begin(), unique_rows.end());

    return output;
}

std::vector< int > test_rows(const std::vector< double > &row, const std::vector< std::vector< double > > &vec2d) {

    bool isEqual;
    const double tolerance = 1e-7;

    std::vector< int > output;

    for (unsigned int i = 0; i < vec2d.size(); ++i)
    {
        isEqual = true;

        for (unsigned int j = 0; j < row.size(); ++j)
        {
            if (std::abs(row[j] - vec2d[i][j]) > tolerance) {
                isEqual = false;
                break;
            }
        }

        if (isEqual) output.push_back(i);
    }

    return output;
}

template <typename T>
unsigned int lower_bound(const std::vector< std::vector< T > > &vec, T key, unsigned int column, unsigned int imin, unsigned int imax) {
	int count = imax - imin;
	int idx, step;

	while (count > 0) {
		step = count/2;
		idx = imin + step;
		if (vec[idx][column] < key) {
			imin = ++idx;
			count -= step + 1;
		}
		else count = step;
	}
	return imin;
}

template <typename T>
unsigned int upper_bound(const std::vector< std::vector< T > > &vec, T key, unsigned int column, unsigned int imin, unsigned int imax) {
	int count = imax - imin;
	int idx, step;

	while (count > 0) {
		step = count/2;
		idx = imin + step;
		if (!(key < vec[idx][column])) {
			imin = ++idx;
			count -= step + 1;
		}
		else count = step;
	}
	return imin;
}

std::vector< std::vector< int > > replace_with_idx(std::vector< std::vector< double > > nodes, std::vector< std::vector< std::vector< double > > > elemCoords) {
	const unsigned int num_elems = elemCoords.size();
    const unsigned int num_nodes = nodes.size();
	const unsigned int nodes_per_elem = elemCoords[0].size();
	const unsigned int dims = elemCoords[0][0].size();
	unsigned int i, j, k;
	int idx, imax;

    // round element coordinates
    if (num_elems < 1000)
    {
	    for (unsigned int i = 0; i < num_elems; ++i)
	    {
	        for (unsigned int j = 0; j < nodes_per_elem; ++j)
	        {
	            for (unsigned int k = 0; k < dims; ++k)
	            {
	            	elemCoords[i][j][k] = std::round(elemCoords[i][j][k] * 1.0e7)/1.0e7;
	            }
	        }
	    }
    }

    else
    {
	    #pragma omp parallel for collapse(3) private(i, j, k)
	    for (i = 0; i < num_elems; ++i)
	    {
	        for (j = 0; j < nodes_per_elem; ++j)
	        {
	            for (k = 0; k < dims; ++k)
	            {
	            	elemCoords[i][j][k] = std::round(elemCoords[i][j][k] * 1.0e7)/1.0e7;
	            }
	        }
	    }
    }

    // round nodal coordinates
    if (num_nodes < 10)
    {
	    for (unsigned int i = 0; i < num_nodes; ++i)
	    {
	        for (unsigned int j = 0; j < dims; ++j)
	        {
            	nodes[i][j] = std::round(nodes[i][j] * 1.0e7)/1.0e7;
	        }
	    }
    }

    else
    {
	    #pragma omp parallel for collapse(2) private(i, j)
	    for (i = 0; i < num_nodes; ++i)
	    {
	        for (j = 0; j < dims; ++j)
	        {
            	nodes[i][j] = std::round(nodes[i][j] * 1.0e7)/1.0e7;
	        }
	    }
    }

	std::vector< std::vector< int > > output( num_elems, std::vector<int> ( nodes_per_elem, 0 ) );
	#pragma omp parallel for collapse(2) private(i, j, idx, imax)
	for (i = 0; i < num_elems; ++i)
	{
		for (j = 0; j < nodes_per_elem; ++j)
		{
			idx = 0;
			imax = num_nodes - 1;
			for (k = 0; k < dims; ++k)
			{
				idx = lower_bound(nodes, elemCoords[i][j][k], k, idx, imax);
				imax = upper_bound(nodes, elemCoords[i][j][k], k, idx, imax);
			}
			output[i][j] = idx;
		}
	}

	return output;
}

std::vector< int > replace_with_idx_int(std::vector< std::vector< int > > global_elems, std::vector< std::vector< int > > local_elems) {
    const unsigned int num_local_elems = local_elems.size();
    const unsigned int num_global_elems = global_elems.size();
    const unsigned int nodes_per_elem = local_elems[0].size();
    unsigned int i, j;
    int idx, imax;

    std::vector< int > output(num_local_elems, 0);

    if (num_local_elems > 1000)
    {
        #pragma omp parallel for collapse(1) private(i, j, idx, imax)
        for (i = 0; i < num_local_elems; ++i)
        {
            idx = 0;
            imax = num_global_elems - 1;
            for (j = 0; j < nodes_per_elem; ++j)
            {
                idx = lower_bound(global_elems, local_elems[i][j], j, idx, imax);
                imax = upper_bound(global_elems, local_elems[i][j], j, idx, imax);  
            }
            output[i] = idx;
        }
    }

    else {
        for (i = 0; i < num_local_elems; ++i)
        {
            idx = 0;
            imax = num_global_elems - 1;
            for (j = 0; j < nodes_per_elem; ++j)
            {
                idx = lower_bound(global_elems, local_elems[i][j], j, idx, imax);
                imax = upper_bound(global_elems, local_elems[i][j], j, idx, imax);  
            }
            output[i] = idx;
        }        
    }

    return output;
}

} // namespace cpp_utils