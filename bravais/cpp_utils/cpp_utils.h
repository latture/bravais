#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <vector>
#include <set>
#include <numeric>
#include <functional>
#include <iostream>
#include <cmath>

namespace cpp_utils {

/**
 * @brief Delete duplicate rows of a 2D vector of ints.
 * 
 * @param input `std::vector<std::vector<int>>`. Vector to process for duplicates.
 * @return 2D vector with duplicate rows removed.
 */
std::vector< std::vector < int > > delete_duplicates_int(const std::vector< std::vector < int > > &input);

/**
 * @brief Delete duplicate rows of a 2D vector of doubles.
 * 
 * @param input `std::vector<std::vector<double>>`. Vector to process for duplicates.
 * @return 2D vector with duplicate rows removed.
 */
std::vector< std::vector < double > > delete_duplicates_dbl(std::vector< std::vector < double > > input);

/**
 * @brief Searchs the entries of the input 2D vector for which the row is equal to the input parameter row.
 * 
 * @param row `std::vector<double>`. The row to search for.
 * @param vector2D `std::vector<std::vector<double>>`. The vector to search for `row`.
 * @return Indices. `std::vector<int>`. Vector of indices where the input `row` matched rows in `vector2D`.
 */
std::vector< int > test_rows(const std::vector< double > &row, const std::vector< std::vector< double > > &vector2D);

/**
 * @brief Replaces nodal positions of `elemCoords` with the index that the coordinate occurs in `nodes`.
 * @details This assumes that the position in `elemCoords` exists in `nodes` and that there are no 
 * duplicate entries in `nodes`. In addition, `nodes` must be sorted.
 * 
 * @param nodes. `std::vector<std::vector<double>>`. Vector of nodal positions of the form: 
 *  \f$ \left[ \left[ x_{1}, y_{1}, z{1} \right], \left[ x_{2}, y_{2}, z{2} \right], \left[ x_{3}, y_{3}, z{3} \right] \ldots \right] \f$.
 *  Rows are not required to be three dimensional: any number of columns per row are valid as long as the number of columns is consistent
 *  between all rows.
 * @param elemCoords `std::vector<std::vector<std::vector<double>>>`. 3D vector of nodal positions of the form:
 *  \f$ \left[ \left[ \left[ x_{11}, y_{11}, z{11} \right], \left[ x_{21}, y_{21}, z{21} \right] \right],
 *             \left[ \left[ x_{12}, y_{12}, z{12} \right], \left[ x_{22}, y_{22}, z{22} \right] \right], 
 *             \left[ \left[ x_{13}, y_{13}, z{13} \right], \left[ x_{23}, y_{23}, z{23} \right] \right], 
 *             \ldots \right] \f$.
 * 
 * @return Element indices. `std::vector<std::vector<int>>` Two-dimensional vector where the rows contain the indices that are connected to form 1 element.
 * Indices reference the position in a list of nodal coordinates.
 */
std::vector< std::vector< int > > replace_with_idx(std::vector< std::vector< double > > nodes, std::vector< std::vector< std::vector< double > > > elemCoords);


/**
 * @brief Replaces local element list, `local_elems` with the indices that element occurs in `global_elems`.
 * @details This assumes that the elements in `local_elems` exist in `global_elems` and that there are no 
 * duplicate entries in `global_elems`. In addition, `global_elems` must be sorted.
 * 
 * @param global_elems. `std::vector<std::vector<int>>`. Vector of nodal positions of the form: 
 *  \f$ \left[ \left[ i_{11}, i_{12} \right], \left[ i_{21}, i_{22} \right], \left[ i_{31}, i_{32} \right] \ldots \right] \f$.
 *  Rows are not required to have only 2 entries: any number of columns per row are valid as long as the number of columns is consistent
 *  between all rows. Here, \f$ i_{11}\f$ is an integer which represents the nodal index that the current element contains.
 * @param local_elems. `std::vector<std::vector<int>>`. Vector of nodal positions of the form: 
 *  \f$ \left[ \left[ j_{11}, j_{12} \right], \left[ j_{21}, j_{22} \right], \left[ j_{31}, j_{32} \right] \ldots \right] \f$.
 *  Rows are not required to have only 2 entries: any number of columns per row are valid as long as the number of columns is consistent
 *  between all rows. Here, \f$ j_{11}\f$ is an integer which represents the nodal index that the current element contains. When searching
 *  for a match in `local_elems` \f$ i_{kl}=j{kl}\f$ for a given \f$ k \f$ and all \f$ l \f$, where \f$ l \f$ is varied over the number
 *  of columns.
 * 
 * @return Indices. `std::vector<std::vector<int>>` Vector containing the indices where the rows of `local_elems` occur in `global_elems`.
 */
std::vector< int > replace_with_idx_int(std::vector< std::vector< int > > global_elems, std::vector< std::vector< int > > local_elems);

} // namespace cpp_utils

#endif //CPP_UTILS_H
