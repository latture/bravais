%module cpp_utils

%{
#include "cpp_utils.h"
%}

%include "std_vector.i"
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(IntVector2D) vector<vector<int>>;
   %template(DoubleVector2D) vector<vector<double>>;
   %template(DoubleVector3D) vector<vector<vector<double>>>;
}

%include "cpp_utils.h"