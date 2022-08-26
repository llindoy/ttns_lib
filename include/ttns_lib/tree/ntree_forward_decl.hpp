///
/// @file ntree_forward_decl.hpp
/// @author Lachlan Lindoy
/// @date 14/08/2018
/// @version 1.0
/// 
/// @brief Interfaces for the ntree class used for constructing the topology of the multilayer multiconfiguration time-depedent hartree wavefunction
/// 
/// This file contains the definitions of the ntree required for setting up the hierarchy of the ml-mctdh wavefunction.  This is a general purpose tree 
/// implementation which supports an arbitrary number of children per node.  
///

#ifndef HTUCKER_DATASTRUCTURES_TREE_FORWARD_DECL_HPP
#define HTUCKER_DATASTRUCTURES_TREE_FORWARD_DECL_HPP

#include <linalg/utils/exception_handling.hpp>
#include "../utils/tmp_funcs.hpp"

namespace ttns
{

template <typename T, typename Alloc> class ntree;
template <typename Tree> class ntree_node;

}   //namespace ttns

#endif  //  HTUCKER_DATASTRUCTURES_NTREE_FORWARD_DECL_HPP    //

