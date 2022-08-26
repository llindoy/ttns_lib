#ifndef HTUCKER_TREE_FORWARD_DECL_HPP
#define HTUCKER_TREE_FORWARD_DECL_HPP

namespace ttns
{

class tree_node_tag{};
class tree_tag{};

//metaprogram tag to determine if object is tree type
template <typename T>  using is_tree = std::is_base_of<tree_tag, T>;

//metaprogram tag to determine if object is tree type
template <typename T>  using is_tree_node = std::is_base_of<tree_node_tag, T>;

template <typename T>class tree_base;
template <typename T> class tree;
template <typename Tree> class tree_node_base;
template <typename Tree> class tree_node;


}   //namespace ttns
#endif  //  HTUCKER_TREE_FORWARD_DECL_HPP    //
