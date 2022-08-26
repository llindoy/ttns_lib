///
/// @file ntree_builder.hpp
/// @author Lachlan Lindoy
/// @date 21/01/2021
/// @version 1.0
/// 
/// @brief File containing classes for building ntree object
/// 
///
#ifndef NTREE_BUILDER_HPP
#define NTREE_BUILDER_HPP

#include "ntree.hpp"
#include <vector>

namespace ttns
{

template <typename T>
class ntree_builder
{
public:
    using tree_type = ntree<T>;
    using node_type = typename tree_type::node_type;
    using size_type = typename tree_type::size_type;
    using leaf_index = std::vector<std::vector<size_type>>;
public:
    /*
     *  Functions for constructing balanced N-ary trees with values specified either by a function of the level or as a constant value.
     *  These functions also return a vector containing vectors indexing the leaf indice that allows for easy addition of nodes to the leaves
     *  of this tree.
     */
    template <typename Func>
    static tree_type balanced_tree(size_type Nleaves, size_type degree, Func&& fl, std::vector<std::vector<size_type>>& linds)
    {
        ASSERT(degree > 1, "Failed to construct a balanced tree that does not branch.");
        ASSERT(Nleaves > 0, "Cannot construct a tree with zero leaves.");
        if(linds.size() != Nleaves){linds.resize(Nleaves);}
    
        //reserve the storage for each of the index arrays.  The maximum length of any of the index arrays is ceil(log_degree(Nleaves))
        double ceil_log = std::ceil(std::log(Nleaves)/std::log(degree));
        size_type depth = static_cast<size_type>(ceil_log);
        for(size_type i=0; i < Nleaves; ++i)
        {
            linds.reserve(depth+1);
        }

        tree_type tree; tree.insert(T(1));
        if(Nleaves != 1)
        {
            balanced_subtree(tree(), Nleaves, degree, std::forward<Func>(fl), linds, false);
        }
        return tree;
    }

    /*
     *  Functions for constructing either a balanced degree-ary subtree or degenerate subtree with a root given by root that is specified in some already defined treein a tree that has already 
     */
    template <typename Func>
    static void balanced_subtree(node_type& root, size_type Nleaves, size_type degree, Func&& fl, std::vector<std::vector<size_type>>& linds, bool allocate = true)
    {
        ASSERT(degree > 1, "Failed to append balanced subtree.  This routine does not work with trees that do not branch.")
        ASSERT(Nleaves > 0, "Failed to append balanced subtree. Cannot create tree with no leafs.");

        if(allocate)
        {
            if(linds.size() != Nleaves){linds.resize(Nleaves);}
            double ceil_log = std::ceil(std::log(Nleaves)/std::log(degree));
            size_type depth = static_cast<size_type>(ceil_log);
            for(size_type i=0; i < Nleaves; ++i)
            {
                linds.reserve(depth);
            }
        }

        if(Nleaves < degree)
        {
            for(size_t i=0; i<Nleaves; ++i)
            {
                linds[i].resize(1);
                linds[i][0] = root.size();
                root.insert(evaluate_value(fl,0));
            }
        }
        else
        {
            size_t r = Nleaves%degree;
            size_type count = 0;
            for(size_t i=0; i<degree; ++i)
            {
                size_t Nchild = Nleaves/degree + (i < r ? 1 : 0);
                for(size_type j=0; j<Nchild; ++j)
                {
                    linds[count+j].push_back(root.size());
                }
                root.insert(evaluate_value(fl,0));
                balanced_subtree_internal(degree, root.back(), Nchild, std::forward<Func>(fl), linds, 1, count);
                count+=Nchild;
            }
        }
    }
protected:
    template <typename Func>
    static void balanced_subtree_internal(size_t degree, node_type& node, size_t nadd, Func&& fl, std::vector<std::vector<size_type>>& linds,  size_type level, size_type count)
    {
        if(nadd < degree)
        {
            if(nadd == 1)
            {
                return ;
            }
            else
            {
                for(size_t i=0; i < nadd; ++i)  
                {
                    linds[count+i].push_back(node.size());
                    node.insert(evaluate_value(fl,level));
                }
                return;
            }
        }
        else
        {
            size_t r = nadd%degree;
            for(size_t i = 0; i < degree; ++i)
            {
                size_t Nchild = nadd/degree + (i < r ? 1 : 0);
                for(size_type j=0; j<Nchild; ++j)
                {
                    linds[count+j].push_back(node.size());
                }
                node.insert(evaluate_value(fl,level));
                balanced_subtree_internal(degree, node.back(), Nchild, std::forward<Func>(fl), linds, level+1, count);
                count+=Nchild;
            }
            return ;
        }       
    }

protected:
    template <typename F>
    static inline typename std::enable_if<std::is_convertible<F,T>::value, T>::type evaluate_value(F t, size_type/* l */)
    {
        return t;
    }
    template <typename F>
    static inline typename std::enable_if<!std::is_convertible<F,T>::value, T>::type evaluate_value(F t, size_type l)
    {
        return t(l);
    }

public:
    /*
     *  Functions fo constructing degenerate trees with values in the tree specified by a function of the level or as a constant value.  
     */
    template <typename Func>
    static tree_type degenerate_tree(size_type Nnodes, Func&& fl, std::vector<std::vector<size_type>>& linds)
    {
        ASSERT(Nnodes > 0, "Cannot construct a tree with zero leaves.");

        tree_type tree; tree.insert(T(1));
        linds.resize(Nnodes);
        for(size_type i = 0; i < Nnodes; ++i)
        {
            linds[i].resize(i);
            for(size_type j=0; j < linds[i].size(); ++j)
            {
                linds[i][j] = 0;
            }
        }
        if(Nnodes != 1)
        {        
            degenerate_subtree(tree(), Nnodes-1, std::forward<Func>(fl), linds, false);
        }
        return tree;
    }

    template <typename Func>
    static void degenerate_subtree(node_type& node, size_type Nnodes, Func&& fl, std::vector<std::vector<size_type>>& linds, bool allocate = true)
    {
        ASSERT(Nnodes > 0, "Failed to append degenerate sybtree. Cannot create tree with no leafs.");
        if(allocate)
        {
            linds.resize(Nnodes);
            for(size_type i = 0; i < Nnodes; ++i)
            {
                linds[i].resize(i+1);
                linds[i][0] = node.size();
                for(size_type j=1; j < linds[i].size(); ++j)
                {
                    linds[i][j] = 0;
                }
            }
        }

        degenerate_subtree_internal(node, Nnodes, std::forward<Func>(fl), 0);
    }

protected:
    template <typename Func>
    static void degenerate_subtree_internal(node_type& node, size_type Nnodes, Func&& fl, size_type level)
    {
        if(level+1 < Nnodes)
        {
            node.insert(evaluate_value(fl, level));
            degenerate_subtree_internal(node.back(), Nnodes, std::forward<Func>(fl), level+1);
        }
        else
        {
            node.insert(evaluate_value(fl, level));
        }
    }

public:
    static void sanitise_tree(tree_type& tree)
    {
        //we need to correctly implement the post_order_iterator here
        for(typename tree_type::post_iterator tree_iter = tree.post_begin(); tree_iter != tree.post_end(); ++tree_iter)
        {
            if(!(tree_iter->is_leaf()))
            {
                T size2 = 1;
                //iterate over all of the children of the node and calcaulte the product of their values.  If the 
                //product of their values is less than the value stored at the current node we set the value at the 
                //current node to their product.
                for(auto& topology_child : *tree_iter)
                {
                    size2 *= topology_child.value();
                }
                if(size2 < tree_iter->value()){tree_iter->value() = size2;}
            }
        }
    }

public:
    template <typename Func>
    static tree_type htucker_tree(const std::vector<T>& Hb, size_type degree, Func&& fl)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        tree_type ret;
        CALL_AND_HANDLE(ret = balanced_tree(Nleaves, degree, std::forward<Func>(fl), linds), "Failed to build balanced tree.");
        
        for(size_type i = 0; i < linds.size(); ++i)
        {
            ret.at(linds[i]).insert(Hb[i]);
        }
        return ret;
    }

    template <typename Func>
    static void htucker_subtree(node_type& root, const std::vector<T>& Hb, size_type degree, Func&& fl)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        CALL_AND_HANDLE(balanced_subtree(root, Nleaves, degree, std::forward<Func>(fl), linds), "Failed to build balanced sub tree.");
        for(size_type i = 0; i < linds.size(); ++i)
        {
            root.at(linds[i]).insert(Hb[i]);
        }
    }

    template <typename Func, typename Func2>
    static tree_type mps_tree(const std::vector<T>& Hb, Func&& fl, Func2&& f2)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        tree_type ret;
        CALL_AND_HANDLE(ret = degenerate_tree(Nleaves, std::forward<Func>(fl), linds), "Failed to build balanced tree.");
        
        for(size_type i = 0; i < linds.size(); ++i)
        {
            size_type ind  = ret.at(linds[i]).insert(evaluate_value(f2, i));
            ret.at(linds[i])[ind].insert(Hb[i]);
        }
        return ret;
    }

    template <typename Func, typename Func2>
    static void mps_subtree(node_type& root, const std::vector<T>& Hb, Func&& fl, Func2&& f2)
    {
        size_type Nleaves = Hb.size();
        std::vector<std::vector<size_type>> linds(Nleaves);
        CALL_AND_HANDLE(degenerate_subtree(root, Nleaves, std::forward<Func>(fl), linds), "Failed to build balanced sub tree.");
        for(size_type i = 0; i < linds.size(); ++i)
        {
            size_type ind  = root.at(linds[i]).insert(evaluate_value(f2, i));
            root.at(linds[i])[ind].insert(Hb[i]);
        }
    }
};

}   //namespace ttns

#endif

