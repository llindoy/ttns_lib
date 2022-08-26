///
/// @file ntree_factory.hpp
/// @author Lachlan Lindoy
/// @date 21/01/2021
/// @version 1.0
/// 
/// @brief File containing classes for building ntree object
/// 
///
#ifndef NTREE_FACTORY_HPP
#define NTREE_FACTORY_HPP

#include "ntree.hpp"

namespace ttns
{

template <typename T>
class ntree_factory
{
protected:
    using tree_type = ntree<T>;
    using node_type = typename tree_type::node_type;
public:
    //function for building a balanced tree with a N branches per level, vint.size()+1 levels and a fixed value (per level) at each internal node
    static tree_type build_balanced_tree(const std::vector<T>& rv, size_t Nmax, T r1, T ri, T root_value = T(1))
    {
        ASSERT(Nmax > 1, "Failed to create balanced tree.  This routine does not work with trees that do not branch.")
        ASSERT(rv.size() > 0, "Failed to create balanced tree. Cannot create tree with no leafs.");

        tree_type tree;  tree.insert(root_value);
        append_balanced_subtree(tree(), rv, Nmax, r1, ri);
        return tree;
    }   

    static void append_balanced_subtree(node_type& node, size_t Nleaves, size_t Nmax, T r1, T ri )
    {
        ASSERT(Nmax > 1, "Failed to append balanced subtree.  This routine does not work with trees that do not branch.")
        ASSERT(Nleaves > 0, "Failed to append balanced subtree. Cannot create tree with no leafs.");
        if(Nleaves < Nmax)
        {
            for(size_t i=0; i<Nleaves; ++i)
            {
                node.insert(r1);
            }
        }
        else
        {
            size_t r = Nleaves%Nmax;
            for(size_t i=0; i<Nmax; ++i)
            {
                size_t Nchild = Nleaves/Nmax + (i < r ? 1 : 0);
                node.insert(r1);
                balanced_subtree_internal(Nmax, ri, node.back(), Nchild);
            }
        }
    }


    static void append_balanced_subtree(node_type& node, const std::vector<T>& rv, size_t Nmax, T r1, T ri )
    {
        ASSERT(Nmax > 1, "Failed to append balanced subtree.  This routine does not work with trees that do not branch.")
        ASSERT(rv.size() > 0, "Failed to append balanced subtree. Cannot create tree with no leafs.");
        if(rv.size() < Nmax)
        {
            for(size_t i=0; i<rv.size(); ++i)
            {
                node.insert(r1);
                node.back().insert(rv[i]);
            }
        }
        else
        {
            size_t r = rv.size()%Nmax;
            size_t ninserted = 0;
            for(size_t i=0; i<Nmax; ++i)
            {
                size_t Nchild = rv.size()/Nmax + (i < r ? 1 : 0);
                node.insert(r1);
                ninserted = balanced_subtree_internal(rv, Nmax, ri, node.back(), Nchild, ninserted);
            }
        }
    }


    static void append_balanced_subtree_100000(node_type& node, const std::vector<T>& rv, size_t Nmax, T r1, size_t nlevels, T ri )
    {
        ASSERT(Nmax > 1, "Failed to append balanced subtree.  This routine does not work with trees that do not branch.")
        ASSERT(rv.size() > 0, "Failed to append balanced subtree. Cannot create tree with no leafs.");
        if(rv.size() < Nmax)
        {
            for(size_t i=0; i<rv.size(); ++i)
            {
                node.insert(r1);
                node.back().insert(rv[i]);
            }
        }
        else
        {
            size_t r = rv.size()%Nmax;
            size_t ninserted = 0;
            for(size_t i=0; i<Nmax; ++i)
            {
                size_t Nchild = rv.size()/Nmax + (i < r ? 1 : 0);
                node.insert(r1);
                ninserted = balanced_subtree_internal_100000(rv, Nmax, r1, ri, node.back(), Nchild, ninserted, 1, nlevels);
            }
        }
    }

    static tree_type build_degenerate_tree(const std::vector<T>& rv, size_t r, T root_value = T(1))
    {
        ASSERT(rv.size() > 0, "Failed to create degenerate tree. Cannot create tree with no leafs.");
        tree_type tree;  tree.insert(root_value);

        append_degenerate_subtree(tree(), rv, r);
        return tree;
    }

    static void append_degenerate_subtree(node_type& node, const std::vector<T>& rv, size_t r)
    {
        ASSERT(rv.size() > 0, "Failed to append degenerate sybtree. Cannot create tree with no leafs.");
        degenerate_subtree_internal(node, rv, r, 0);
    }

    static void append_balanced_subtree_wang_meyer(node_type& node, const std::vector<T>& rv, size_t Nmax1, size_t Nmax2, T r1, size_t nlevels, T ri )
    {
        ASSERT(Nmax1 > 1, "Failed to append balanced subtree.  This routine does not work with trees that do not branch.")
        ASSERT(Nmax2 > 1, "Failed to append balanced subtree.  This routine does not work with trees that do not branch.")
        ASSERT(rv.size() > 0, "Failed to append balanced subtree. Cannot create tree with no leafs.");
        if(rv.size() < Nmax1)
        {
            for(size_t i=0; i<rv.size(); ++i)
            {
                node.insert(r1);
                node.back().insert(rv[i]);
            }
        }
        else
        {
            size_t r = rv.size()%Nmax1;
            size_t ninserted = 0;
            for(size_t i=0; i<Nmax1; ++i)
            {
                size_t Nchild = rv.size()/Nmax1 + (i < r ? 1 : 0);
                node.insert(r1);
                ninserted = balanced_subtree_internal_100000(rv, Nmax2, r1, ri, node.back(), Nchild, ninserted, 1, nlevels);
            }
        }
    }
protected:
    static void balanced_subtree_internal(size_t Nmax, T ri, node_type& node, size_t nadd)
    {
        if(nadd < Nmax)
        {
            if(nadd == 1)
            {
                return ;
            }
            else
            {
                for(size_t i=0; i < nadd; ++i)  
                {
                    node.insert(ri);
                }
                return;
            }
        }
        else
        {
            size_t r = nadd%Nmax;
            for(size_t i = 0; i < Nmax; ++i)
            {
                size_t Nchild = nadd/Nmax + (i < r ? 1 : 0);
                node.insert(ri);
                balanced_subtree_internal(Nmax, ri, node.back(), Nchild);
            }
            return ;
        }       
    }

    static size_t balanced_subtree_internal(const std::vector<T>& rv, size_t Nmax, T ri, node_type& node, size_t nadd, size_t nskip)
    {
        if(nadd < Nmax)
        {
            if(nadd == 1)
            {
                node.insert(rv[nskip]);
                return nskip+1;
            }
            else
            {
                for(size_t i=0; i < nadd; ++i)  
                {
                    node.insert(ri);
                    node.back().insert(rv[nskip+i]);
                }
                return nskip + nadd;
            }
        }
        else
        {
            size_t r = nadd%Nmax;
            size_t ninserted = nskip;
            for(size_t i = 0; i < Nmax; ++i)
            {
                size_t Nchild = nadd/Nmax + (i < r ? 1 : 0);
                node.insert(ri);
                ninserted = balanced_subtree_internal(rv, Nmax, ri, node.back(), Nchild, ninserted);
            }
            return ninserted;
        }       
    }

    static size_t balanced_subtree_internal_100000(const std::vector<T>& rv, size_t Nmax, T ri, T rj, node_type& node, size_t nadd, size_t nskip, size_t level, size_t nlevels)
    {
        size_t ra = level < nlevels ? ri : rj;
        if(nadd < Nmax)
        {
            if(nadd == 1)
            {
                node.insert(rv[nskip]);
                return nskip+1;
            }
            else
            {
                for(size_t i=0; i < nadd; ++i)  
                {
                    node.insert(ra);
                    node.back().insert(rv[nskip+i]);
                }
                return nskip + nadd;
            }
        }
        else
        {
            size_t r = nadd%Nmax;
            size_t ninserted = nskip;
            for(size_t i = 0; i < Nmax; ++i)
            {
                size_t Nchild = nadd/Nmax + (i < r ? 1 : 0);
                node.insert(ra);
                ninserted = balanced_subtree_internal_100000(rv, Nmax, ri, rj, node.back(), Nchild, ninserted, level+1, nlevels);
            }
            return ninserted;
        }       
    }

    static void degenerate_subtree_internal(node_type& node, const std::vector<T>& rv, size_t r, size_t index)
    {
        if(index+1 < rv.size())
        {
            node.insert(r);
            node.back().insert(rv[index]);
            degenerate_subtree_internal(node.back(), rv, r, index+1);
        }
        else
        {
            node.insert(rv[index]);
        }
    }

public:
    static void sanitise_tree(tree_type& tree)
    {
        for(typename tree_type::post_iterator tree_iter = tree.post_begin(); tree_iter != tree.post_end(); ++tree_iter)
        {
            std::cerr << tree_iter->value() << std::endl;
            if(!(tree_iter->is_leaf()))
            {
                T size2 = 1;
                //iterate over both the children of the node and the array storing the node info and set the
                //number of primitive functions in the node info array 
                for(auto& topology_child : *tree_iter)
                {
                    size2 *= topology_child.value();
                }
                if(size2 > tree_iter->value()){tree_iter->value() = size2;}
            }
        }
    }
};

}   //namespace ttns

#endif

