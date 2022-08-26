///
/// @file ntree.hpp
/// @author Lachlan Lindoy
/// @date 14/08/2018
/// @version 1.0
/// 
/// @brief Interfaces for the ntree class used for constructing the topology of the multilayer multiconfiguration time-depedent hartree wavefunction
/// 
/// This file contains the definitions of the ntree required for setting up the hierarchy of the ml-mctdh wavefunction.  This is a general purpose tree 
/// implementation which supports an arbitrary number of children per node.  
///

#ifndef HTUCKER_DATASTRUCTURES_NTREE_HPP
#define HTUCKER_DATASTRUCTURES_NTREE_HPP

#include "ntree_forward_decl.hpp"
#include "ntree_iterator.hpp"
#include "ntree_node.hpp"

#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <type_traits>
#include <vector>
#include <list>

namespace ttns
{

template <typename T, typename Alloc = std::allocator<T> > 
class ntree 
{
public:
    using size_type = typename Alloc::size_type;
    using difference_type = typename Alloc::difference_type;

    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;

    using tree_type = ntree<T, Alloc>;
    using tree_reference = tree_type&;
    using const_tree_reference = const tree_type&;

    using node_type = ntree_node<tree_type>;
    using node_reference = node_type&;
    using const_node_reference = const node_type&;

    using allocator_type = Alloc;
    using node_allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<node_type>;
    using node_pointer_allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<node_type*>;

    using bfs_iterator = ntree_bfs_iterator<node_type, false, Alloc>;
    using const_bfs_iterator = ntree_bfs_iterator<node_type, true, Alloc>;

    using dfs_iterator = ntree_dfs_preorder_iterator<node_type, false, Alloc>;
    using const_dfs_iterator = ntree_dfs_preorder_iterator<node_type, true, Alloc>;

    using post_iterator = ntree_dfs_postorder_iterator<node_type, false, Alloc>;
    using const_post_iterator = ntree_dfs_postorder_iterator<node_type, true, Alloc>;

    using pre_post_iterator = ntree_dfs_pre_post_iterator<node_type, false, Alloc>;
    using const_pre_post_iterator = ntree_dfs_pre_post_iterator<node_type, true, Alloc>;

    using leaf_iterator = ntree_leaf_iterator<node_type, false, Alloc>;
    using const_leaf_iterator = ntree_leaf_iterator<node_type, true, Alloc>;

    using iterator = dfs_iterator;
    using const_iterator = const_dfs_iterator;

    friend class ntree_node<tree_type>;
private:
    node_type* m_root;
    node_allocator_type m_allocator;

public:
    ntree() : m_root(nullptr), m_allocator() {}
    ntree(const node_allocator_type& alloc) : m_root(nullptr), m_allocator(alloc) {}
    ntree(const ntree& tree) : m_root(nullptr), m_allocator(tree.m_allocator) { *this = tree; }
    ~ntree(){clear();}

    ntree& operator=(const ntree& src)
    {
        if(&src == this){return *this;}
        if(src.empty()){clear();    return *this;}

        if(empty())
        {
            m_root = create_node();
            m_root->m_tree = this;
            m_root->m_parent = nullptr;
        }
        *m_root = *src.m_root;
        
        return *this;
    }

    bool empty() const {return m_root == nullptr;}

    size_type nleaves() const
    {
        if(empty()){return 0;}
        return m_root->nleaves();
    }
    
    size_type size() const 
    {
        if(empty()){return 0;}
        return m_root->subtree_size();
    }

    node_type& operator()()
    {
        ASSERT(!empty(), "Failed to access root node of tree.  An empty tree has no root.");
        return *m_root;
    }

    const node_type& operator()() const
    {
        ASSERT(!empty(), "Failed to access root node of tree.  An empty tree has no root.");
        return *m_root;
    }

    node_type& operator[](size_type n)
    {
        CALL_AND_HANDLE(return this->operator()()[n], "Failed to access child of root.");
    }
    const node_type& operator[](size_type n) const
    {
        CALL_AND_HANDLE(return this->operator()()[n], "Failed to access child of root");
    }

    node_type& at(const std::vector<size_type>& inds)
    {
        if(inds.size() == 0){CALL_AND_HANDLE(return this->operator()(), "Failed to access root node in the tree.")};
        CALL_AND_HANDLE(return this->operator()().at(inds, 0), "Failed to access node in the tree.");
    }

    const node_type& at(const std::vector<size_type>& inds) const
    {
        if(inds.size() == 0){CALL_AND_HANDLE(return this->operator()(), "Failed to access root node in the tree.")};
        CALL_AND_HANDLE(return this->operator()().at(inds, 0), "Failed to access node in the tree.");
    }

    node_type& root() 
    {
        ASSERT(!empty(), "Failed to access root node of tree.  An empty tree has no root.");
        return *m_root;
    }

    const node_type& root() const
    {
        ASSERT(!empty(), "Failed to access root node of tree.  An empty tree has no root.");
        return *m_root;
    }
    
    void insert(const value_type& data = value_type())  
    {
        clear();
        m_root = create_node();
        m_root->m_data = data;
        m_root->m_parent = nullptr;
        m_root->m_tree = this;
    }

    void clear()
    {
        if(empty()){return;}
        m_root->clear();
        if(m_root != nullptr)
        {
            destroy_node(m_root);
            m_root = nullptr;
        }
    }

public:
    iterator begin() {  return iterator(m_root);  }
    iterator end() {  return iterator();  }
    const_iterator begin() const {  return const_iterator(m_root);  }
    const_iterator end() const {  return const_iterator();  }

    dfs_iterator dfs_begin() {  return dfs_iterator(m_root);  }
    dfs_iterator dfs_end() {  return dfs_iterator();  }
    const_dfs_iterator dfs_begin() const {  return const_dfs_iterator(m_root);  }
    const_dfs_iterator dfs_end() const {  return const_dfs_iterator();  }

    post_iterator post_begin() {  return post_iterator(m_root);  }
    post_iterator post_end() {  return post_iterator();  }
    const_post_iterator post_begin() const {  return const_post_iterator(m_root);  }
    const_post_iterator post_end() const {  return const_post_iterator();  }

    pre_post_iterator pre_post_begin() {  return pre_post_iterator(m_root);  }
    pre_post_iterator pre_post_end() {  return pre_post_iterator();  }
    const_pre_post_iterator pre_post_begin() const {  return const_pre_post_iterator(m_root);  }
    const_pre_post_iterator pre_post_end() const {  return const_pre_post_iterator();  }

    bfs_iterator bfs_begin() {  return bfs_iterator(m_root);  }
    bfs_iterator bfs_end() {  return bfs_iterator();  }
    const_bfs_iterator bfs_begin() const {  return const_bfs_iterator(m_root);  }
    const_bfs_iterator bfs_end() const {  return const_bfs_iterator();  }

    leaf_iterator leaf_begin() {return leaf_iterator(m_root);}
    leaf_iterator leaf_end() {return leaf_iterator();}
    const_leaf_iterator leaf_begin() const {  return const_leaf_iterator(m_root);  }
    const_leaf_iterator leaf_end() const {  return const_leaf_iterator();  }

protected: 
    node_type* create_node() 
    {
        node_type* n = std::allocator_traits<node_allocator_type>::allocate(m_allocator, 1); 
        std::allocator_traits<node_allocator_type>::construct(m_allocator, n);
        return n;
    }

    void destroy_node(node_type* n)
    {
        std::allocator_traits<node_allocator_type>::destroy(m_allocator, n);
        std::allocator_traits<node_allocator_type>::deallocate(m_allocator, n, 1);
    }
};  //class ntree


template <typename T> 
std::ostream& operator<<(std::ostream& os, const ntree<T>& t){os << "ntree : " << t() << ";" << std::endl;return os;}

}   //namespace ttns
#endif  //  HTUCKER_DATASTRUCTURES_NTREE_HPP    //

