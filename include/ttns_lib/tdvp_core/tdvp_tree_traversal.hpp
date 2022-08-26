#ifndef HTUCKER_TDVP_TREE_TRAVERSAL_HPP
#define HTUCKER_TDVP_TREE_TRAVERSAL_HPP

#include <vector>

namespace ttns
{

class tdvp_tree_traversal
{
public:
    using size_type = std::size_t;

    using iterator = typename std::vector<size_type>::iterator;
    using const_iterator = typename std::vector<size_type>::const_iterator;
    using reverse_iterator = typename std::vector<size_type>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<size_type>::const_reverse_iterator;
protected:
    std::vector<size_type> m_traversal_order;
    std::vector<size_type> m_times_visited;

public:
    tdvp_tree_traversal() : m_traversal_order(), m_times_visited() {}
    
    template <typename T, typename backend>
    tdvp_tree_traversal(const httensor<T, backend>& A) 
    {   
        CALL_AND_HANDLE(resize(A), "Failed to construct tdvp_tree_traversal object.");
    }

    tdvp_tree_traversal(const tdvp_tree_traversal& t) = default;
    tdvp_tree_traversal(tdvp_tree_traversal&& t) = default;

    tdvp_tree_traversal& operator=(const tdvp_tree_traversal& t) = default;
    tdvp_tree_traversal& operator=(tdvp_tree_traversal&& t) = default;

    template <typename T, typename backend>
    void resize(const httensor<T, backend>& A)
    {
        try
        {
            //resize the number of times each node was visited array
            m_times_visited.resize(A.size()); 
            reset_times_visited();

            //resize the traversal order array
            size_type ntraversal_sites = 0;
            for(const auto& a : A){ntraversal_sites += a.is_leaf() ? 2 : (1 + a.size());}
            m_traversal_order.resize(ntraversal_sites);

            //now initialise the traversal order array
            const httensor_node<T, backend>* curr_node = &A.root();
            for(size_type i=0; i<ntraversal_sites; ++i)
            {
                size_type curr_node_id = curr_node->id();
                m_traversal_order[i] = curr_node_id;

                size_type times_visited = m_times_visited[curr_node_id];

                if(times_visited == 0 && curr_node->is_leaf()){}
                else if(times_visited < curr_node->size())
                {
                    CALL_AND_HANDLE
                    (
                        curr_node = curr_node->child_pointer(m_times_visited[curr_node_id]), 
                        "Failed to access a child of a node when constructing the traversal order array."
                    );
                }
                else if(!curr_node->is_root())
                {
                    CALL_AND_HANDLE
                    (
                        curr_node = curr_node->parent_pointer(), 
                        "Failed to access parent of node when constructing the traversal order array."
                    );
                }
                else if(i+1 != ntraversal_sites){RAISE_EXCEPTION("Critical Error: This condition should never be meet");}

                ++m_times_visited[curr_node_id];
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize tdvp_tree_traversal object.");
        }
    }

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_traversal_order.clear(), "Failed to clear traversal order array.");
            CALL_AND_HANDLE(m_times_visited.clear(), "Failed to clear times visited array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear tdvp_tree_traversal object.");
        }
    }
    
    const size_type& times_visited(size_type i) const
    {
        ASSERT(i < m_times_visited.size(), "Failed to access the number of times the node was visited.");
        return m_times_visited[i];
    }

    size_type& times_visited(size_type i)
    {
        ASSERT(i < m_times_visited.size(), "Failed to access the number of times the node was visited.");
        return m_times_visited[i];
    }

    void reset_times_visited()
    {
         CALL_AND_HANDLE(std::fill(std::begin(m_times_visited), std::end(m_times_visited), 0), "Failed to reset the number of times each node was visited.");
    }
public:
    //iterator functions
    iterator begin() {  return iterator(m_traversal_order.begin());  }
    iterator end() {  return iterator(m_traversal_order.end());  }
    const_iterator begin() const {  return const_iterator(m_traversal_order.begin());  }
    const_iterator end() const {  return const_iterator(m_traversal_order.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_traversal_order.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_traversal_order.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_traversal_order.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_traversal_order.rend());  }
};

}   //namespace ttns

#endif  //HTUCKER_TDVP_TREE_TRAVERSAL_HPP//
