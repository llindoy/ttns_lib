#ifndef HTUCKER_TENSOR_HPP
#define HTUCKER_TENSOR_HPP

#include "ttn_nodes/httensor_node.hpp"
#include <tmp_funcs.hpp>

namespace ttns
{

template <typename T, typename backend = blas_backend>
class httensor
    : public tree<httensor_node_data<T, backend> > 
{
public:
    static_assert(is_number<T>::value, "The first template argument to the httensor object must be a valid number type.");
    static_assert(is_valid_backend<backend>::value, "The second template argument to the httensor object must be a valid backend.");

    using matrix_type = matrix<T, backend>;
    using base_type = tree<httensor_node_data<T, backend> >;

    using value_type = matrix_type;
    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using size_type = typename backend::size_type;

    using tree_type = typename base_type::tree_type;
    using tree_reference = typename base_type::tree_reference;
    using const_tree_reference = typename base_type::const_tree_reference;

    using node_type = typename base_type::node_type;

private:
    //provide access to base class operators
    using base_type::m_nodes;
    using base_type::m_nleaves;

    std::vector<size_type> m_dim_sizes;
    std::vector<size_type> m_leaf_indices;
public:
    using base_type::size;
    using base_type::clear;

public:
    httensor() : base_type(), m_dim_sizes() {}
    template <typename U> httensor(const httensor<U, backend>& other) : base_type(other), m_dim_sizes(other.mode_dimensions()), m_leaf_indices(other.m_leaf_indices) {}

    template <typename INTEGER, typename Alloc, typename = typename std::enable_if<std::is_integral<INTEGER>::value, void>::type >
    httensor(const ntree<INTEGER, Alloc>& topology){CALL_AND_HANDLE(construct_topology(topology), "Failed to construct the hierarchical tucker tensor object.  Failed to allocate tree structure from topology ntree.");}

    template <typename INTEGER, typename Alloc, typename = typename std::enable_if<std::is_integral<INTEGER>::value, void>::type >
    httensor(const ntree<INTEGER, Alloc>& topology, const ntree<INTEGER, Alloc>& capacity){CALL_AND_HANDLE(construct_topology(topology, capacity), "Failed to construct the hierarchical tucker tensor object.  Failed to allocate tree structure from topology ntree.");}


    template <typename U, typename be>
    httensor& operator=(const httensor<U, be>& other)
    {
        CALL_AND_RETHROW(base_type::operator=(other)); 
        m_dim_sizes = other.mode_dimensions(); 
        m_leaf_indices = other.leaf_indices();
        return *this;
    }

    template <typename INTEGER, typename Alloc, typename = typename std::enable_if<std::is_integral<INTEGER>::value, void>::type >
    void resize(const ntree<INTEGER, Alloc>& topology)
    {
        CALL_AND_HANDLE(clear(), "Failed to resize hierarchical tucker tensor object.  Failed to clear currently allocated data.");
        CALL_AND_HANDLE(construct_topology(topology), "Failed to resize the hierarchical tucker tensor object.  Failed to allocate tree structure from topology ntree.");
    }


    template <typename INTEGER, typename Alloc, typename = typename std::enable_if<std::is_integral<INTEGER>::value, void>::type >
    void resize(const ntree<INTEGER, Alloc>& topology, const ntree<INTEGER, Alloc>& capacity)
    {
        CALL_AND_HANDLE(clear(), "Failed to resize hierarchical tucker tensor object.  Failed to clear currently allocated data.");
        CALL_AND_HANDLE(construct_topology(topology, capacity), "Failed to resize the hierarchical tucker tensor object.  Failed to allocate tree structure from topology ntree.");
    }

    void zero(){for(auto& ch : m_nodes){ch.zero();}}

    const std::vector<size_type>& mode_dimensions() const{return m_dim_sizes;}
    size_type dim(size_type i) const{return m_dim_sizes[i];}
    size_type nmodes() const noexcept {return m_dim_sizes.size();}

    size_type nelems() const 
    {
        if(m_dim_sizes.size() == 0){return 0;}
        size_type nelems = 1;
        for(size_type i=0; i<m_dim_sizes.size(); ++i){nelems *= m_dim_sizes[i];}
        return nelems;
    }

    bool is_orthogonalised() const{return m_nodes[0].is_orthogonalised();}
    void set_is_orthogonalised(bool is_orthog = true){for(auto& ch : m_nodes){ch.set_is_orthogonalised(is_orthog);}}

    size_type get_leaf_index(size_type lid)
    {
        ASSERT(lid < m_nleaves, "Invalid leaf index.");
        return m_leaf_indices[lid];
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<tree<httensor_node_data<T, backend> >>(this)), "Failed to serialise httensor object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode_dimensions", m_dim_sizes)), "Failed to serialise httensor object.  Failed to serialise its mode_dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("leaf_indices", m_leaf_indices)), "Failed to serialise httensor object.  Failed to serialise its leaf_indices.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<tree<httensor_node_data<T, backend> >>(this)), "Failed to serialise httensor object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode_dimensions", m_dim_sizes)), "Failed to serialise httensor object.  Failed to serialise its mode_dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("leaf_indices", m_leaf_indices)), "Failed to serialise httensor object.  Failed to serialise its leaf_indices.");
    }
#endif

protected:    
    template <typename INTEGER, typename Alloc, typename = typename std::enable_if<std::is_integral<INTEGER>::value, void>::type >
    void construct_topology(const ntree<INTEGER, Alloc>& _tree)
    {
        CALL_AND_RETHROW(construct_topology(_tree, _tree));
    }


    template <typename INTEGER, typename Alloc, typename = typename std::enable_if<std::is_integral<INTEGER>::value, void>::type >
    void construct_topology(const ntree<INTEGER, Alloc>& _tree, const ntree<INTEGER, Alloc>& capacity)
    {
        ASSERT(_tree.size() > 1, "Failed to build hierarchical tucker tensor from topology tree.  The input topology must contain at least 2 elements.");
        ASSERT(_tree.size() == capacity.size(), "Failed to construct httensor topology with capacity.");

        //otherwise if the topology tree contains 2 elements we are considering the full problem
        if(_tree.size() == 2)
        {
            m_nleaves = 1;
            size_type required_size = 1;
            if(m_nodes.size() != required_size){m_nodes.resize(required_size);}

            m_nodes[0].m_children.resize(0);
            m_nodes[0].m_data.m_mode_dims.resize(1);
            m_nodes[0].m_data.m_mode_dims[0] = _tree()[0]();
            m_nodes[0].m_data.m_orthogonalised = false;
    
            m_nodes[0].m_id = 0;
            m_nodes[0].m_child_id = 0;
            m_nodes[0].m_level = 0;

            ASSERT(_tree()() <= _tree()[0](), "Failed to construct the hierarchical tucker tensor.  No single subtensor in the hierarchical tucker tensor can have a larger hierarchical rank than it has basis functions.");

            m_nodes[0].m_data.as_matrix().reallocate(capacity()[0](), capacity()());
            m_nodes[0].m_data.as_matrix().resize(_tree()[0](), _tree()());
            m_nodes[0].m_parent = nullptr;
            m_dim_sizes.resize(1);
            m_dim_sizes[0] = _tree()[0]();
            m_leaf_indices.resize(1);   m_leaf_indices[0] = 0;
        }

        //otherwise if the topology tree contains more than 2 more elements we will attempt to interpret it as a (hierarchical) tucker tensor
        //and solve the problem in this space.
        else if(_tree.size() > 2)
        {
            m_nleaves = _tree.nleaves();
            m_dim_sizes.resize(m_nleaves);
            m_leaf_indices.resize(m_nleaves);
            //resize the m_nodes array to the correct size
            size_type required_size = _tree.size() - _tree.nleaves();
            if(m_nodes.size() != required_size){m_nodes.resize(required_size);}

            //now we can begin building the tree
            size_type count = 0;
            size_type leaf_counter = 0; 
            auto this_it = m_nodes.begin();
            typename ntree<INTEGER, Alloc>::const_iterator capacity_iter = capacity.begin();
            for(typename ntree<INTEGER, Alloc>::const_iterator tree_iter = _tree.begin(); tree_iter != _tree.end(); ++tree_iter, ++capacity_iter)
            {
                ASSERT(capacity_iter != capacity.end(), "The capacity and tree iter objects are not the same size.");
                ASSERT(tree_iter->value() <= capacity_iter->value(), "Failed to construct httensor object.  The capacity is less than the size.");

                ASSERT(tree_iter->is_leaf() == capacity_iter->is_leaf(), "The capacity and topology trees do not have the same structure.");
                //we skip the leaves of the tree object as those are used to specify the topology
                //of the layer above but themselves do not correspond to a node of the mlmctdh topology
                if(!tree_iter->is_leaf())
                {
                    if(tree_iter->size() == 1)
                    {
                        if(tree_iter->operator[](0).is_leaf())
                        {
                            this_it->set_leaf_index(leaf_counter);
                            m_leaf_indices[leaf_counter] = count;
                        }
                        else
                        {
                            this_it->set_leaf_index(m_nleaves);
                        }
                    }
                    else
                    {
                        this_it->set_leaf_index(m_nleaves);
                    }

                    //now we determine the number of children of tree_iter that are not leaves 
                    //if the node has a child which is a leaf that must be its only child
                    //as otherwise this size_tree does not represent a valid topology
                    size_type nchildren = 0;
                    size_type _nleaves = 0;

                    for(auto child_it = tree_iter->begin(); child_it != tree_iter->end(); ++child_it)
                    {
                        if(!child_it->is_leaf()){++nchildren;}
                        else{++_nleaves;}
                    }

                    size_type ncapacity_children = 0;
                    size_type capacity_nleaves = 0;
                    for(auto child_it = capacity_iter->begin(); child_it != capacity_iter->end(); ++child_it)
                    {
                        if(!child_it->is_leaf()){++ncapacity_children;}
                        else{++capacity_nleaves;}
                    }
                    ASSERT(ncapacity_children == nchildren && capacity_nleaves == _nleaves, "The capacity and topology trees do not have the same structure.");

                    //now we can allocate the children array associated with this node
                    this_it->m_children.resize(nchildren);
                    this_it->m_data.m_mode_dims.resize(nchildren+_nleaves);
                    this_it->m_data.m_mode_capacity.resize(nchildren+_nleaves);
                    this_it->m_id = count;
                    this_it->m_level = tree_iter->level();  

                    ++count;


                    //first we need to go ahead and determine the size of the transfer or basis matrix corresponding to the current node.
                    size_type size1 = tree_iter->value();
                    size_type size2 = 1;

                    //iterate over both the children of the node and the array storing the node info and set the
                    //number of primitive functions in the node info array 
                    size_type index2 = 0;
                    for(auto& topology_child : *tree_iter)
                    {
                        size2 *= topology_child.value();
                        this_it->m_data.m_mode_dims[index2] = topology_child.value();
                        ++index2;
                    }

                    size_type capacity1 = capacity_iter->value();
                    size_type capacity2 = 1;
                    index2=0;
                    for(auto& capacity_child : *capacity_iter)
                    {
                        capacity2 *= capacity_child.value();
                        this_it->m_data.m_mode_capacity[index2] = capacity_child.value();
                        ++index2;
                    }
                    this_it->m_data.m_max_hrank = capacity1;
                    this_it->m_data.m_max_dimen = capacity2;

                    ASSERT(size1 <= size2, "Failed to construct the hierarchical tucker tensor.  No single subtensor in the hierarchical tucker tensor can have a larger hierarchical rank than it has basis functions.");

                    this_it->m_data.as_matrix().reallocate(capacity2, capacity1);
                    this_it->m_data.as_matrix().resize(size2, size1);
                    this_it->m_data.fill_zeros();
                    this_it->m_data.m_orthogonalised = false;
                    
                    //now we set up the children pointers
                    size_type index3 = 1;

                    for(size_type child_index = 0; child_index < nchildren; ++child_index)
                    {
                        this_it->m_children[child_index] = &(*(this_it + index3));
                        this_it->m_children[child_index]->m_child_id = child_index;     
                        (this_it+index3)->m_parent = &(*this_it);
                        index3 += tree_iter->operator[](child_index).subtree_size() - tree_iter->operator[](child_index).nleaves();
                    }
                    
                    ++this_it;
                }
                //if we are at a leaf node then we need to store the number of functions associated with this dimension
                else
                {
                    m_dim_sizes[leaf_counter] = tree_iter->value();    ++leaf_counter;
                }
            }
        }
        //base_type::initialise_level_indexing();
    }
};

template <typename T, typename backend>
using httensor_node = typename httensor<T, backend>::node_type;

template <typename T, typename backend> 
std::ostream& operator<<(std::ostream& os, const httensor<T, backend>& t)
{
    os << "dims: [";
    for(size_t i = 0; i<t.nmodes(); ++i){os << t.dim(i) << (i+1 != t.nmodes() ? ", " : "]");}
    os << std::endl << static_cast<const tree<httensor_node_data<T, backend> >&>(t);
    return os;
}


}   //namespace ttns

#endif  // HTUCKER_TENSOR_HPP //

