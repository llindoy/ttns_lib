#ifndef HTUCKER_TENSOR_NODE_HPP
#define HTUCKER_TENSOR_NODE_HPP


#include <linalg/linalg.hpp>
#include "../tree/tree.hpp"
#include "../tree/tree_node.hpp"
#include "../tree/ntree.hpp"

#include <vector>
#include <stdexcept>


namespace ttns
{
using namespace linalg;

template <typename T, typename backend> class httensor;

template <typename T, typename backend>
class httensor_node_data : public matrix<T, backend> 
{
public:
    using matrix_type = matrix<T, backend>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

protected:
    std::vector<size_type> m_mode_dims;
    std::vector<size_type> m_mode_capacity;
    size_type m_max_hrank;
    size_type m_max_dimen;
    bool m_orthogonalised;

    friend tree_node<tree_base<httensor_node_data<T, backend> > >;
    friend httensor<T, backend>;

public:
    httensor_node_data() : matrix_type(), m_mode_dims(), m_mode_capacity(), m_max_hrank(0), m_max_dimen(0), m_orthogonalised(false) {}
    httensor_node_data& operator=(const httensor_node_data& o) 
    {
        CALL_AND_HANDLE(matrix_type::operator=(static_cast<const matrix<T, backend>&>(o)), "Failed to copy assign httensor_node_data. Failed when applying base type copy operator.");
        m_mode_dims = o.m_mode_dims;
        m_mode_capacity = o.m_mode_capacity;
        m_max_hrank = o.m_max_hrank;
        m_max_dimen = o.m_max_dimen;
        m_orthogonalised = o.m_orthogonalised;
        return *this;
    }

    template <typename be>
    typename std::enable_if<not std::is_same<be, backend>::value, httensor_node_data&>::type 
    operator=(const httensor_node_data<T, be> & o) 
    {
        CALL_AND_HANDLE(matrix_type::operator=(static_cast<const matrix<T, be>&>(o)), "Failed to copy assign httensor_node_data. Failed when applying base type copy operator.");
        m_mode_dims = o.dims();
        m_mode_capacity = o.max_dims();
        m_max_hrank = o.max_hrank();
        m_max_dimen = o.m_max_dimen();
        m_orthogonalised = o.is_orthogonalised();
        return *this;
    }

    httensor_node_data& operator=(httensor_node_data&& o)
    {
        CALL_AND_HANDLE(matrix_type::operator=(std::forward<matrix<T, backend>>(o)), "Failed to move assign httensor_node_data. Failed when applying base type move operator.");
        m_mode_dims = std::move(o.m_mode_dims);
        m_mode_capacity = std::move(o.m_mode_capacity);
        m_max_hrank = o.m_max_hrank;
        m_max_dimen = o.m_max_dimen;
        m_orthogonalised = o.m_orthogonalised;
        return *this;
    }

    template <typename U>
    typename std::enable_if<std::is_convertible<U, T>::value, httensor_node_data&>::type operator=(const matrix<U, backend>& mat)
    {
        ASSERT(mat.shape() == matrix_type::shape(), "Failed to copy assign httensor_node_data from matrix.  The matrix is not the correct size.")
        CALL_AND_HANDLE(matrix_type::operator=(mat), "Failed to copy assign httensor_node_data from matrix. Failed when applying base type copy operator.");
        return *this;
    }

    void resize(size_type hrank, const std::vector<size_type>& mode_dims)
    {
        m_mode_capacity.resize(mode_dims.size());
        size_type ndimen = 1;
        for(size_type i = 0; i < mode_dims.size(); ++i)
        {
            if(m_mode_capacity[i] < mode_dims[i]){m_mode_capacity[i] = mode_dims[i];}
            ndimen *= mode_dims[i];
            m_mode_dims[i] = mode_dims[i];
        }
        matrix_type::resize(ndimen, hrank);

        if(ndimen > m_max_dimen){m_max_dimen = ndimen;}
        if(hrank > m_max_hrank){m_max_hrank = hrank;}
        m_orthogonalised = false;
    }

    void reallocate(size_type max_hrank, const std::vector<size_type>& max_mode_dims)
    {
        m_mode_capacity = max_mode_dims;    
        size_type ndimen = 1;
        for(size_type i = 0; i < max_mode_dims.size(); ++i)
        {
            ndimen *= max_mode_dims[i];
        }
        m_max_hrank = max_hrank;
        m_max_dimen = ndimen;
        matrix_type::reallocate(ndimen, max_hrank);
    }

    bool is_orthogonalised() const{return m_orthogonalised;}

    size_type nmodes() const{return m_mode_dims.size();}


    size_type hrank(bool use_max_dim = false) const
    {
        if(!use_max_dim){return this->shape(1);}
        else{return m_max_hrank;}
    }
    size_type dimen(bool use_max_dim = false) const 
    {
        if(!use_max_dim){return this->shape(0);}
        else{return this->m_max_dimen;}
    }
    size_type dim(size_type n, bool use_max_dim = false) const
    {
        if(!use_max_dim){return m_mode_dims[n];}
        else{return m_mode_capacity[n];}
    }
    const std::vector<size_type>& dims() const
    {
        return m_mode_dims;
    }

    size_type max_hrank() const{return m_max_hrank;}
    size_type max_dimen() const{return m_max_dimen;}
    size_type max_dim(size_type n) const{return m_mode_capacity[n];}
    const std::vector<size_type>& max_dims() const{return m_mode_capacity;}
    
    reinterpreted_tensor<T, 2, backend> as_rank_2(bool use_max_dim = false)
    {
        if(!use_max_dim){return this->reinterpret_shape(dimen(), hrank());}
        else{return this->reinterpret_capacity(dimen(use_max_dim), hrank(use_max_dim));}
    }

    reinterpreted_tensor<const T, 2, backend> as_rank_2(bool use_max_dim = false) const
    {
        if(!use_max_dim){return this->reinterpret_shape(dimen(), hrank());}
        else{return this->reinterpret_capacity(dimen(use_max_dim), hrank(use_max_dim));}
    }


    reinterpreted_tensor<T, 3, backend> as_rank_3(size_type mode, bool use_max_dim = false)
    {
        try
        {
            ASSERT(mode <= nmodes(), "Failed to interpret httensor_node_data as rank 3 tensor.  The mode index is out of bounds.");
            if(mode < nmodes())
            {
                if(!use_max_dim)
                {
                    std::array<size_type, 3> shape{{1, dim(mode), hrank()}};
                    for(size_type i=0; i<mode; ++i){shape[0]*=dim(i);}   
                    for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= dim(i);}
                    return this->reinterpret_shape(shape[0], shape[1], shape[2]);
                }
                else
                {
                    std::array<size_type, 3> shape{{1, max_dim(mode), max_hrank()}};
                    for(size_type i=0; i<mode; ++i){shape[0]*=max_dim(i);}   
                    for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= max_dim(i);}
                    return this->reinterpret_capacity(shape[0], shape[1], shape[2]);
                }
            }
            else
            {
                if(!use_max_dim)
                {
                    return this->reinterpret_shape(dimen(), hrank(), 1);
                }
                else
                {
                    return this->reinterpret_capacity(max_dimen(), max_hrank(), 1);
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to reinterpret hierarchical tucker tensor node as a rank 3 tensor.");
        }
    }

    reinterpreted_tensor<const T, 3, backend> as_rank_3(size_type mode, bool use_max_dim = false) const
    {
        try
        {
            ASSERT(mode <= nmodes(), "Failed to interpret httensor_node_data as rank 3 tensor.  The mode index is out of bounds.");

            if(mode < nmodes())
            {
                if(!use_max_dim)
                {
                    std::array<size_type, 3> shape{{1, dim(mode), hrank()}};
                    for(size_type i=0; i<mode; ++i){shape[0]*=dim(i);}   
                    for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= dim(i);}
                    return this->reinterpret_shape(shape[0], shape[1], shape[2]);
                }
                else
                {
                    std::array<size_type, 3> shape{{1, max_dim(mode), max_hrank()}};
                    for(size_type i=0; i<mode; ++i){shape[0]*=max_dim(i);}   
                    for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= max_dim(i);}
                    return this->reinterpret_capacity(shape[0], shape[1], shape[2]);
                }        
            }
            else
            {
                if(!use_max_dim)
                {
                    return this->reinterpret_shape(dimen(), hrank(), 1);
                }
                else
                {
                    return this->reinterpret_capacity(max_dimen(), max_hrank(), 1);
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to reinterpret hierarchical tucker tensor node as a rank 3 tensor.");
        }
    }

    reinterpreted_tensor<T, 4, backend> as_rank_4(size_type mode)
    {
        ASSERT(mode < nmodes(), "Failed to interpret httensor_node_data as rank 3 tensor.  The mode index is out of bounds.");
        std::array<size_type, 4> shape{{1, dim(mode), 1, hrank()}};
        for(size_type i=0; i<mode; ++i){shape[0]*=dim(i);}   
        for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= dim(i);}
        return this->reinterpret_shape(shape[0], shape[1], shape[2], shape[3]);
    }

    reinterpreted_tensor<const T, 4, backend> as_rank_4(size_type mode) const
    {
        ASSERT(mode < nmodes(), "Failed to interpret httensor_node_data as rank 3 tensor.  The mode index is out of bounds.");
        std::array<size_type, 4> shape{{1, dim(mode), 1, hrank()}};
        for(size_type i=0; i<mode; ++i){shape[0]*=dim(i);}   
        for(size_type i=mode+1; i<nmodes(); ++i){shape[2] *= dim(i);}
        return this->reinterpret_shape(shape[0], shape[1], shape[2], shape[3]);
    }

    linalg::matrix<T, backend>& as_matrix(){return *this;}
    const linalg::matrix<T, backend>& as_matrix() const {return *this;} 

    void clear()
    {
        m_orthogonalised = false;
        m_mode_dims.clear();
        matrix_type::clear();
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<matrix<T, backend>>(this)), "Failed to serialise httensor_node_data object.  Error when serialising the base matrix object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("dims", m_mode_dims)), "Failed to serialise httensor_node_object object.  Error when serialising mode dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("is_orthogonalised", m_orthogonalised)), "Failed to serialise httensor_node_data object.  Failed to serialise whether the node was orthogonalised.");
    }

    template <typename archive>
    void load(archive& ar) 
    {
        CALL_AND_HANDLE(ar(cereal::base_class<matrix<T, backend>>(this)), "Failed to serialise httensor_node_data object.  Error when serialising the base matrix object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("dims", m_mode_dims)), "Failed to serialise httensor_node_object object.  Error when serialising mode dimensions.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("is_orthogonalised", m_orthogonalised)), "Failed to serialise httensor_node_data object.  Failed to serialise whether the node was orthogonalised.");
    }
#endif
};

template <typename T, typename backend, typename = typename std::enable_if<std::is_same<backend, blas_backend>::value, void>::type> 
std::ostream& operator<<(std::ostream& os, const httensor_node_data<T, backend>& t)
{
    os << (t.is_orthogonalised() ? "orthogonal" : "not orthogonal") << std::endl;
    os << "dims: " << "[ ";     for(size_t i=0; i<t.nmodes(); ++i){os << t.dim(i) << (i+1 != t.nmodes() ? ", " : "]");}    os << std::endl;
    os << static_cast<const matrix<T, backend>&>(t) << std::endl;
    return os;
}

}

#include "node_traits/httensor_node_traits.hpp"
#include "node_traits/tensor_node_traits.hpp"

namespace ttns
{

template <typename T, typename backend> 
class tree_node<tree_base<httensor_node_data<T, backend> > > : 
    public tree_node_base<tree_base<httensor_node_data<T, backend> > >
{
    static_assert(std::is_base_of<backend_base, backend>::value, "The second template argument to the httensor_node object must be a valid backend.");
public:
    using matrix_type = matrix<T, backend>;
    using value_type = httensor_node_data<T, backend>;
    using tree_type = tree_base<httensor_node_data<T, backend> >;
    using base_type = tree_node_base<tree_type>;
    using size_type = typename backend::size_type;
    using node_type = tree_node<tree_base<httensor_node_data<T, backend> > >;

    friend class tree<httensor_node_data<T, backend>>;
    friend class tree_base<httensor_node_data<T, backend>>;
    friend class httensor<T, backend>;

protected:
    using base_type::m_data;
    using base_type::m_children;
    using base_type::m_parent;

public:
    tree_node() : base_type(){}

    void zero(){m_data.fill_zeros();   m_data.m_orthogonalised = false;}

    size_type hrank() const{return m_data.hrank();}
    size_type nmodes() const{return m_data.nmodes();}
    size_type dimen() const {return m_data.dimen();}
    size_type dim(size_type n) const{return m_data.dim(n);}
    const std::vector<size_type>& dims() const{return m_data.dims();}

    const value_type& operator()() const {return m_data;}
    value_type& operator()() {this->lose_orthogonalisation();   return m_data;}

    bool is_orthogonalised() const{return m_data.m_orthogonalised;}

#ifdef CEREAL_LIBRARY_FOUND
    friend class serialisation_node_save_wrapper<node_type, size_type>;
    friend class serialisation_node_load_wrapper<node_type, size_type>;
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<tree_node_base<tree_base<httensor_node_data<T, backend> > >>(this)), "Failed to serialise httensor_node object.  Error when serialising the base object.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<tree_node_base<tree_base<httensor_node_data<T, backend> > >>(this)), "Failed to serialise httensor_node object.  Error when serialising the base object.");
    }
#endif
protected:
    void set_is_orthogonalised(bool is_orthog = true){m_data.m_orthogonalised = is_orthog;}

    //function for recursively propagating the loss of orthogonalisation up the tree.
    //This function will continue until we reach a region which has already lost its 
    //orthogonalisation or when we reach the root 
    void lose_orthogonalisation()
    {
        if(!m_data.m_orthogonalised){m_data.m_orthogonalised = false;   if(m_parent != nullptr){m_parent->lose_orthogonalisation();}}
        else{m_data.m_orthogonalised = false;}
    }
};


}   //namespace ttns


#endif  //HTUCKER_TENSOR_NODE_HPP//


