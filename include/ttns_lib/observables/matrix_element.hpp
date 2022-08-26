#ifndef HTUCKER_MATRIX_ELEMENT_HPP
#define HTUCKER_MATRIX_ELEMENT_HPP

//#include "../operators/sop_operator.hpp"
#include "../tdvp_core/matrix_element_core.hpp"
#include "../ttn_nodes/node_traits/bool_node_traits.hpp"

namespace ttns
{

template <typename T, typename backend=linalg::blas_backend>
class matrix_element
{
protected:
    //using sop_type = sop_operator<T, backend>;

    using real_type = typename tmp::get_real_type<T>::type;
    using size_type = typename backend::size_type;

    using me_core = matrix_element_engine<T, backend>;
    using op_base = typename me_core::op_base;

protected:
    tree<bool> m_is_identity;
    tree<linalg::matrix<T, backend> > m_matel;
    linalg::matrix<T, backend> m_opA;
    linalg::matrix<T, backend> m_temp;
    linalg::matrix<T> m_hmat;


protected:
    void reallocate_working_buffers(size_t maxcapacity)
    {
        CALL_AND_HANDLE(m_temp.reallocate(maxcapacity), "Failed to reallocate temporary array");
        CALL_AND_HANDLE(m_opA.reallocate(maxcapacity), "Failed to reallocate temporary array");
    }

    void resize_working_buffers(size_type s1, size_type s2)
    {
        CALL_AND_HANDLE(m_temp.resize(s1, s2), "Failed to resize temporary array");
        CALL_AND_HANDLE(m_opA.resize(s1, s2), "Failed to resize temporary array");
    }

    size_t get_maximum_size(const httensor<T, backend>& A) const
    {
        size_type maxsize = 0;
        for(const auto& a : A)
        {
            size_type size = a().size();            if(size > maxsize){maxsize = size;}
        }
        return maxsize;
    }

    size_t get_maximum_capacity(const httensor<T, backend>& A) const
    {
        size_type maxcapacity = 0;
        for(const auto& a : A)
        {
            size_type capacity = a().capacity();    if(capacity > maxcapacity){maxcapacity = capacity;}
        }
        return maxcapacity;
    }

public:
    matrix_element() {}
    matrix_element(const httensor<T, backend>& A, bool use_capacity = false) {CALL_AND_HANDLE(resize(A, use_capacity), "Failed to construct matrix_element object.  Failed to allocate internal buffers.");}
    matrix_element(const httensor<T, backend>& A, const httensor<T, backend>& B, bool use_capacity = false) {CALL_AND_HANDLE(resize(A, B, use_capacity), "Failed to construct matrix_element object.  Failed to allocate internal buffers.");}
    matrix_element(const matrix_element& o) = default;
    matrix_element(matrix_element&& o) = default;

    matrix_element& operator=(const matrix_element& o) = default;
    matrix_element& operator=(matrix_element&& o) = default;

    void clear()
    {
        try
        {
            m_matel.clear();
            m_is_identity.clear();
            m_opA.clear();
            m_temp.clear();
            m_hmat.clear();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear matrix element object.");
        }
    }
    
    void resize(const httensor<T, backend>& A, bool use_capacity = false)
    {
        try
        {
            size_type maxsize = get_maximum_size(A);

            if(use_capacity)
            {
                size_type maxcapacity = use_capacity ? get_maximum_capacity(A) : maxsize;
                reallocate_working_buffers(maxcapacity);
            }
            resize_working_buffers(1, maxsize);

            CALL_AND_HANDLE(m_hmat.resize(1, 1), "Failed to resize the host matrix.");
            CALL_AND_HANDLE(m_matel.construct_topology(A), "Failed to construct the topology of the matrix element buffer tree.");
            CALL_AND_HANDLE(m_is_identity.construct_topology(A), "Failed to construct the topology of the is_identity matrix tree.");

            for(auto z : zip(m_matel, m_is_identity, A))
            {
                auto& mel = std::get<0>(z); auto& is_id = std::get<1>(z); const auto& a = std::get<2>(z);
                is_id() = false;
                CALL_AND_HANDLE(me_core::resize(a(), mel(), use_capacity), "Failed to resize the matrix element buffers.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize internal buffers of the matrix_element object.");
        }
    }

    void resize(const httensor<T, backend>& A, const httensor<T, backend>& B, bool use_capacity = false)
    {
        try
        {
            ASSERT(has_same_structure(A, B), "The input hierarchical tucker tensors do not have the same topology.");

            size_type maxsize = get_maximum_size(B);
            size_type maxcapacity = use_capacity ? get_maximum_capacity(B) : maxsize;

            reallocate_working_buffers(maxcapacity);
            resize_working_buffers(1, maxsize);

            CALL_AND_HANDLE(m_matel.construct_topology(B), "Failed to construct the topology of the matrix element buffer tree.");
            CALL_AND_HANDLE(m_is_identity.construct_topology(B), "Failed to construct the topology of the is_identity matrix tree.");

            for(auto z : zip(m_matel, m_is_identity, A, B))
            {
                auto& mel = std::get<0>(z); auto& is_id = std::get<1>(z); const auto& a = std::get<2>(z); const auto& b = std::get<3>(z);
                is_id() = false;
                CALL_AND_HANDLE(me_core::resize(a(), b(), mel(), use_capacity), "Failed to resize the matrix element buffers.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize internal buffers of the matrix_element object.");
        }
    }

    real_type operator()(const httensor<T, backend>& psi){CALL_AND_RETHROW(return this->operator()(psi, false));}
    real_type operator()(const httensor<T, backend>& psi, bool compute_explicit)
    {
        try
        {
            ASSERT(has_same_structure(psi, m_matel), "The input hiearchical tucker tensor does not have the same topology as the matrix_element object.");
            for(auto z : rzip(psi, m_matel, m_is_identity))
            {
                const auto& p = std::get<0>(z); auto& mel = std::get<1>(z); auto& is_id = std::get<2>(z); 
                CALL_AND_HANDLE(m_opA.resize(p().shape(0), p().shape(1)), "Failed to resize HA buffer.");
                CALL_AND_HANDLE(m_temp.resize(p().shape(0), p().shape(1)), "Failed to resize HA buffer.");
                auto& HA = m_opA; auto& temp = m_temp;      
                if(!p.is_leaf())
                {
                    CALL_AND_HANDLE(me_core::compute_branch(p, HA, temp, mel, is_id, compute_explicit), "Failed to compute root node contraction.");
                }
                else
                {
                    CALL_AND_HANDLE(me_core::compute_leaf(p, mel, is_id, compute_explicit), "Failed to compute leaf node contraction.");
                }
            }
            CALL_AND_HANDLE(return real(gather_result(m_matel[0]())), "Failed to return result.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing inner product of hierarchical tucker tensor with itself.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of hierarchical tucker tensor with itself.");
        }
    }

    T Ct(const httensor<T, backend>& psi, bool compute_explicit)
    {
        try
        {
            ASSERT(has_same_structure(psi, m_matel), "The input hiearchical tucker tensor does not have the same topology as the matrix_element object.");
            for(auto z : rzip(psi, m_matel, m_is_identity))
            {
                const auto& p = std::get<0>(z); auto& mel = std::get<1>(z); auto& is_id = std::get<2>(z); 
                CALL_AND_HANDLE(m_opA.resize(p().shape(0), p().shape(1)), "Failed to resize HA buffer.");
                CALL_AND_HANDLE(m_temp.resize(p().shape(0), p().shape(1)), "Failed to resize HA buffer.");
                auto& HA = m_opA; auto& temp = m_temp;      
                if(!p.is_leaf())
                {
                    CALL_AND_HANDLE(me_core::ct_branch(p, HA, temp, mel, is_id, compute_explicit), "Failed to compute root node contraction.");
                }
                else
                {
                    CALL_AND_HANDLE(me_core::ct_leaf(p, mel, is_id, compute_explicit), "Failed to compute leaf node contraction.");
                }
            }
            CALL_AND_HANDLE(return gather_result(m_matel[0]()), "Failed to return result.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing inner product of hierarchical tucker tensor with itself.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of hierarchical tucker tensor with itself.");
        }
    }

    T operator()(const httensor<T, backend>& bra, const httensor<T, backend>& ket)
    {
        try
        {
            ASSERT(has_same_structure(bra, m_matel) && has_same_structure(ket, m_matel), "The input hiearchical tucker tensors do not both have the same topology as the matrix_element object.");
            if(&bra == &ket){CALL_AND_RETHROW(return operator()(bra));}

            for(auto z : rzip(bra, ket, m_matel, m_is_identity))
            {
                const auto& b = std::get<0>(z); auto& k = std::get<1>(z); auto& mel = std::get<2>(z); auto& is_id = std::get<3>(z);
                CALL_AND_HANDLE(m_opA.resize(k().shape(0), k().shape(1)), "Failed to resize HA buffer.");
                CALL_AND_HANDLE(m_temp.resize(k().shape(0), k().shape(1)), "Failed to resize HA buffer.");
                auto& HA = m_opA; auto& temp = m_temp;      
                if(!b.is_leaf())
                {
                    CALL_AND_HANDLE(me_core::compute_branch(b,  k, HA, temp, mel, is_id), "Failed to compute root node contraction.");
                }
                else
                {
                    CALL_AND_HANDLE(me_core::compute_leaf(b, k, mel, is_id), "Failed to compute leaf node contraction.");
                }
            }
            CALL_AND_HANDLE(return gather_result(m_matel[0]()), "Failed to return result.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing inner product of two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of two hierarchical tucker tensors.");
        }
    }

    //a function that calculates the expectation value of an operator acting on a single mode of the ttns.  If the flag reuse_nodes is set
    //to true this only updates the nodes in the tree that are ancestors of the leaf node with index mode.  This will only 
    //be correct if this function has been called before with an operator acting on the same mode and with the same psi.
    template <typename op_type>
    inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, T>::type operator()(op_type& op, size_type mode, const httensor<T, backend>& psi, bool reuse_nodes = false)
    {
        try
        {
            ASSERT(has_same_structure(psi, m_matel), "The input hiearchical tucker tensor does not have the same topology as the matrix_element object.");
            ASSERT(mode < psi.nmodes(), "The mode that the input operator acts on is out of bounds.");

            if(!reuse_nodes)
            {
                for(auto z : rzip(psi, m_matel, m_is_identity))
                {
                    const auto& p = std::get<0>(z); auto& mel = std::get<1>(z); auto& is_id = std::get<2>(z);
                    CALL_AND_HANDLE(m_opA.resize(p().shape(0), p().shape(1)), "Failed to resize HA buffer.");
                    CALL_AND_HANDLE(m_temp.resize(p().shape(0), p().shape(1)), "Failed to resize HA buffer.");
                    auto& HA = m_opA; auto& temp = m_temp;      
                    if(!p.is_leaf())
                    {
                        CALL_AND_HANDLE(me_core::compute_branch(p, HA, temp, mel, is_id), "Failed to compute root node contraction.");
                    }
                    else
                    {
                        if(p.leaf_index() != mode)
                        {
                            CALL_AND_HANDLE(me_core::compute_leaf(p, mel, is_id), "Failed to compute leaf node contraction.");
                        }
                        else
                        {
                            CALL_AND_HANDLE(me_core::compute_leaf(op, p, HA, mel, is_id), "Failed to compute leaf node contraction.");
                        }
                    }
                }
            }
            else
            {

            }
            CALL_AND_HANDLE(return gather_result(m_matel[0]()), "Failed to return result.");
        }        
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing expectation value of operator using a hierarchical tucker tensor state.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute expectation value of operator using a hierarchical tucker tensor state.");
        }
    }


/*  
    T operator()(const po_type& op, const httensor<T, backend>& psi)
    {
        try
        {
            ASSERT(has_same_structure(psi, m_matel), "The input hiearchical tucker tensor does not have the same topology as the matrix_element object.");
            ASSERT(op.nmodes() == psi.nmodes(), "The input product operator does not act on the correct number of modes.");
            for(auto z : rzip(psi, m_matel, m_is_identity, m_opA, m_temp))
            {
                const auto& p = std::get<0>(z); auto& mel = std::get<1>(z); auto& is_id = std::get<2>(z); auto& HA = std::get<3>(z); auto& temp = std::get<4>(z);
                if(!p.is_leaf())
                {
                    CALL_AND_HANDLE(me_core::compute_branch(p, HA, temp, mel, is_id), "Failed to compute root node contraction.");
                }
                else
                {
                    if(!op.is_identity(p.leaf_index()))
                    {
                        CALL_AND_HANDLE(me_core::compute_leaf(op[p.leaf_index()], p, HA, mel, is_id), "Failed to compute leaf node contraction.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(me_core::compute_leaf(p, mel, is_id), "Failed to compute leaf node contraction.");
                    }
                }
            }
            CALL_AND_HANDLE(return gather_result(m_matel[0]()), "Failed to return result.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing expectation value of product operator using a hierarchical tucker tensor state.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute expectation value of product operator using a hierarchical tucker tensor state.");
        }
    }

    T operator()(const sop_type& op, const httensor<T, backend>& psi)
    {
        try
        {   
            ASSERT(has_same_structure(psi, m_matel), "The input hiearchical tucker tensor does not have the same topology as the matrix_element object.");
            ASSERT(op.nmodes() == psi.nmodes(), "The input product operator does not act on the correct number of modes.");
            
            T ret(0);
            for(size_type r=0; r<op.nterms(); ++r)
            {
                CALL_AND_HANDLE(ret += operator()(op[r], psi)*op.coeff(r), "Failed to compute product operator contribution to sum of product operator.");
            }
            return ret;
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing expectation value of sum of product operator using a hierarchical tucker tensor state.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute expectation value of sum of product operator using a hierarchical tucker tensor state.");
        }
    }
*/
    //a function that calculates the matrix element of a an operator that acts on single mode of the ttns.  If the flag reuse_nodes is set
    //to true this only updates the nodes in the tree that are ancestors of the leaf node with index mode.  This will only 
    //be correct if this function has been called before with an operator acting on the same mode and with the same bra and ket.
    template <typename op_type>
    inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, T>::type operator()(const op_type& op, size_type mode, const httensor<T, backend>& bra, const httensor<T, backend>& ket, bool reuse_nodes = false)
    {
        try
        {
            if(&bra == &ket){CALL_AND_RETHROW(return operator()(op, mode, bra));}

            ASSERT(has_same_structure(bra, m_matel) && has_same_structure(ket, m_matel), "The two input hiearchical tucker tensor do not both have the same topology as the matrix_element object.");
            ASSERT(mode < ket.nmodes(), "The mode that the input operator acts on is out of bounds.");
            if(!reuse_nodes)
            {
                for(auto z : rzip(bra, ket, m_matel, m_is_identity))
                {
                    const auto& b = std::get<0>(z); const auto& k = std::get<1>(z); auto& mel = std::get<2>(z); auto& is_id = std::get<3>(z);
                    CALL_AND_HANDLE(m_opA.resize(k().shape(0), k().shape(1)), "Failed to resize HA buffer.");
                    CALL_AND_HANDLE(m_temp.resize(k().shape(0), k().shape(1)), "Failed to resize HA buffer.");
                    auto& HA = m_opA; auto& temp = m_temp;      
                    if(!b.is_leaf())
                    {
                        CALL_AND_HANDLE(me_core::compute_branch(b, k, HA, temp, mel, is_id), "Failed to compute root node contraction.");
                    }
                    else
                    {
                        if(b.leaf_index() != mode)
                        {
                            CALL_AND_HANDLE(me_core::compute_leaf(b, k, mel, is_id), "Failed to compute leaf node contraction.");
                        }
                        else
                        {
                            CALL_AND_HANDLE(me_core::compute_leaf(op, b, k, HA, mel, is_id), "Failed to compute leaf node contraction.");
                        }
                    }
                }
            }
            else
            {

            }
            CALL_AND_HANDLE(return gather_result(m_matel[0]()), "Failed to return result.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing matrix element of operator using two hierarchical tucker tensor states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute matrix element of operator using two hierarchical tucker tensor states.");
        }
    }

/*
    T operator()(const po_type& op, const httensor<T, backend>& bra, const httensor<T, backend>& ket)
    {
        try
        {
            if(&bra == &ket){CALL_AND_RETHROW(return operator()(op, bra));}
            ASSERT(has_same_structure(bra, m_matel) && has_same_structure(ket, m_matel), "The two input hiearchical tucker tensor do not both have the same topology as the matrix_element object.");
            ASSERT(op.nmodes() == bra.nmodes(), "The input product operator does not act on the correct number of modes.");
            for(auto z : rzip(bra, ket, m_matel, m_is_identity, m_opA, m_temp))
            {
                const auto& b = std::get<0>(z); const auto& k = std::get<1>(z); auto& mel = std::get<2>(z); auto& is_id = std::get<3>(z); auto& HA = std::get<4>(z); auto& temp = std::get<5>(z);
                if(!b.is_leaf())
                {
                    CALL_AND_HANDLE(me_core::compute_branch(b, k, HA, temp, mel, is_id), "Failed to compute root node contraction.");
                }
                else
                {

                    if(!op.is_identity(b.leaf_index()))
                    {
                        CALL_AND_HANDLE(me_core::compute_leaf(op[b.leaf_index()], b, k, HA, mel, is_id), "Failed to compute leaf node contraction.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(me_core::compute_leaf(b, k, mel, is_id), "Failed to compute leaf node contraction.");
                    }
                }
            }
            CALL_AND_HANDLE(return gather_result(m_matel[0]()), "Failed to return result.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing matrix element of product operator using two hierarchical tucker tensor states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute matrix element of product operator using two hierarchical tucker tensor states.");
        }
    }

    T operator()(const sop_type& op, const httensor<T, backend>& bra, const httensor<T, backend>& ket)
    {
        try
        {
            if(&bra == &ket){CALL_AND_RETHROW(return operator()(op, bra));}
            ASSERT(has_same_structure(bra, m_matel) && has_same_structure(ket, m_matel), "The two input hiearchical tucker tensor do not both have the same topology as the matrix_element object.");
            ASSERT(op.nmodes() == bra.nmodes(), "The input product operator does not act on the correct number of modes.");

            T ret(0);
            for(size_type r=0; r<op.nterms(); ++r)
            {
                CALL_AND_HANDLE(ret += operator()(op[r], bra, ket)*op.coeff(r), "Failed to compute product operator contribution to sum of product operator.");
            }
            return ret;
        }        
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing matrix element of sum of product operator using two hierarchical tucker tensor states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute matrix element of sum of product operator using two hierarchical tucker tensor states.");

        }
    }
*/

protected:
#ifdef __NVCC__
    T gather_result(const linalg::matrix<T, linalg::cuda_backend>& o)
    {
        CALL_AND_HANDLE(m_hmat = o, "Failed to copy device result back to host.");
        CALL_AND_HANDLE(return m_hmat(0,0), "Failed to return result.");
    }
#endif

    T gather_result(const linalg::matrix<T, linalg::blas_backend>& o)
    {
        CALL_AND_HANDLE(return o(0,0), "Failed to return result.");
    }
};

}   //namespace ttns

#endif  //HTUCKER_MATRIX_ELEMENT_HPP//

