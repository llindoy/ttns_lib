#ifndef HTUCKER_DECOMPOSITION_ENGINE_HPP
#define HTUCKER_DECOMPOSITION_ENGINE_HPP

#include <linalg/decompositions/singular_value_decomposition/singular_value_decomposition.hpp>

namespace ttns
{
template <typename T, typename backend=blas_backend, bool use_divide_and_conquer = true>
class decomposition_engine
{
public:
    using real_type = typename tmp::get_real_type<T>::type;
    using size_type = typename httensor<T,backend>::size_type;
    using matrix_type = matrix<T, backend>;
    using dmat_type = diagonal_matrix<real_type, backend>;
public:
    decomposition_engine() {}

    decomposition_engine& operator=(const decomposition_engine& o) = default;
    decomposition_engine& operator=(decomposition_engine&& o) = default;

    template <typename decomp_type>
    void resize(const httensor<T, backend>& A, matrix_type& U, tree<matrix_type>& R, bool use_capacity = false)
    {
        try
        {
            CALL_AND_HANDLE(resize_buffers<decomp_type>(A, use_capacity), "Failed to resize the buffers.");

            size_type max_ws = 0;
            //iterate over all nodes and determine the maximum worksize
            for(auto z : zip(A, R))
            {
                size_type ws;
                auto& a = std::get<0>(z);
                auto& r = std::get<1>(z);
                auto& u = U;    CALL_AND_HANDLE(u.resize(a().dimen(use_capacity), a().hrank(use_capacity)), "Failed to resize u array.");
                CALL_AND_HANDLE(ws = decomp_type::maximum_work_size_node(*this, a, u, r, use_capacity), "Failed to when attempting to query maximum work size for node.");
                if(ws > max_ws){max_ws = ws;}
            }
            CALL_AND_HANDLE(m_svd.resize_work_space(max_ws), "Faile to resize work space for svd object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize decomposition engine object.");
        }
    }


    template <typename decomp_type>
    void resize(const httensor<T, backend>& A, matrix_type& U, matrix_type& r, bool use_capacity = false)
    {
        try
        {
            CALL_AND_HANDLE(resize_buffers<decomp_type>(A), "Failed to resize the buffers.");

            size_type max_ws = 0;
            //iterate over all nodes and determine the maximum worksize
            for(auto& a : A)
            {
                size_type ws;
                r.resize(a().hrank(use_capacity), a().hrank(use_capacity));
                auto& u = U;    CALL_AND_HANDLE(u.resize(a().dimen(use_capacity), a().hrank(use_capacity)), "Failed to resize u array.");
                CALL_AND_HANDLE(ws = decomp_type::maximum_work_size_node(*this, a, u, r, use_capacity), "Failed to when attempting to query maximum work size for node.");
                if(ws > max_ws){max_ws = ws;}
            }
            CALL_AND_HANDLE(m_svd.resize_work_space(max_ws), "Faile to resize work space for svd object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize decomposition engine object.");
        }
    }

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_s.clear(), "Failed to clear the s matrix.");
            CALL_AND_HANDLE(m_temp.clear(), "Failed to clear a temporary working matrix.");
            CALL_AND_HANDLE(m_temp2.clear(), "Failed to clear a temporary working matrix..");
            CALL_AND_HANDLE(m_svd.clear(), "Failed to clear the svd engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the decomposition engine object.");
        }
    }

    template <typename Atype, typename Utype, typename Vtype>
    size_type query_work_size(const Atype& A, Utype& U, Vtype& V)
    {
        try
        {
            CALL_AND_HANDLE(m_s.resize(A.shape()), "Failed to resize m_s matrix.");
            size_type ws;
            CALL_AND_HANDLE(ws = m_svd.query_work_space(A, m_s, U, V, false), "Failed when making query work space call for the underlying svd object.");
            return ws;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed when querying the work size for the decomposition engine object.");
        }
    }

    template <typename Atype, typename Utype, typename Vtype>
    void operator()(const Atype& A, Utype& U, Vtype& V)
    {
        try
        {
            //check that the temporary arrays have the correct capacity
            ASSERT(V.size() <= m_temp.capacity(), "The temporary matrix does not have sufficient capacity.");
            ASSERT(A.shape(1) <= m_s.capacity(), "The matrix of singular values does not have sufficient capacity.");

            CALL_AND_HANDLE(m_s.resize(A.shape(1), A.shape(1)), "Failed to resize S matrix.");
            CALL_AND_HANDLE(m_temp.resize(V.shape()), "Failed to resize temporary V matrix to ensure it has the correct shape.");
            CALL_AND_HANDLE(m_svd(A, m_s, U, m_temp, A.shape(0) < A.shape(1)), "Failed when evaluating the decomposition.")


            //if A.shape(0) < A.shape(1) then the resultant U matrix is the square matrix.  For our implementation this matrix must be 
            //a rectangular matrix and so now we resize and zero pad the U and S matrices to make sure everything is correct.
            using memfill = linalg::memory::filler<real_type, backend>;

            if(A.shape(0) < A.shape(1))
            {       
                ASSERT(A.size() <= m_temp2.capacity(), "The second temporary matrix does not have sufficient capacity.");
                CALL_AND_HANDLE(m_temp2.resize(A.shape(0), A.shape(1)), "Failed to resize the second temporary matrix so that it has the correct shape.");

                try
                {
                    backend::fill_matrix_block(U.buffer(), U.shape(0), U.shape(1), m_temp2.buffer(), m_temp2.shape(0), m_temp2.shape(1));
                }
                catch(const std::exception& ex)
                {
                    RAISE_EXCEPTION("Failed to zero pad the U matrix so that it has the correct shape.");
                }

                CALL_AND_HANDLE(U.resize(A.shape(0), A.shape(1)), "Failed to resize the U matrix.");

                //we might want to change this for a swap operation
                CALL_AND_HANDLE(U = m_temp2, "Failed to assign the U matrix.");
                
                
                CALL_AND_HANDLE(m_s.resize(A.shape(1), A.shape(1)), "Failed to resize the S matrix.");

                try
                {
                    memfill::fill(m_s.buffer()+A.shape(0), A.shape(1)-A.shape(0), real_type(0.0));
                }
                catch(const std::exception& ex)
                {
                    RAISE_EXCEPTION("Failed to zero pad the S matrix so that it has the correct shape.");
                }
            }

            CALL_AND_HANDLE(V = m_s*m_temp, "Failed to assign rb matrix.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying the decomposition engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the decomposition engine.");
        }
    }

    template <typename Vtype>
    void transposeV(Vtype& V)
    {
        try
        {
            if(std::is_same<backend, linalg::blas_backend>::value)
            {
                CALL_AND_HANDLE(V = trans(V), "Failed to evaluate inplace transpose of V matrix.");
            }
            else
            {
                ASSERT(V.size() <= m_temp.capacity(), "The temporary matrix does not have sufficient capacity.");
                CALL_AND_HANDLE(m_temp = trans(V), "Failed to evaluate transpose of V matrix into temporary buffer.");
                CALL_AND_HANDLE(V = m_temp, "Failed to store transposed V matrix.");
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying transposeV.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply transposeV on result of decomposition engine.");
        }
    }

    const dmat_type& S() const
    {
        return m_s;
    }
protected:
    template <typename decomp_type>
    void resize_buffers(const httensor<T, backend>& A, bool use_capacity = false)
    {
        try
        {
            std::array<size_type, 2> max_dim{{0,0}};
            size_type temp2_cap = 0;
            //iterate over all nodes and determine the maximum dimension in each 
            for(const auto& a : A)
            {
                std::array<size_type, 2> dim = decomp_type::maximum_matrix_dimension_node(a, use_capacity);
                for(size_type i = 0; i<2; ++i){if(dim[i] > max_dim[i]){max_dim[i] = dim[i];}}

                size_type node_size = use_capacity ? a().capacity() : a().size();
                if(node_size > temp2_cap){temp2_cap = node_size;}
            }

            //resize the svd object
            CALL_AND_HANDLE(m_svd.resize(max_dim[0], max_dim[1], false), "Failed to resize svd object.");

            //and the temporary buffer we use for constructing the temporary results into
            CALL_AND_HANDLE(m_temp.resize(max_dim[1], max_dim[1]), "Failed to resize temporary buffer.");
            CALL_AND_HANDLE(m_temp2.resize(1, temp2_cap), "Failed to resize second temporary buffer.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize decomposition engine object.");
        }
    }


    dmat_type m_s;
    matrix_type m_temp;
    matrix_type m_temp2;
    singular_value_decomposition<matrix_type, use_divide_and_conquer> m_svd;
};

}

#endif


