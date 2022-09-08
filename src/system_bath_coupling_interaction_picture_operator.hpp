#ifndef SYSTEM_BATH_COUPLING_INTERACTION_PICTURE_HPP
#define SYSTEM_BATH_COUPLING_INTERACTION_PICTURE_HPP


#include <ttns_lib/operators/primitive_operator.hpp>


//A class for wrapping the multiplication by a sparse matrix.  This stores the matrix in csr form to make it easy to perform all of the required operations.
template <typename T, typename backend = linalg::blas_backend>
class interaction_picture_sb_operator;


//blas specialisation of the sparse matrix operator.
template <typename T> 
class interaction_picture_sb_operator<T, linalg::blas_backend> : public ttns::ops::primitive<T, linalg::blas_backend>
{
public:
    using backend = linalg::blas_backend;
    using base_type = ttns::ops::primitive<T, backend>;

    //use the parent class type aliases
    using typename base_type::matrix_type;
    using typename base_type::size_type;
    using typename base_type::matrix_ref;
    using typename base_type::const_matrix_ref;
    using typename base_type::real_type;
    using typename base_type::vector_type;
    using typename base_type::vector_ref;
    using typename base_type::const_vector_ref;

protected:
    interaction_picture_sb_operator(const std::vector<real_type>& wk, const std::vector<real_type>& gk, const linalg::matrix<size_t> & nk, const linalg::matrix<int32_t>& nkp1, const linalg::matrix<int32_t>& nkm1, const linalg::csr_matrix<T, backend>& mat)
    {
        m_wk = wk;
        m_gk = gk;
        m_nk = nk;
        m_nkp1 = nkp1;
        m_nkm1 = nkm1;
        m_operator = mat;

        base_type::m_size = m_operator.shape(0);
    }
public:
    interaction_picture_sb_operator() : base_type() {}
    interaction_picture_sb_operator(const interaction_picture_sb_operator& o) = default;
    interaction_picture_sb_operator(interaction_picture_sb_operator&& o) = default;

    void resize(size_type /*n*/){ASSERT(false, "This shouldn't be called.");}
    base_type* clone() const{return new interaction_picture_sb_operator(m_wk, m_gk, m_nk, m_nkp1, m_nkm1, m_operator);}

    void apply(const_matrix_ref A, matrix_ref HA) final{CALL_AND_HANDLE(HA = m_operator*A, "Failed to apply sparse matrix operator.  Failed to compute sparse matrix matrix product.");}
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(this->apply(A, HA));}  
    void apply(const_vector_ref A, vector_ref HA) final{CALL_AND_HANDLE(HA = m_operator*A, "Failed to apply sparse matrix operator.  Failed to compute sparse matrix vector product.");}
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(this->apply(A, HA));}  
    void update(real_type t, real_type dt) final
    {   
        size_t nstates = m_nk.shape(0);     size_t N = m_nk.shape(1);
        linalg::vector<T> gkt(N);
        for(size_t i = 0; i < N; ++i)
        {
            real_type wkdt = m_wk[i]*dt;    
            T iwkdt = T(0, m_wk[i]*dt);    
            T Ukt = T(cos(m_wk[i]*t), sin(m_wk[i]*t));
            gkt[i] = T(0, 0);
            if(std::abs(wkdt) < 1e-5)
            {
                T factor = 1;
                for(size_t k = 0; k < 20; ++k)
                {
                    gkt[i] += factor;
                    factor *= iwkdt/(k+1.0);
                }
                gkt[i] *= m_gk[i]*Ukt;
            }
            else
            {
                T Ukdt = T(cos(m_wk[i]*dt), sin(m_wk[i]*dt));
                gkt[i] = m_gk[i] * Ukt * (Ukdt - 1.0)/(T(0, wkdt));
            }
        }

        //now we can set the colind and buffer values
        size_t counter = 0;
        auto buffer = m_operator.buffer();
        for(size_t i=0; i<nstates; ++i)
        {
            for(size_t j=0; j<N; ++j)
            {
                if(m_nkm1(i, j) != -1)
                {
                    real_type n = static_cast<real_type>(m_nk(i, j));
                    buffer[counter] = sqrt(n)*gkt[j];
                    ++counter;
                }
            }

            for(size_t k=0; k<N; ++k)
            {
                size_t j = N-(k+1);
                if(m_nkp1(i, j) != -1)
                {
                    real_type n = static_cast<real_type>(m_nk(i, j)+1.0);
                    buffer[counter] = sqrt(n)*std::conj(gkt[j]);
                    ++counter;
                }
            }
        }
    }  

    void initialise(const std::vector<real_type>& wk, const std::vector<real_type>& gk, const linalg::matrix<size_t> & nk, const linalg::matrix<int32_t>& nkp1, const linalg::matrix<int32_t>& nkm1)
    {
        m_wk = wk;
        m_gk = gk;
        m_nk = nk;
        m_nkp1 = nkp1;
        m_nkm1 = nkm1;
        CALL_AND_RETHROW(initialise_topology());
        base_type::m_size = m_operator.shape(0);
    }
protected:
    void initialise_topology()
    {
        size_t nstates = m_nk.shape(0);     size_t N = m_nk.shape(1);
        m_operator.resize(nstates, nstates);   

        //we determine the number of non-zeros in the coupling element, while also setting the rowptr array
        auto rowptr = m_operator.rowptr();   rowptr[0] = 0;
        size_t nnz = 0;
        for(size_t i=0; i<nstates; ++i)
        {
            for(size_t j=0; j<N; ++j)
            {
                nnz += (m_nkm1(i, j) != -1 ? 1 : 0) + (m_nkp1(i, j) != -1 ? 1 : 0);
            }
            rowptr[i+1] = nnz;
        }
        m_operator.resize(nnz);

        //now we can set the colind and buffer values
        size_t counter = 0;
        auto colind = m_operator.colind();
        for(size_t i=0; i<nstates; ++i)
        {
            for(size_t j=0; j<N; ++j)
            {
                if(m_nkm1(i, j) != -1)
                {
                    colind[counter] = m_nkm1(i, j);
                    ++counter;
                }
            }

            for(size_t k=0; k<N; ++k)
            {
                size_t j = N-(k+1);
                if(m_nkp1(i, j) != -1)
                {
                    colind[counter] = m_nkp1(i, j);
                    ++counter;
                }
            }
        }
    }

protected:
    std::vector<real_type> m_wk;
    std::vector<real_type> m_gk;

    linalg::matrix<size_t> m_nk;
    linalg::matrix<int32_t> m_nkp1;
    linalg::matrix<int32_t> m_nkm1;

    linalg::csr_matrix<T, backend> m_operator;
#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<ttns::ops::primitive<T, backend> >(this)), "Failed to serialise sparse_matrix operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operator)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("wk", m_wk)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("gk", m_gk)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nk", m_nk)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nkp1", m_nkp1)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nkm1", m_nkm1)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<ttns::ops::primitive<T, backend> >(this)), "Failed to serialise sparse_matrix operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operator)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("wk", m_wk)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("gk", m_gk)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nk", m_nk)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nkp1", m_nkp1)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nkm1", m_nkm1)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
    }
#endif
};

#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(interaction_picture_sb_operator, ttns::ops::primitive)
#endif

#endif

