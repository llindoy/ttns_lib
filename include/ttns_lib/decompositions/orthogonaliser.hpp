#ifndef HTUCKER_ORTHOGONALISER_HPP
#define HTUCKER_ORTHOGONALISER_HPP


#include <memory>
#include "../utils/zip.hpp"
#include "../tdvp_core/decomposition_engine.hpp"
#include "../tdvp_core/leaf_to_root_decomposition.hpp"

namespace ttns
{

//might make the type of orthogonalisation a template parameter (all for qr)
template <typename T, typename backend = blas_backend>
class orthogonaliser
{
public:
    using real_type = typename tmp::get_real_type<T>::type;
    using size_type = typename httensor<T,backend>::size_type;
    using engine_type = decomposition_engine<T, backend, false>;
    using matrix_type = matrix<T, backend>;
    using l2r = leaf_to_root_decomposition_engine<T, backend>;
    
public:
    orthogonaliser() {}
    orthogonaliser(const httensor<T, backend>& A, bool use_capacity = false) 
    {
        CALL_AND_HANDLE(resize(A, use_capacity), "Failed to constructor orthogonaliser object.  Failed to resize buffers.");
    }

    void resize(const httensor<T, backend>& A, bool use_capacity = false)
    {
        try
        {
            CALL_AND_HANDLE(m_r.construct_topology(A), "Failed to construct the topology of the r tensor.");
            
            size_type max_size = 0;
            for(auto z : zip(m_r, A))
            {
                auto& r = std::get<0>(z);   const auto& a = std::get<1>(z);
                size_type asize = use_capacity ? a().capacity() : a().size();
                if(asize > max_size){max_size = asize;}
                CALL_AND_HANDLE(l2r::resize_r_matrix(a(), r(), use_capacity), "Failed to resize elements of the r tensor.");
            }
            CALL_AND_HANDLE(m_u.resize(1, max_size), "Failed to resize u tensor.");

            CALL_AND_HANDLE(m_temp.resize(1, max_size), "Failed to resize temporary tensor.");

            try
            {
                m_ortho_engine.template resize<l2r>(A, m_u, m_r, use_capacity);
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to resize the decomposition engine object.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize orthogonaliser object.");
        }
    }

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_u.clear(), "Failed to clear the u array.");
            CALL_AND_HANDLE(m_r.clear(), "Failed to clear the r array.");
            CALL_AND_HANDLE(m_temp.clear(), "Failed to clear the temporary working array.");
            CALL_AND_HANDLE(m_ortho_engine.clear(), "Failed to clear the orthogonalisation engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear orthogonaliser object.");
        }
    }


    void operator()(httensor<T, backend>& A)
    {
        try
        {
            //check whether A is already orthogonalised.  If it is we don't need to do anything.
            if(!A.is_orthogonalised())
            {
                for(auto z : rzip(A, m_r))
                {
                    auto& a = std::get<0>(z);
                    auto& r = std::get<1>(z);
                    auto& u = m_u;
                    m_u.resize(a().shape(0), a().shape(1));
                    m_temp.resize(a().shape(0), a().shape(1));

                    if(!a.is_root())
                    {
                        //evaluate the decomposition acting on the node
                        CALL_AND_HANDLE(l2r::evaluate(m_ortho_engine, a, u, r), "Failed to evaluate the leaf_to_root_decomposition for a given node.");
                        
                        //now we apply the decomposition
                        CALL_AND_HANDLE(l2r::apply(a, u, r, m_temp), "Failed when applying the result of the decomposition.");
                    }
                }
                A.set_is_orthogonalised();
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("orthogonalising the hierarchical tucker tensor object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to orthogonalise the hierarchical tucker tensor object.");
        }
    }

protected:
    //temporary objects for storing the intermediate quantities required for computing the orthogonalisation of the hierarchical tucker tensor
    matrix_type m_u;
    tree<matrix_type> m_r;
    matrix_type m_temp;

    //an object wrapping the singular value decomposition.  This also stores the workspace arrays required for the computations.
    engine_type m_ortho_engine;
};

}   //namespace ttns

#endif //HTUCKER_ORTHOGONALISER_HPP//
