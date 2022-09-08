#ifndef HTUCKER_PROJECTOR_SPLITTING_ENGINE_HPP
#define HTUCKER_PROJECTOR_SPLITTING_ENGINE_HPP


#include <omp.hpp>
#include <krylov_integrator.hpp>

#include "tdvp_algorithm_engine.hpp"
#include "projector_splitting_evolution_functions.hpp"

namespace ttns
{

template <typename T, typename backend = linalg::blas_backend>
class projector_splitting_engine : public tdvp_algorithm_base<projector_splitting_engine, T, backend>
{
protected:
    using base_type = tdvp_algorithm_base<projector_splitting_engine, T, backend>;
    using vec_type = typename base_type::vec_type;
    using mat_type = typename base_type::mat_type;
    using triad_type = typename base_type::triad_type;
    
    using opdata_type = typename base_type::opdata_type;

    using operator_type = typename base_type::operator_type;

    using size_type = typename base_type::size_type;
    using real_type = typename base_type::real_type;

    using integ_type = utils::krylov_integrator<T, backend>;

    using hnode = typename base_type::hnode;
    using opnode = typename base_type::opnode;
    using matnode = typename base_type::matnode;

    friend base_type;
public:
    projector_splitting_engine() : base_type(), m_krylov_dim(0), m_num_threads(1), m_dt(0), m_t(0), m_coeff(1) {}
    projector_splitting_engine(const httensor<T, backend>& A, size_type krylov_dim = 4, real_type krylov_tolerance = 1e-12, size_type num_threads = 1)  : base_type(), m_num_threads(1)
    {
        CALL_AND_HANDLE(initialise(A, krylov_dim, krylov_tolerance, num_threads), "Failed to construct projector_splitting_engine.");
    }   
    projector_splitting_engine(const projector_splitting_engine& o) = default;
    projector_splitting_engine(projector_splitting_engine&& o) = default;

    projector_splitting_engine& operator=(const projector_splitting_engine& o) = default;
    projector_splitting_engine& operator=(projector_splitting_engine&& o) = default;
    
    void initialise(const httensor<T, backend>& A, size_type krylov_dim = 4, real_type krylov_tolerance = 1e-12, size_type num_threads = 1)
    {
        try
        {
            ASSERT(num_threads > 0, "Invalid number of threads.");
            ASSERT(A.is_orthogonalised(), "The input hierarchical tucker tensor must have been orthogonalised.");

            CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");

            m_num_threads = num_threads;

            //set up openmp environment
            omp_set_dynamic(0);         omp_set_num_threads(m_num_threads);
            mkl_set_dynamic(0);         blas_set_num_threads(m_num_threads);
            
            //set the krylov dimension
            m_krylov_dim =  krylov_dim;

            size_type maxcapacity = 0;
            for(const auto& a : A){size_type capacity = a().capacity();    if(capacity > maxcapacity){maxcapacity = capacity;}}
            CALL_AND_HANDLE(m_integ.resize(krylov_dim, maxcapacity), "Failed to initialise the krylov subspace engine object.");
            m_integ.error_tolerance() = krylov_tolerance;

            base_type::initialise_base(A);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
        }
    }

    void clear()
    {
        try
        {
            CALL_AND_RETHROW(base_type::clear_base());
            CALL_AND_HANDLE(m_integ.clear(), "Failed to clear the krylov subspace engine.");
            m_krylov_dim = 0;
            m_num_threads = 0;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
        }
    }

    void advance_half_step(){m_t += m_dt/2.0;}

    T& coefficient(){return m_coeff;}
    const T& coefficient() const{return m_coeff;}

    real_type& t(){return m_t;}
    const real_type& t() const{return m_t;}

    real_type& dt(){return m_dt;}
    const real_type& dt() const{return m_dt;}
protected:
    size_type nbuffers() const
    {
#ifdef PARALLELISE_FOR_LOOPS
        return m_num_threads;
#else
        return 1;
#endif
    }

    void update_node_tensor(hnode& A, opnode& h, operator_type& op)
    {                    
        if(!A.is_leaf())
        {
#ifdef PARALLELISE_FOR_LOOPS
            CALL_AND_HANDLE
            (
                base_type::m_nh_evals += m_integ(A().as_matrix(), m_dt/2.0, m_coeff, ceb, h, A.hrank(), A.dims(), base_type::m_HA, base_type::m_temp, base_type::m_temp2),
                "Failed to evolve the branch coefficient matrix."
            );
#else
            CALL_AND_HANDLE
            (
                base_type::m_nh_evals += m_integ(A().as_matrix(), m_dt/2.0, m_coeff, ceb, h, A.hrank(), A.dims(), base_type::m_HA[0], base_type::m_temp[0], base_type::m_temp2[0]),
                "Failed to evolve the branch coefficient matrix."
            );
#endif
        }
        else
        {
            //if we are at a leaf node we update its child operators.  This is simply done by calling the update function
            //for its hamiltonia operator object
            CALL_AND_HANDLE(op.update(A.leaf_index(), m_t, m_dt/2.0), "Failed to update primitive Hamiltonian object.");

#ifdef PARALLELISE_FOR_LOOPS
            CALL_AND_HANDLE
            (
                base_type::m_nh_evals += m_integ(A().as_matrix(), m_dt/2.0, m_coeff, cel, h, op, base_type::m_HA, base_type::m_temp),
                "Failed to evolve the leaf coefficient matrix."
            );
#else
            CALL_AND_HANDLE
            (
                base_type::m_nh_evals += m_integ(A().as_matrix(), m_dt/2.0, m_coeff, cel, h, op, base_type::m_HA[0], base_type::m_temp[0]),
                "Failed to evolve the leaf coefficient matrix."
            );
#endif
        }

    }

    void update_R_tensor(matnode& r, opnode& h)   
    {
#ifdef PARALLELISE_FOR_LOOPS
        CALL_AND_HANDLE(m_integ(r(), -m_dt/2.0, m_coeff, fha, h, base_type::m_HA, base_type::m_temp), "Failed to time evolve the r matrix backwards in time.");
#else   
        CALL_AND_HANDLE(m_integ(r(), -m_dt/2.0, m_coeff, fha, h, base_type::m_HA[0]), "Failed to time evolve the r matrix backwards in time.");
#endif
    }

protected:
    full_hamiltonian_action<T, backend> fha;        //used for evolving the r matrix under the full Hamiltonian
    coefficient_evolution_leaf<T, backend> cel;     //used for evolving the coefficient tensors of leaf nodes
    coefficient_evolution_branch<T, backend> ceb;   //used for evolving the coefficient tensors of branch nodes

    //the krylov subspace engine
    integ_type m_integ;

    size_type m_krylov_dim;
    size_type m_num_threads;

    real_type m_dt;
    real_type m_t;
    T m_coeff;
    
};  //class projector_splitting_engine

}   //namespace ttns

#endif  //HTUCKER_PROJECTOR_SPLITTING_ENGINE_HPP//

