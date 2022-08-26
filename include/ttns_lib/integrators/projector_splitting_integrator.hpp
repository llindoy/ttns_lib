#ifndef HTUCKER_PROJECTOR_SPLITTING_INTEGRATOR_HPP
#define HTUCKER_PROJECTOR_SPLITTING_INTEGRATOR_HPP

#include <limits>

#include "../tdvp_core/operator_container.hpp"
#include "../tdvp_core/projector_splitting_engine.hpp"
#include "../tdvp_core/subspace_expansion_projector_splitting_engine.hpp"
//#include "../tdvp_core/subspace_expansion_full_two_site_projector_splitting_engine.hpp"
#include "../observables/matrix_element.hpp"

namespace ttns
{

template <typename T, typename backend = linalg::blas_backend>
class projector_splitting_integrator
{
public:
    using operator_type = sop_operator<T, backend>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

    projector_splitting_integrator() : m_nsteps(0), m_krylov_dim(4), m_krylov_tolerance(1e-12), m_nthreads(1) {}
    projector_splitting_integrator(size_type krylov_dim, real_type krylov_tolerance, size_type nthreads) : m_nsteps(0), m_krylov_dim(krylov_dim), m_krylov_tolerance(krylov_tolerance), m_nthreads(nthreads) {}
    virtual ~projector_splitting_integrator(){}
    virtual void initialise(const httensor<T, backend>& A, operator_type& op) = 0;
    void clear()
    {   
        m_nsteps = 0;
        m_krylov_dim = 4;
        m_krylov_tolerance = 1e-12;
        m_nthreads = 1;
    }
    void reset_nsteps(){m_nsteps = 0;}

    virtual real_type operator()(httensor<T, backend>& A, operator_type& op, T coeff) = 0;
    virtual void initialise_engine(const httensor<T, backend>& A);
    virtual real_type t() const = 0;

    size_type& krylov_dim() {return m_krylov_dim;}
    const size_type& krylov_dim() const {return m_krylov_dim;}

    size_type& nthreads() {return m_nthreads;}
    const size_type& nthreads() const {return m_nthreads;}


    real_type& krylov_tolerance() {return m_krylov_tolerance;}
    const real_type& krylov_tolerance() const {return m_krylov_tolerance;}

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("nsteps", m_nsteps)), "Failed to serialise projector splitting integrator.  Failed to serialise the number of steps.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("krylov_dim", m_krylov_dim)), "Failed to serialise projector splitting integrator.  Failed to serialise the number of steps.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("krylov_tolerance", m_krylov_tolerance)), "Failed to serialise projector splitting integrator.  Failed to serialise the number of steps.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nthreads", m_nthreads)), "Failed to serialise projector splitting integrator.  Failed to serialise the number of steps.");
    }
    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("nsteps", m_nsteps)), "Failed to serialise projector splitting integrator.  Failed to serialise the number of steps.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("krylov_dim", m_krylov_dim)), "Failed to serialise projector splitting integrator.  Failed to serialise the number of steps.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("krylov_tolerance", m_krylov_tolerance)), "Failed to serialise projector splitting integrator.  Failed to serialise the number of steps.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nthreads", m_nthreads)), "Failed to serialise projector splitting integrator.  Failed to serialise the number of steps.");
    }
#endif
protected:
    size_type m_nsteps;
    size_type m_krylov_dim;
    real_type m_krylov_tolerance;
    size_type m_nthreads;
};

template <typename T, typename backend = linalg::blas_backend>
class fixed_step_projector_splitting_integrator final : public projector_splitting_integrator<T, backend>
{
public:
    using operator_type = sop_operator<T, backend>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using base_type = projector_splitting_integrator<T, backend>;

    fixed_step_projector_splitting_integrator() = default;
    fixed_step_projector_splitting_integrator(const httensor<T, backend>& A, operator_type& op)  : m_dt(0)
    {   
        CALL_AND_HANDLE(initialise(A, op), "Failed to construct projector splitting integrator.");
    }

    fixed_step_projector_splitting_integrator(const httensor<T, backend>& A, operator_type& op, size_type krylov_dim, real_type krylov_tolerance, size_type nthreads) : base_type(krylov_dim, krylov_tolerance, nthreads), m_dt(0)
    {   
        CALL_AND_HANDLE(initialise(A, op), "Failed to construct projector splitting integrator.");
    }

    fixed_step_projector_splitting_integrator(const fixed_step_projector_splitting_integrator& o) = default;
    fixed_step_projector_splitting_integrator(fixed_step_projector_splitting_integrator&& o) = default;

    fixed_step_projector_splitting_integrator& operator=(const fixed_step_projector_splitting_integrator& o) = default;
    fixed_step_projector_splitting_integrator& operator=(fixed_step_projector_splitting_integrator&& o) = default;
    ~fixed_step_projector_splitting_integrator(){}

    void initialise(const httensor<T, backend>& A, operator_type& op)
    {
        try
        {
            CALL_AND_HANDLE(m_engine.initialise(A, base_type::m_krylov_dim, base_type::m_krylov_tolerance, base_type::m_nthreads), "Failed to initialise projector splitting integrator object.");
            CALL_AND_HANDLE(m_hbuf.resize(A, op), "Failed to resize the hamiltonian buffer object.");
            CALL_AND_HANDLE(m_mel.resize(A), "Failed to resize the matrix element object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise projector splitting integrator object.");
        }
    }

    void clear()
    {
        try
        {
            base_type::clear();
            m_dt = 0.0;
            CALL_AND_HANDLE(m_engine.clear(), "Failed to clear projector splitting engine.");
            CALL_AND_HANDLE(m_hbuf.clear(), "Failed to clear Hamiltonian buffer.");
            CALL_AND_HANDLE(m_mel.clear(), "Failed to clear Hamiltonian buffer.");
        
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear projector splitting integrator object.");
        }
    }

    real_type operator()(httensor<T, backend>& A, operator_type& op, T coeff)
    {
        if(base_type::m_nsteps == 0 )
        {
            //if it is our first step then we need to prepare the Hamiltonian buffer for evolution and attempt to 
            //add in unoccupied single particle functions

            m_engine.coefficient() = coeff;
        
            real_type tevolve = 0;
            real_type dt = m_dt/1e5;
            real_type dt_evolve = (dt + tevolve <= m_dt ? dt : m_dt - tevolve);
            m_engine.dt() = dt_evolve;      

            CALL_AND_HANDLE(m_engine.prepare_evolution(A, op, m_hbuf), "Failed to prepare the hamiltonian buffer for evolution.");

            while(tevolve < m_dt)
            {
                dt_evolve = (dt + tevolve <= m_dt ? dt : m_dt - tevolve);
                m_engine.dt() = dt_evolve;      m_engine.t() = tevolve;
                CALL_AND_HANDLE(m_engine(A, op, m_hbuf), "Failed to apply the projector splitting integrator to evolve a hierarchical tucker tensor.");
                tevolve += dt;
                dt *= 10;
            }
        }
        else
        {
            m_engine.coefficient() = coeff;
            m_engine.dt() = m_dt;      m_engine.t() = base_type::m_nsteps*m_dt;
            CALL_AND_HANDLE(m_engine(A, op, m_hbuf), "Failed to apply the projector splitting integrator to evolve a hierarchical tucker tensor.");
        }
        ++base_type::m_nsteps;
        return m_dt;
    }
    
    void initialise_engine(const httensor<T, backend>& A)
    {
        try
        {
            CALL_AND_HANDLE(m_engine.initialise(A, base_type::m_krylov_dim, base_type::m_krylov_tolerance, base_type::m_nthreads), "Failed to initialise projector splitting integrator object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise projector splitting integrator object.");
        }
    }

    const real_type& dt() const{return m_dt;}
    real_type& dt(){return m_dt;}

    void set_dt(real_type dt) {m_dt = dt;}
    real_type t() const{return m_dt*base_type::m_nsteps;}

    size_type nh_applications() const{return m_engine.nh_applications();}
    matrix_element<T, backend>& mel(){return m_mel;}
protected:
    projector_splitting_engine<T, backend> m_engine;
    operator_container<T, backend> m_hbuf;
    real_type m_dt;
    matrix_element<T, backend> m_mel;

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise projector splitting integrator.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("hamiltonian_container", m_hbuf)), "Failed to serialise projector splitting integrator.  Failed to serialise the hamiltonian container object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("dt", m_dt)), "Failed to serialise projector splitting integrator.  Failed to serialise the time step.");
        //CALL_AND_HANDLE(ar(cereal::make_nvp("mel", m_mel)), "Failed to serialise projector splitting integrator.  Failed to serialise the time step.");
    }
    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise projector splitting integrator.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("hamiltonian_container", m_hbuf)), "Failed to serialise projector splitting integrator.  Failed to serialise the hamiltonian container object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("dt", m_dt)), "Failed to serialise projector splitting integrator.  Failed to serialise the time step.");
        //CALL_AND_HANDLE(ar(cereal::make_nvp("mel", m_mel)), "Failed to serialise projector splitting integrator.  Failed to serialise the time step.");
    }
#endif

};  //class projector_splitting_integrator


template <typename T, typename backend = linalg::blas_backend>
class subspace_expansion_projector_splitting_integrator final : public projector_splitting_integrator<T, backend>
{
public:
    using operator_type = sop_operator<T, backend>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using base_type = projector_splitting_integrator<T, backend>;

    subspace_expansion_projector_splitting_integrator() = default;
    subspace_expansion_projector_splitting_integrator(const httensor<T, backend>& A, operator_type& op)  : m_dt(0)
    {   
        CALL_AND_HANDLE(initialise(A, op), "Failed to construct projector splitting integrator.");
    }

    subspace_expansion_projector_splitting_integrator(const httensor<T, backend>& A, operator_type& op, size_type krylov_dim, real_type krylov_tolerance, size_type nthreads) : base_type(krylov_dim, krylov_tolerance, nthreads), m_dt(0)
    {   
        CALL_AND_HANDLE(initialise(A, op), "Failed to construct projector splitting integrator.");
    }

    subspace_expansion_projector_splitting_integrator(const subspace_expansion_projector_splitting_integrator& o) = default;
    subspace_expansion_projector_splitting_integrator(subspace_expansion_projector_splitting_integrator&& o) = default;

    subspace_expansion_projector_splitting_integrator& operator=(const subspace_expansion_projector_splitting_integrator& o) = default;
    subspace_expansion_projector_splitting_integrator& operator=(subspace_expansion_projector_splitting_integrator&& o) = default;
    ~subspace_expansion_projector_splitting_integrator(){}

    void initialise(const httensor<T, backend>& A, operator_type& op)
    {
        try
        {
            CALL_AND_HANDLE(m_hbuf.resize(A, op, true), "Failed to resize the hamiltonian buffer object.");
            CALL_AND_HANDLE(m_engine.initialise(A, m_hbuf, base_type::m_krylov_dim, base_type::m_krylov_tolerance, base_type::m_nthreads), "Failed to initialise projector splitting integrator object.");
            CALL_AND_HANDLE(m_mel.resize(A, true), "Failed to resize the matrix element object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise projector splitting integrator object.");
        }
    }

    void clear()
    {
        try
        {
            base_type::clear();
            m_dt = 0.0;
            CALL_AND_HANDLE(m_engine.clear(), "Failed to clear projector splitting engine.");
            CALL_AND_HANDLE(m_hbuf.clear(), "Failed to clear Hamiltonian buffer.");
            CALL_AND_HANDLE(m_mel.clear(), "Failed to clear Hamiltonian buffer.");
        
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear projector splitting integrator object.");
        }
    }

    real_type operator()(httensor<T, backend>& A, operator_type& op, T coeff)
    {
        try
        {
            bool resized = false;
            if(base_type::m_nsteps == 0 )
            {
                //if it is our first step then we need to prepare the Hamiltonian buffer for evolution and attempt to 
                //add in unoccupied single particle functions

                m_engine.coefficient() = coeff;
                real_type tevolve = 0;
                m_engine.dt() = m_dt;      
            
                CALL_AND_HANDLE(resized = m_engine.prepare_evolution(A, op, m_hbuf), "Failed to prepare the hamiltonian buffer for evolution.");
                real_type dt = m_dt/1e5;
                real_type dt_evolve = (dt + tevolve <= m_dt ? dt : m_dt - tevolve);
                m_engine.dt() = dt_evolve;      

                while(tevolve < m_dt)
                {
                    dt_evolve = (dt + tevolve <= m_dt ? dt : m_dt - tevolve);
                    m_engine.dt() = dt_evolve;      m_engine.t() = tevolve;
                    bool resized_local = false;
                    CALL_AND_HANDLE(resized_local = m_engine(A, op, m_hbuf), "Failed to apply the projector splitting integrator to evolve a hierarchical tucker tensor.");
                    if(resized_local){resized = true;}
                    tevolve += dt;
                    dt *= 10;
                }
            }
            else
            {
                m_engine.coefficient() = coeff;
                m_engine.dt() = m_dt;      m_engine.t() = base_type::m_nsteps*m_dt;
                CALL_AND_HANDLE(resized = m_engine(A, op, m_hbuf), "Failed to apply the projector splitting integrator to evolve a hierarchical tucker tensor.");
            }
            if(resized){CALL_AND_HANDLE(m_mel.resize(A, false), "Failed to resize the matrix element object.");}
            ++base_type::m_nsteps;
            return m_dt;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply projector splitting integrator.");
        }
    }
    
    void initialise_engine(const httensor<T, backend>& A)
    {
        try
        {
            CALL_AND_HANDLE(m_engine.initialise(A, m_hbuf, base_type::m_krylov_dim, base_type::m_krylov_tolerance, base_type::m_nthreads), "Failed to initialise projector splitting integrator object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise projector splitting integrator object.");
        }
    }

    bool& only_apply_when_no_unoccupied(){return m_engine.only_apply_when_no_unoccupied();}
    const bool& only_apply_when_no_unoccupied() const{return m_engine.only_apply_when_no_unoccupied();}

    const real_type& dt() const{return m_dt;}
    real_type& dt(){return m_dt;}

    const real_type& spawning_threshold() const{return m_engine.spawning_threshold();}
    real_type& spawning_threshold(){return m_engine.spawning_threshold();}

    const real_type& unoccupied_threshold() const{return m_engine.unoccupied_threshold();}
    real_type& unoccupied_threshold(){return m_engine.unoccupied_threshold();}

    size_type& minimum_unoccupied(){return m_engine.minimum_unoccupied();}    
    const size_type& minimum_unoccupied() const {return m_engine.minimum_unoccupied();}    

    size_type& neigenvalues(){return m_engine.neigenvalues();}    
    const size_type& neigenvalues() const {return m_engine.neigenvalues();}    

    real_type t() const{return m_dt*base_type::m_nsteps;}

    size_type nh_applications() const{return m_engine.nh_applications();}
    matrix_element<T, backend>& mel(){return m_mel;}

    const subspace_expansion_projector_splitting_engine<T, backend>& engine() const{return m_engine;}
    subspace_expansion_projector_splitting_engine<T, backend>& engine(){return m_engine;}
protected:
    subspace_expansion_projector_splitting_engine<T, backend> m_engine;
    //subspace_expansion_full_two_site_projector_splitting_engine<T, backend> m_engine;
    operator_container<T, backend> m_hbuf;
    real_type m_dt;
    matrix_element<T, backend> m_mel;

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise projector splitting integrator.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("hamiltonian_container", m_hbuf)), "Failed to serialise projector splitting integrator.  Failed to serialise the hamiltonian container object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("dt", m_dt)), "Failed to serialise projector splitting integrator.  Failed to serialise the time step.");
        //CALL_AND_HANDLE(ar(cereal::make_nvp("mel", m_mel)), "Failed to serialise projector splitting integrator.  Failed to serialise the time step.");
    }
    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise projector splitting integrator.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("hamiltonian_container", m_hbuf)), "Failed to serialise projector splitting integrator.  Failed to serialise the hamiltonian container object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("dt", m_dt)), "Failed to serialise projector splitting integrator.  Failed to serialise the time step.");
        //CALL_AND_HANDLE(ar(cereal::make_nvp("mel", m_mel)), "Failed to serialise projector splitting integrator.  Failed to serialise the time step.");
    }
#endif

};  //class projector_splitting_integrator
}   //namespace ttns

#endif

