#ifndef HTUCKER_SUBSPACE_EXPANSION_PROJECTOR_SPLITTING_ENGINE_HPP
#define HTUCKER_SUBSPACE_EXPANSION_PROJECTOR_SPLITTING_ENGINE_HPP

#include <random>
#include <omp.hpp>
#include <krylov_integrator.hpp>

#include "tdvp_algorithm_engine.hpp"


#include "projector_splitting_evolution_functions.hpp"
#include "two_site_energy_variations.hpp"

namespace ttns
{


template <typename T, typename backend>
class subspace_expansion_projector_splitting_engine : public tdvp_algorithm_base<subspace_expansion_projector_splitting_engine, T, backend>
{
protected:
    using twosite = two_site_variations<T, backend>;
    using base_type = tdvp_algorithm_base<subspace_expansion_projector_splitting_engine, T, backend>;
    using vec_type = typename base_type::vec_type;
    using mat_type = typename base_type::mat_type;
    using triad_type = typename base_type::triad_type;
    
    using opdata_type = typename base_type::opdata_type;

    using operator_type = typename base_type::operator_type;

    using size_type = typename base_type::size_type;
    using real_type = typename base_type::real_type;

    using integ_type = utils::krylov_integrator<T, backend>;

    using hnode = typename base_type::hnode;
    using hdata = typename base_type::hdata;
    using opnode = typename base_type::opnode;
    using matnode = typename base_type::matnode;
    using dmat_type = typename base_type::dmat_type;

    friend base_type;
public:
    subspace_expansion_projector_splitting_engine() : base_type(), m_twosite(), m_krylov_dim(0), m_num_threads(1), m_dt(0), m_t(0), m_coeff(1), m_spawning_threshold(-1), m_unoccupied_threshold(-1), m_eigenvalue_tolerance(1e-6), m_minimum_unoccupied(0), m_neigenvalues(2), m_only_apply_when_no_unoccupied(false) {}
    subspace_expansion_projector_splitting_engine(const httensor<T, backend>& A, const operator_container<T, backend>& ham, size_type krylov_dim = 4, real_type krylov_threshold = 1e-12, size_type num_threads = 1, size_type seed = 0)  : base_type(), m_twosite(seed), m_num_threads(1), m_spawning_threshold(-1), m_unoccupied_threshold(-1), m_eigenvalue_tolerance(1e-6), m_minimum_unoccupied(0), m_neigenvalues(2), m_only_apply_when_no_unoccupied(false)
    {
        CALL_AND_HANDLE(initialise(A, ham, krylov_dim, krylov_threshold, num_threads), "Failed to construct subspace_expansion_projector_splitting_engine.");
    }   
    subspace_expansion_projector_splitting_engine(const subspace_expansion_projector_splitting_engine& o) = default;
    subspace_expansion_projector_splitting_engine(subspace_expansion_projector_splitting_engine&& o) = default;

    subspace_expansion_projector_splitting_engine& operator=(const subspace_expansion_projector_splitting_engine& o) = default;
    subspace_expansion_projector_splitting_engine& operator=(subspace_expansion_projector_splitting_engine&& o) = default;
    
    void initialise(const httensor<T, backend>& A, const operator_container<T, backend>& ham, size_type krylov_dim = 4, real_type krylov_threshold = 1e-12, size_type num_threads = 1)
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

            size_type maxcapacity = 0;  size_type maxnmodes = 0;
            for(const auto& a : A)
            {
                size_type capacity = a().capacity();    if(capacity > maxcapacity){maxcapacity = capacity;}
                size_type nmodes = a().nmodes();    if(nmodes > maxnmodes){maxnmodes = nmodes;}
            }

            size_type max_two_site_energy_terms = 0;

            for(const auto& h : ham.op())
            {
                size_type two_site_energy_terms = twosite::get_nterms(h());
                if(two_site_energy_terms > max_two_site_energy_terms){max_two_site_energy_terms = two_site_energy_terms;}
            }

            CALL_AND_HANDLE(m_2s_1.resize(max_two_site_energy_terms), "Failed to resize two site spf buffer.");
            CALL_AND_HANDLE(m_2s_2.resize(max_two_site_energy_terms), "Failed to resize two site mf buffer.");

            for(size_type i = 0; i < max_two_site_energy_terms; ++i)
            {
                CALL_AND_HANDLE(m_2s_1[i].reallocate(maxcapacity), "Failed to reallocate two site spf buffer.");
                CALL_AND_HANDLE(m_2s_2[i].reallocate(maxcapacity), "Failed to reallocate two site mf buffer.");
            }

            m_inds.resize(maxnmodes);
            m_dim.resize(maxnmodes);

            m_U.reallocate(maxcapacity);
            m_V.reallocate(maxcapacity);
            m_rvec.reallocate(maxcapacity);
            m_trvec.reallocate(maxcapacity);
            m_trvec2.reallocate(maxcapacity);

            CALL_AND_HANDLE(m_integ.resize(krylov_dim, maxcapacity), "Failed to initialise the krylov subspace engine object.");
            m_integ.error_tolerance() = krylov_threshold;

            base_type::initialise_base(A);
            m_onesite_expansions = 0;
            m_twosite_expansions = 0;

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
            for(size_type i = 0; i < m_2s_1.size(); ++i)
            {
                CALL_AND_HANDLE(m_2s_1[i].clear(), "Failed to clear the rvec object.");
                CALL_AND_HANDLE(m_2s_2[i].clear(), "Failed to clear the rvec object.");
            }
            CALL_AND_HANDLE(m_inds.clear(), "Failed to clear temporary inds array.");
            CALL_AND_HANDLE(m_integ.clear(), "Failed to clear the krylov subspace engine.");
            CALL_AND_HANDLE(m_rvec.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_trvec.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_trvec2.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_2s_1.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_2s_2.clear(), "Failed to clear the rvec object.");
            m_krylov_dim = 0;
            m_num_threads = 0;
            m_onesite_expansions = 0;
            m_twosite_expansions = 0;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
        }
    }

    void advance_half_step(){m_t += m_dt/2.0;}


    bool& only_apply_when_no_unoccupied(){return m_only_apply_when_no_unoccupied;}
    const bool& only_apply_when_no_unoccupied() const{return m_only_apply_when_no_unoccupied;}

    T& coefficient(){return m_coeff;}
    const T& coefficient() const{return m_coeff;}

    real_type& t(){return m_t;}
    const real_type& t() const{return m_t;}

    real_type& dt(){return m_dt;}
    const real_type& dt() const{return m_dt;}

    real_type& spawning_threshold(){return m_spawning_threshold;}
    const real_type& spawning_threshold() const{return m_spawning_threshold;}

    real_type& unoccupied_threshold(){return m_unoccupied_threshold;}
    const real_type& unoccupied_threshold() const{return m_unoccupied_threshold;}

    size_type& minimum_unoccupied(){return m_minimum_unoccupied;}    
    const size_type& minimum_unoccupied() const {return m_minimum_unoccupied;}    

    real_type& eigenvalue_tolerance(){return m_eigenvalue_tolerance;}
    const real_type& eigenvalue_tolerance() const{return m_eigenvalue_tolerance;}

    size_type& neigenvalues(){return m_neigenvalues;}    
    const size_type& neigenvalues() const {return m_neigenvalues;}    

    template <typename Arg>
    void set_rng(const Arg& rng){m_twosite.set_rng(rng);}

protected:
    size_type nbuffers() const
    {
#ifdef PARALLELISE_FOR_LOOPS
        return m_num_threads;
#else
        return 1;
#endif
    }

    size_type get_n_unoccupied(const dmat_type& pops)
    {
        size_type nunocc = 0;
        for(size_type i = 0; i < pops.size(); ++i)
        {
            if(pops(i, i) < m_unoccupied_threshold){++nunocc;}
        }
        return nunocc;
    }

    //perform the subspace expansion as we are moving down a tree.  This requires us to evaluate the optimal functions to add 
    //into A2.  For A1 they will be overwriten by the r matrix in the next step so we just 
    //here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
    bool subspace_expansion_down(hnode& A1, hnode& A2, mat_type& r, const dmat_type& pops, opnode& h, const operator_type& op)
    {
        try
        {
            size_type mode = A1.child_id();

            //first check if the mode we are interested in has the capacity to be expanded
            if(A2().dim(mode) >= A2().max_dim(mode) || A1().hrank() >= A1().max_hrank()){return false;}

            //get the maximum number of elements that we can add to the tensor
            size_type max_add = A2().max_dim(mode) - A2().dim(mode);
            bool subspace_expanded = false;

            //determine if we should attempt subspace expansion.
            
            auto A2tens = A2().as_rank_3(mode);
            size_type max_dimension = A2tens.shape(0)*A2tens.shape(2);
            size_type curr_dim = A2().dim(mode);

            //if the matricisation of the tensor we are currently trying to expand is square then we don't even attempt to expand
            if(A2tens.shape(1) >= max_dimension){return false;}

            size_type n_unocc = get_nunoccupied(pops);
            if(m_only_apply_when_no_unoccupied && n_unocc >= m_minimum_unoccupied){return false;}

            //get the maximum number of elements that we can add
            size_type max_add2 = max_dimension - A2tens.shape(1);

            //we can add min(max_add, max_add2) elements
            max_add = max_add < max_add2 ? max_add : max_add2;

            //now determine the number of unoccupied functions, and using this determine the minimum number of terms we need to add
            size_type required_terms = 0;
            if(n_unocc < m_minimum_unoccupied)
            {
                required_terms = m_minimum_unoccupied - n_unocc;
            }
            //and limit the required number of terms to the maximum number of terms we can add
            if(required_terms > max_add){required_terms = max_add;}


            size_type nadd = 0;
            //if the other tensor is not square then we are in a regime where the two-site algorithm will provide a search direction so we should attempt it
            if(A1().shape(1) < A1().shape(0) && A1().shape(1) != 1)
            {
                //now resize the onesite buffer objects used for evaluation of the two site (projected energy) so that everything is the correct size
                size_type nterms = twosite::get_nterms(h()); 
                ASSERT(m_2s_1.size() >= nterms, "Unable to store temporaries needed for computing the two-site energy.  The buffers are not sufficient.");
                CALL_AND_HANDLE(resize_two_site_energy_buffers(nterms, A1(), A2(), mode), "Failed to resize the objects needed to compute the two-site energy.");
                CALL_AND_HANDLE(m_inds.resize(nterms), "Failed to resize indices array.");

                twosite::set_indices(h(), m_inds); 

                //compute the one site objects that are used to compute the two site projected energy
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_lower(A1, h, op, m_2s_1, base_type::m_HA, base_type::m_temp, m_inds, r, true), 
                                "Failed to construct the component of the two site energy acting on the lower site.");
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_upper(A2, h, m_2s_2, base_type::m_HA, base_type::m_temp, base_type::m_temp2, m_inds, true), 
                                "Failed to construct the component of the two site energy acting on the upper site.");

                //now we compute the singular value using the sparse functions
                CALL_AND_HANDLE(m_twosite.generate_orthogonal_trial_vector(A2(), mode, m_rvec), "Failed to generate othrogonal tensor.");
                
                //now compute the eigenvectors using this 
                bool mconjm = true;
                size_type maxkrylov_dim = m_krylov_dim;
                if(m_rvec.size() < maxkrylov_dim){maxkrylov_dim = m_rvec.size();}
                real_type scale_factor = 1;

                //computes the complex conjugate of the right singular vectors
                CALL_AND_HANDLE(m_integ.eigs(m_rvec, m_S, m_V, scale_factor, m_neigenvalues, maxkrylov_dim, m_eigenvalue_tolerance, m_twosite, m_2s_1, m_2s_2, m_trvec, m_trvec2, base_type::m_temp[0], mconjm), "Failed to compute sparse svd.");

                CALL_AND_HANDLE(m_V = linalg::conj(m_V), "Failed to conjugate the right singular vectors.");

                for(size_type i = 0; i < m_S.size(); ++i)
                {
                    real_type sv = 0;
                    //check if any of the dominant svds of the Hamiltonian acting on the twosite coefficient tensor are occupied through the half step
                    if(std::real(m_S(i, i)) > 0){sv = std::sqrt(std::real(m_S(i, i)))*(m_dt/2.0);}   
                    //std::cerr << i << " " << m_S(i, i) << " " << sv <<  " " << m_spawning_threshold << std::endl;
                    if(!std::isnan(sv))
                    {
                        if(sv > m_spawning_threshold){++nadd;}
                    }
                }

                //std::cerr << nadd << std::endl;
                //if(nadd == 0){++nadd;}
                if(nadd > max_add){nadd = max_add;}
                if(nadd != 0)
                {
                    //expand the A1 tensor zero padding
                    CALL_AND_HANDLE(twosite::expand_tensor(A1(), base_type::m_temp[0], A1.nmodes(), nadd, m_dim), "Failed to expand A1 tensor.");

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, base_type::m_temp[0], nadd), "Failed to expand R matrix.");

                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(twosite::expand_tensor(A2(), base_type::m_temp[0], mode, m_V, nadd, m_dim), "Failed to expand A2 tensor.");

                    //resize all of the working buffers. 
                    CALL_AND_HANDLE(base_type::resize_working_buffers(A2().shape(0), A2().shape(1)), "Failed to resize working arrays.");

                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                    subspace_expanded = true;
                }
                ++m_twosite_expansions;
            } 
            //otherwise we have a square other tensor and so the two-site algorithm will fail to provide a search direction so instead add on
            //a random search direction to this tensor
            else
            {
                while(nadd < required_terms)
                {
                    size_type add = 1;
                    //now we compute the singular value using the sparse functions
                    CALL_AND_HANDLE(m_twosite.generate_orthogonal_trial_vector(A2(), mode, m_rvec), "Failed to generate othrogonal tensor.");

                    CALL_AND_HANDLE(m_V.resize(1, m_rvec.size()), "Failed to resize V array.");
                    CALL_AND_HANDLE(m_V[0] = m_rvec, "Failed to copy random vector to V");

                    //expand the A1 tensor zero padding
                    CALL_AND_HANDLE(twosite::expand_tensor(A1(), base_type::m_temp[0], A1.nmodes(), add, m_dim), "Failed to expand A1 tensor.");

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, base_type::m_temp[0], add), "Failed to expand R matrix.");

                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(twosite::expand_tensor(A2(), base_type::m_temp[0], mode, m_V, add, m_dim), "Failed to expand A2 tensor.");

                    //resize all of the working buffers. 
                    CALL_AND_HANDLE(base_type::resize_working_buffers(A2().shape(0), A2().shape(1)), "Failed to resize working arrays.");

                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");
                
                    ++nadd;
                    subspace_expanded = true;
                }
                ++m_onesite_expansions;
            }
            return subspace_expanded;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing down the tree.");
        }
    }

    //here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
    bool subspace_expansion_up(hnode& A1, hnode& A2, mat_type& r, const dmat_type&  pops, opnode& h, const operator_type& op)
    {
        try
        {
            size_type mode = A1.child_id();

            //first check if the mode we are interested in has the capacity to be expanded
            if(A1().hrank() >= A1().max_hrank()){return false;}

            //get the maximum number of elements that we can add to the tensor
            size_type max_add = A1().max_hrank() - A1().hrank();
            bool subspace_expanded = false;

            //determine if we should attempt subspace expansion.
            
            size_type max_dimension = A1().shape(0);
            size_type curr_dim = A1().hrank();

            auto A2tens = A2().as_rank_3(mode);
            //if the matricisation of the tensor we are currently trying to expand is square then we don't even attempt to expand
            if(A1().shape(1) >= max_dimension){return false;}

            size_type n_unocc = get_nunoccupied(pops);
            if(m_only_apply_when_no_unoccupied && n_unocc >= m_minimum_unoccupied){return false;}

            //get the maximum number of elements that we can add
            size_type max_add2 = max_dimension - A1().shape(1);

            //we can add min(max_add, max_add2) elements
            max_add = max_add < max_add2 ? max_add : max_add2;

            //now determine the number of unoccupied functions, and using this determine the minimum number of terms we need to add
            size_type required_terms = 0;
            if(n_unocc < m_minimum_unoccupied)
            {
                required_terms = m_minimum_unoccupied - n_unocc;
            }
            //and limit the required number of terms to the maximum number of terms we can add
            if(required_terms > max_add){required_terms = max_add;}
    
            size_type nadd = 0;
            //if the other tensor we are expanding is not a square tensor then the two-site algorithm will provide a sensible search direction 
            if(A2tens.shape(1) < A2tens.shape(0)*A2tens.shape(2) && A2tens.shape(1) != 1)
            {
                //now resize the onesite buffer objects used for evaluation of the two site (projected energy) so that everything is the correct size
                size_type nterms = twosite::get_nterms(h()); 

                ASSERT(m_2s_1.size() >= nterms, "Unable to store temporaries needed for computing the two-site energy.  The buffers are not sufficient.");
                CALL_AND_HANDLE(resize_two_site_energy_buffers(nterms, A1(), A2(), mode), "Failed to resize the objects needed to compute the two-site energy.");
                CALL_AND_HANDLE(m_inds.resize(nterms), "Failed to resize indices array.");

                twosite::set_indices(h(), m_inds); 

                //compute the one site objects that are used to compute the two site projected energy
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_upper(A2, h, m_2s_2, base_type::m_HA, base_type::m_temp, base_type::m_temp2, m_inds, true), 
                                "Failed to construct the component of the two site energy acting on the upper site.");
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_lower(A1, h, op, m_2s_1, base_type::m_HA, base_type::m_temp, m_inds, r, true), 
                                "Failed to construct the component of the two site energy acting on the lower site.");
                        

                //now we compute the singular value using the sparse functions
                CALL_AND_HANDLE(m_twosite.generate_orthogonal_trial_vector(A1(), A1().nmodes(), m_rvec), "Failed to generate othrogonal tensor.");

                //firs go ahead and generate random r vector
                size_type maxkrylov_dim = m_krylov_dim;
                if(m_rvec.size() < maxkrylov_dim){maxkrylov_dim = m_rvec.size();}

                bool mconjm = false;
                real_type scale_factor = 1;

                //computes U but stored with its columns as rows - e.g. this is U^T.  E.g. the singular vectors are currently the rows of m_U
                CALL_AND_HANDLE(m_integ.eigs(m_rvec, m_S, m_U, scale_factor, m_neigenvalues, maxkrylov_dim, 1e-4, m_twosite, m_2s_1, m_2s_2, m_trvec, m_trvec2, base_type::m_temp[0], mconjm), "Failed to compute sparse svd.");

                for(size_type i = 0; i < m_S.size(); ++i)
                {
                    real_type sv = 0;
                    //check if any of the dominant svds of the Hamiltonian acting on the twosite coefficient tensor are occupied through the half step
                    if(std::real(m_S(i, i)) > 0){sv = std::sqrt(std::real(m_S(i, i)))*(m_dt)/2.0;}
                    //std::cerr << i << " " << m_S(i, i) << " " << sv <<  " " << m_spawning_threshold << std::endl;
                    if(!std::isnan(sv))
                    {
                        if(sv > m_spawning_threshold){++nadd;}
                    }
                }

                if(nadd > max_add){nadd = max_add;}
                //here we need to compute the eigenstates of the 
                if(nadd != 0)
                {
                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(twosite::expand_tensor(A2(), base_type::m_temp[0], mode, nadd, m_dim), "Failed to expand A2 tensor.");

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, base_type::m_temp[0], nadd), "Failed to expand R matrix.");

                    //expand the A1 tensor around the index pointing to the root zero padding
                    CALL_AND_HANDLE(twosite::expand_tensor(A1(), base_type::m_temp[0], A1.nmodes(), m_U, nadd, m_dim), "Failed to expand A1 tensor.");

                    //resize all of the working buffers. 
                    CALL_AND_HANDLE(base_type::resize_working_buffers(A1().shape(0), A1().shape(1)), "Failed to resize working arrays.");

                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                    subspace_expanded = true;
                    ++m_twosite_expansions;
                }
            }
            //otherwise we just use random one-site tensor expansion
            else
            {
                while(nadd < required_terms)
                {
                    size_type add = 1;
                    CALL_AND_HANDLE(m_twosite.generate_orthogonal_trial_vector(A1(), A1().nmodes(), m_rvec), "Failed to generate othrogonal tensor.");
                    CALL_AND_HANDLE(m_U.resize(1, m_rvec.size()), "Failed to resize V array.");
                    CALL_AND_HANDLE(m_U[0] = m_rvec, "Failed to copy random vector to V");

                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(twosite::expand_tensor(A2(), base_type::m_temp[0], mode, add, m_dim), "Failed to expand A2 tensor.");

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, base_type::m_temp[0], add), "Failed to expand R matrix.");

                    //expand the A1 tensor around the index pointing to the root zero padding
                    CALL_AND_HANDLE(twosite::expand_tensor(A1(), base_type::m_temp[0], A1.nmodes(), m_U, add, m_dim), "Failed to expand A1 tensor.");

                    //resize all of the working buffers. 
                    CALL_AND_HANDLE(base_type::resize_working_buffers(A1().shape(0), A1().shape(1)), "Failed to resize working arrays.");

                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                    ++nadd;
                    subspace_expanded = true;
                }
                ++m_onesite_expansions;
            }
                          
            return subspace_expanded;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing up the tree.");
        }

    }

public:
    const size_type& Nonesite() const{return m_onesite_expansions;}
    const size_type& Ntwosite() const{return m_twosite_expansions;}

protected:
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
    void resize_two_site_energy_buffers(size_type n, const hdata& a1, const hdata& a2, size_type mode)
    {
        auto atens = a2.as_rank_3(mode);
        for(size_type _n = 0; _n < n; ++_n)
        {
            CALL_AND_HANDLE(m_2s_1[_n].resize(a1.shape(0), a1.shape(1)), "Failed to reshape two-site energy buffer object.");
            CALL_AND_HANDLE(m_2s_2[_n].resize(atens.shape(0), atens.shape(1), atens.shape(2)), "Failed to reshape two-site energy buffer object.");
        }
    }

    size_type get_nunoccupied(const dmat_type&  pops)
    {
        size_type nunocc = 0;
        for(size_type i = 0; i  < pops.size(); ++i)
        {
            if(pops(i, i)/pops(0, 0) < m_unoccupied_threshold){++nunocc;}
        }
        return nunocc;
    }

protected:
    full_hamiltonian_action<T, backend> fha;        //used for evolving the r matrix under the full Hamiltonian
    coefficient_evolution_leaf<T, backend> cel;     //used for evolving the coefficient tensors of leaf nodes
    coefficient_evolution_branch<T, backend> ceb;   //used for evolving the coefficient tensors of branch nodes
    two_site_variations<T, backend> m_twosite;

    vec_type m_rvec;
    mat_type m_trvec;
    mat_type m_trvec2;

    //the krylov subspace engine
    integ_type m_integ;

    std::vector<size_type> m_dim;
    size_type m_krylov_dim;
    size_type m_num_threads;

    //add in a second set of indices
    linalg::vector<size_type> m_inds;
    triad_type m_2s_1;
    std::vector<linalg::tensor<T, 3, backend>> m_2s_2;

    real_type m_dt;
    real_type m_t;
    T m_coeff;
    real_type m_spawning_threshold;
    real_type m_unoccupied_threshold;
    real_type m_eigenvalue_tolerance;
    size_type m_minimum_unoccupied;
    size_type m_neigenvalues;
    size_type m_onesite_expansions;
    size_type m_twosite_expansions;

    mat_type m_U;
    linalg::diagonal_matrix<T, backend> m_S;
    mat_type m_V;
    bool m_only_apply_when_no_unoccupied;
};  //class projector_splitting_engine

}   //namespace ttns

#endif  //HTUCKER_SUBSPACE_PROJECTOR_SPLITTING_ENGINE_HPP//

